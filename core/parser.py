from __future__ import annotations

import ast
import importlib.util
import logging
import math
import operator
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models import ExtendedSource, Lens, PointSource, RangeSpec, Source
from runs import EnsembleRun, GlaficRun


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableRef:
    name: str


@dataclass(frozen=True)
class ArithmeticExpression:
    expression: str
    transformed_expression: str
    placeholders: dict[str, Any]
    function_placeholders: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class DeferredExpression:
    expression_text: str


class DerivedRangeSpec(RangeSpec):
    def __init__(
        self,
        base_range: RangeSpec,
        start: float,
        end: float,
        count: int,
        expression_text: str | None = None,
        base_ranges: tuple[RangeSpec, ...] | None = None,
    ):
        super().__init__(start=start, end=end, count=count)
        self.base_range = base_range
        self.expression_text = expression_text
        self.base_ranges = base_ranges if base_ranges is not None else (base_range,)

    def evaluate_at(self, sample_value: float | dict[int, float], variables: dict[str, Any]) -> float:
        if not self.expression_text:
            return float(self.start)
        if isinstance(sample_value, dict):
            return float(_evaluate_expression_text(self.expression_text, variables, axis_samples=sample_value))
        return float(_evaluate_expression_text(self.expression_text, variables, sample_value=sample_value))


def _root_base_range(value: Any) -> RangeSpec | None:
    if isinstance(value, DerivedRangeSpec):
        return _root_base_range(value.base_range)
    if isinstance(value, RangeSpec):
        return value
    return None


@dataclass
class ExpansionAxis:
    label: str
    range_spec: RangeSpec
    log_space: bool
    occurrences: list[tuple[str, Any, Any]]


KNOWN_CONFIG_KEYS = {
    "omega",
    "lambda",
    "weos",
    "hubble",
    "prefix",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "pix_ext",
    "pix_poi",
    "maxlev",
    "galfile",
    "srcfile",
    "ran_seed",
    "flag_extnorm",
    "flatfix",
    "obs_gain",
    "obs_ncomb",
    "obs_readnoise",
    "psfconv_size",
    "seeing_sub",
    "flag_outpot",
    "addwcs",
    "wcs_ra0",
    "wcs_dec0",
}


LOG_SPACED_LENS_PARAMETERS = {
    "point": {1},
    "sie": {1},
    "jaffe": {1},
    "pow": {},
    "powpot": {},
    "sers": {1},
    "gauss": {1},
    "serspot": {1},
    "hern": {1},
    "ahern": {1},
    "hernpot": {1},
    "nfw": {1},
    "anfw": {1},
    "nfwpot": {1},
    "gnfw": {1},
    "gnfwpot": {1},
    "tnfw": {1},
    "tnfwpot": {1},
    "acnfw": {1},
    "ein": {1},
    "einpot": {1},
    "gaupot": {6},
    "gals": {1},
    "crline": {7},
}


BLOCKED_COMMANDS = {
    "optimize",
    "optextend",
    "optpoint",
    "c2calc",
    "randomize",
    "opt",
    "varyone",
    "varytwo",
    "varyzs",
    "varycosmo",
    "mcmc",
    "readobs",
    "parprior",
    "mapprior",
    "resetopt",
    "start_setopt",
    "end_setopt",
}


MAIN_ROOT = Path(__file__).resolve().with_name("main.py").parent
_FUNCTION_CALL_PATTERN = re.compile(r"(?:(!)([A-Za-z_][A-Za-z0-9_]*)|(\bnp))\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@lru_cache(maxsize=1)
def _main_root_modules() -> dict[str, Path]:
    return {
        module_path.stem: module_path
        for module_path in MAIN_ROOT.glob("*.py")
        if module_path.is_file()
    }


@lru_cache(maxsize=None)
def _load_main_root_module(module_name: str) -> Any:
    module_path = _main_root_modules().get(module_name)
    if module_path is None:
        raise ValueError(f"Module '{module_name}' was not found in {MAIN_ROOT}")

    module_spec = importlib.util.spec_from_file_location(f"forward_lensing_sim.{module_name}", module_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Failed to load module '{module_name}' from {module_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _resolve_external_function(module_name: str, function_name: str) -> Callable[..., Any]:
    if module_name == "np":
        func = getattr(np, function_name, None)
        if func is None or not callable(func):
            raise ValueError(f"NumPy function 'np.{function_name}' was not found")
        return func
    module = _load_main_root_module(module_name)
    if not hasattr(module, function_name):
        raise ValueError(f"Function '{function_name}' was not found in module '{module_name}'")

    func = getattr(module, function_name)
    if not callable(func):
        raise ValueError(f"'{module_name}.{function_name}' is not callable")
    return func


def _transform_function_calls(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    function_placeholders: dict[str, tuple[str, str]] = {}

    def replace_function_call(match: re.Match[str]) -> str:
        module_name = match.group(2) if match.group(1) else match.group(3)
        function_name = match.group(4)
        placeholder = f"__func__{len(function_placeholders)}"
        function_placeholders[placeholder] = (module_name, function_name)
        return f"{placeholder}("

    transformed = _FUNCTION_CALL_PATTERN.sub(replace_function_call, text)
    return transformed, function_placeholders


def _loud_warning(message: str) -> None:
    if logging.getLogger().handlers:
        logger.warning(message)
    else:
        print(f"[WARNING] {message}", file=sys.stderr)


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _tokenize(line: str) -> list[str]:
    return re.findall(r"\{[^{}]*\}|[^\s]+", line)


def _parse_value(token: str) -> Any:
    text = token.strip()

    if text.startswith("{") and text.endswith("}"):
        content = text[1:-1]
        parts = [part.strip() for part in content.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid range token: {token}")
        return RangeSpec(float(parts[0]), float(parts[1]), int(float(parts[2])))

    if "," in text:
        if _looks_like_expression(text):
            return _parse_arithmetic_expression(text)
        raise ValueError(f"Invalid comma-separated token outside braces: {token}")

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    if text.startswith("$") and len(text) > 1:
        if re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", text):
            return VariableRef(text[1:])
        if _looks_like_expression(text):
            return _parse_arithmetic_expression(text)

    if text.startswith("!") and _looks_like_expression(text):
        return _parse_arithmetic_expression(text)

    try:
        integer_value = int(text)
        if str(integer_value) == text or text in {f"+{integer_value}", f"-{abs(integer_value)}"}:
            return integer_value
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        if _looks_like_expression(text):
            try:
                return _parse_arithmetic_expression(text)
            except ValueError:
                return text
        return text


def _looks_like_expression(text: str) -> bool:
    if not text:
        return False
    if text.startswith(("'", '"')):
        return False
    if text.startswith("{") and text.endswith("}"):
        return False
    if text.lower() in {"true", "false"}:
        return False
    if text.startswith("$"):
        return True
    if text.startswith("!"):
        return True
    if text.startswith("."):
        return False

    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return False

    if re.search(r"\$|[()+\-*/%]|\*\*|//", text):
        return True

    if "/" in text or "\\" in text:
        if re.search(r"\s", text):
            return False
        if re.fullmatch(r"[A-Za-z0-9._/-]+", text):
            return False
        return True

    return False


def _parse_arithmetic_expression(text: str) -> ArithmeticExpression:
    placeholders: dict[str, Any] = {}

    def replace_range_tokens(match: re.Match[str]) -> str:
        name = f"__range_{len(placeholders)}"
        placeholders[name] = _parse_value(match.group(0))
        return name

    transformed_expression = re.sub(r"\{[^{}]*\}", replace_range_tokens, text)
    transformed_expression = re.sub(r"\$(\w+)", r"__var__\1", transformed_expression)
    transformed_expression, function_placeholders = _transform_function_calls(transformed_expression)
    try:
        ast.parse(transformed_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {text}") from exc

    return ArithmeticExpression(
        expression=text,
        transformed_expression=transformed_expression,
        placeholders=placeholders,
        function_placeholders=function_placeholders,
    )


def _apply_arithmetic_to_range(op: Callable[[float, float], float], left: Any, right: Any) -> Any:
    if isinstance(left, RangeSpec) and isinstance(right, RangeSpec):
        if left.count != right.count:
            raise ValueError("RangeSpec operands must have matching counts")
        return RangeSpec(
            start=float(op(left.start, right.start)),
            end=float(op(left.end, right.end)),
            count=left.count,
        )

    if isinstance(left, RangeSpec):
        if not isinstance(right, (int, float)):
            raise ValueError("RangeSpec arithmetic requires numeric operands")
        return DerivedRangeSpec(
            base_range=_root_base_range(left) or left,
            start=float(op(left.start, right)),
            end=float(op(left.end, right)),
            count=left.count,
        )

    if isinstance(right, RangeSpec):
        if not isinstance(left, (int, float)):
            raise ValueError("RangeSpec arithmetic requires numeric operands")
        return DerivedRangeSpec(
            base_range=_root_base_range(right) or right,
            start=float(op(left, right.start)),
            end=float(op(left, right.end)),
            count=right.count,
        )

    return op(left, right)


def _coerce_to_numeric(value: Any) -> float:
    if isinstance(value, RangeSpec):
        if value.count == 1:
            return float(value.start)
        raise ValueError("RangeSpec cannot be coerced to a single numeric value")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Cannot coerce value {value!r} to numeric")


def _evaluate_expression_text(
    expression_text: str,
    variables: dict[str, Any],
    *,
    sample_value: float | None = None,
    axis_samples: dict[int, float] | None = None,
) -> Any:
    transformed_expression = re.sub(r"\$(\w+)", r"__var__\1", expression_text)
    transformed_expression, function_placeholders = _transform_function_calls(transformed_expression)
    try:
        tree = ast.parse(transformed_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression_text}") from exc

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id.startswith("__var__"):
                variable_name = node.id[len("__var__") :]
                value = variables.get(variable_name)
                if value is None:
                    raise ValueError(f"Undefined variable '{variable_name}'")
                if isinstance(value, RangeSpec):
                    base_range = _root_base_range(value) or value
                    if axis_samples is not None:
                        axis_sample = axis_samples.get(id(base_range))
                        if axis_sample is not None:
                            return float(axis_sample)
                    if sample_value is not None:
                        return float(sample_value)
                    if value.count == 1:
                        return float(value.start)
                    return float(value.start)
                return value
            if node.id in function_placeholders:
                module_name, function_name = function_placeholders[node.id]
                return _resolve_external_function(module_name, function_name)
            if node.id in {"abs", "min", "max", "pow", "round"}:
                return getattr(__import__("builtins"), node.id)
            raise ValueError(f"Unsupported arithmetic name: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return operator.add(left, right)
            if isinstance(node.op, ast.Sub):
                return operator.sub(left, right)
            if isinstance(node.op, ast.Mult):
                return operator.mul(left, right)
            if isinstance(node.op, ast.Div):
                return operator.truediv(left, right)
            if isinstance(node.op, ast.FloorDiv):
                return operator.floordiv(left, right)
            if isinstance(node.op, ast.Mod):
                return operator.mod(left, right)
            if isinstance(node.op, ast.Pow):
                return operator.pow(left, right)
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")
            func = evaluate(node.func)
            args = [evaluate(arg) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported arithmetic syntax: {ast.dump(node)}")

    return evaluate(tree.body)


def _evaluate_arithmetic_expression(
    expression: ArithmeticExpression,
    raw_variables: dict[str, Any],
    resolved_variables: dict[str, Any],
    stack: set[str],
    *,
    defer_functions: bool = False,
) -> Any:
    transformed_expression = expression.transformed_expression
    try:
        tree = ast.parse(transformed_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression.expression}") from exc

    def evaluate(node: ast.AST, values: dict[str, float] | None = None) -> Any:
        values = values if values is not None else {}

        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id.startswith("__var__"):
                variable_name = node.id[len("__var__") :]
                if variable_name in values:
                    return values[variable_name]
                resolved_value = _resolve_variable(variable_name, raw_variables, resolved_variables, stack)
                if isinstance(resolved_value, RangeSpec):
                    return resolved_value
                return resolved_value
            if node.id.startswith("__range_"):
                return expression.placeholders[node.id]
            if node.id in expression.function_placeholders:
                module_name, function_name = expression.function_placeholders[node.id]
                return _resolve_external_function(module_name, function_name)
            if node.id in {"abs", "min", "max", "pow", "round"}:
                return getattr(__import__("builtins"), node.id)
            raise ValueError(f"Unsupported arithmetic name: {node.id}")
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand, values)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left, values)
            right = evaluate(node.right, values)
            if isinstance(node.op, ast.Add):
                return operator.add(left, right)
            if isinstance(node.op, ast.Sub):
                return operator.sub(left, right)
            if isinstance(node.op, ast.Mult):
                return operator.mul(left, right)
            if isinstance(node.op, ast.Div):
                return operator.truediv(left, right)
            if isinstance(node.op, ast.FloorDiv):
                return operator.floordiv(left, right)
            if isinstance(node.op, ast.Mod):
                return operator.mod(left, right)
            if isinstance(node.op, ast.Pow):
                return operator.pow(left, right)
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")
            func = evaluate(node.func, values)
            args = [evaluate(arg, values) for arg in node.args]
            return func(*args)
        raise ValueError(f"Unsupported arithmetic syntax: {ast.dump(node)}")

    def collect_variable_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        if isinstance(node, ast.Name) and node.id.startswith("__var__"):
            names.add(node.id[len("__var__") :])
        for child in ast.iter_child_nodes(node):
            names.update(collect_variable_names(child))
        return names

    referenced_variables = sorted(collect_variable_names(tree.body))
    base_ranges: list[RangeSpec] = []
    seen_base_range_ids: set[int] = set()
    has_deferred_variable = False
    for variable_name in referenced_variables:
        value = _resolve_variable(variable_name, raw_variables, resolved_variables, stack)
        has_deferred_variable = has_deferred_variable or isinstance(value, DeferredExpression)
        if isinstance(value, RangeSpec):
            base_range = _root_base_range(value) or value
            base_range_id = id(base_range)
            if base_range_id not in seen_base_range_ids:
                base_ranges.append(base_range)
                seen_base_range_ids.add(base_range_id)

    if defer_functions and (expression.function_placeholders or has_deferred_variable) and not base_ranges:
        return DeferredExpression(expression.expression)

    if base_ranges:
        sampled_lists = [
            [float(value) for value in base_range.values(log_space=False)]
            for base_range in base_ranges
        ]
        derived_values: list[float] = []

        for combination in product(*sampled_lists):
            axis_samples = {id(base_range): float(sample_value) for base_range, sample_value in zip(base_ranges, combination)}
            values: dict[str, float] = {}
            for variable_name in referenced_variables:
                variable_value = _resolve_variable(variable_name, raw_variables, resolved_variables, stack)
                if isinstance(variable_value, RangeSpec):
                    base_range = _root_base_range(variable_value) or variable_value
                    values[variable_name] = float(axis_samples.get(id(base_range), base_range.start))
                else:
                    values[variable_name] = float(variable_value)
            derived_values.append(float(evaluate(tree.body, values)))

        base_range = base_ranges[0]
        total_count = math.prod(len(values) for values in sampled_lists)
        return DerivedRangeSpec(
            base_range=base_range,
            start=float(min(derived_values)),
            end=float(max(derived_values)),
            count=total_count,
            expression_text=expression.expression,
            base_ranges=tuple(base_ranges),
        )

    return evaluate(tree.body)


def _parse_variable_definition(line: str) -> tuple[str, Any] | None:
    match = re.match(r"^\$([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", line)
    if match is None:
        return None

    return match.group(1), _parse_value(match.group(2))


def _parse_string_token(token: str) -> str:
    return token.strip().strip('"').strip("'")


def _parse_label_value_directive(tokens: list[str], directive: str) -> tuple[str, Any]:
    if len(tokens) < 3:
        raise ValueError(f"Invalid {directive} directive: {' '.join(tokens)}")

    label = _parse_string_token(tokens[1])
    if not label:
        raise ValueError(f"{directive.upper()} directive is missing a label: {' '.join(tokens)}")

    value_text = " ".join(tokens[2:])
    return label, _parse_value(value_text)


def _parse_sfh_profiles(lines: list[str]) -> dict[str, dict[str, Any]]:
    sfh_profiles: dict[str, dict[str, Any]] = {}
    index = 0

    while index < len(lines):
        stripped = _strip_comment(lines[index])
        if not stripped:
            index += 1
            continue

        tokens = _tokenize(stripped)
        if len(tokens) >= 3 and tokens[0].lower() == "sfh" and tokens[-1].lower() == "start":
            label = _parse_string_token(tokens[1])
            if not label:
                raise ValueError(f"SFH block is missing a label: {stripped}")
            if label in sfh_profiles:
                raise ValueError(f"Duplicate SFH block for label '{label}'")

            index += 1
            profile: dict[str, Any] = {}
            while index < len(lines):
                body_line = _strip_comment(lines[index])
                if not body_line:
                    index += 1
                    continue

                body_tokens = _tokenize(body_line)
                if not body_tokens:
                    index += 1
                    continue

                if body_tokens[0].lower() == "end":
                    break

                if body_tokens[0].lower() == "form":
                    if len(body_tokens) != 3:
                        raise ValueError(f"Invalid SFH form line for label '{label}': {body_line}")
                    profile["massformed"] = _parse_value(body_tokens[1])
                    profile["form_time"] = _parse_value(body_tokens[2])
                elif body_tokens[0].lower() == "starburst":
                    if len(body_tokens) != 4:
                        raise ValueError(f"Invalid SFH starburst line for label '{label}': {body_line}")
                    profile["starburst"] = [_parse_value(body_tokens[1]), _parse_value(body_tokens[2]), _parse_value(body_tokens[3])]
                elif body_tokens[0].lower() == "dust":
                    if len(body_tokens) != 4:
                        raise ValueError(f"Invalid SFH dust line for label '{label}': {body_line}")
                    profile["dust"] = [_parse_value(body_tokens[1]), _parse_value(body_tokens[2]), _parse_value(body_tokens[3])]
                else:
                    raise ValueError(f"Unrecognized SFH directive for label '{label}': {body_line}")

                index += 1

            if index >= len(lines):
                raise ValueError(f"Unterminated SFH block for label '{label}'")

            required_keys = {"massformed", "form_time", "starburst", "dust"}
            missing_keys = sorted(required_keys - set(profile))
            if missing_keys:
                raise ValueError(f"SFH block '{label}' is missing required keys: {', '.join(missing_keys)}")

            sfh_profiles[label] = profile
        index += 1

    return sfh_profiles


def _resolve_variable(
    name: str,
    raw_variables: dict[str, Any],
    resolved_variables: dict[str, Any],
    stack: set[str],
    *,
    defer_functions: bool = False,
) -> Any:
    if name in resolved_variables:
        return resolved_variables[name]

    if name in stack:
        raise ValueError(f"Cyclic variable reference detected for '${name}'")

    if name not in raw_variables:
        raise ValueError(f"Undefined variable '${name}'")

    stack.add(name)
    resolved_value = _resolve_value(
        raw_variables[name],
        raw_variables,
        resolved_variables,
        stack,
        defer_functions=defer_functions,
    )
    stack.remove(name)
    resolved_variables[name] = resolved_value
    return resolved_value


def _resolve_value(
    value: Any,
    raw_variables: dict[str, Any],
    resolved_variables: dict[str, Any],
    stack: set[str],
    *,
    defer_functions: bool = False,
) -> Any:
    if isinstance(value, VariableRef):
        return _resolve_variable(
            value.name,
            raw_variables,
            resolved_variables,
            stack,
            defer_functions=defer_functions,
        )

    if isinstance(value, ArithmeticExpression):
        return _evaluate_arithmetic_expression(
            value,
            raw_variables,
            resolved_variables,
            stack,
            defer_functions=defer_functions,
        )

    if isinstance(value, list):
        return [
            _resolve_value(item, raw_variables, resolved_variables, stack, defer_functions=defer_functions)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _resolve_value(item, raw_variables, resolved_variables, stack, defer_functions=defer_functions)
            for item in value
        )

    if isinstance(value, dict):
        return {
            key: _resolve_value(item, raw_variables, resolved_variables, stack, defer_functions=defer_functions)
            for key, item in value.items()
        }

    return value


def _resolve_lens(
    lens: Lens,
    raw_variables: dict[str, Any],
    resolved_variables: dict[str, Any],
    *,
    defer_functions: bool = False,
) -> None:
    lens.kind = _resolve_value(lens.kind, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    lens.redshift = _resolve_value(lens.redshift, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    lens.pos = tuple(
        _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for value in lens.pos
    )
    lens.parameters = [
        _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for value in lens.parameters
    ]
    lens.metadata = {
        key: _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for key, value in lens.metadata.items()
    }


def _resolve_source(
    source: Source,
    raw_variables: dict[str, Any],
    resolved_variables: dict[str, Any],
    *,
    defer_functions: bool = False,
) -> None:
    source.kind = _resolve_value(source.kind, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    source.redshift = _resolve_value(source.redshift, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    source.pos = tuple(
        _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for value in source.pos
    )
    source.parameters = [
        _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for value in source.parameters
    ]
    source.metadata = {
        key: _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        for key, value in source.metadata.items()
    }

    if isinstance(source, ExtendedSource):
        source.flux = _resolve_value(source.flux, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        source.ellipticity = _resolve_value(source.ellipticity, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        source.PA = _resolve_value(source.PA, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
        source.kwargs = {
            key: _resolve_value(value, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
            for key, value in source.kwargs.items()
        }


def _resolve_run_variables(
    run: GlaficRun,
    raw_variables: dict[str, Any],
    *,
    defer_functions: bool = False,
) -> dict[str, Any]:
    resolved_variables: dict[str, Any] = {}
    for variable_name in raw_variables:
        _resolve_variable(
            variable_name,
            raw_variables,
            resolved_variables,
            set(),
            defer_functions=defer_functions,
        )

    run.cfg = _resolve_value(run.cfg, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    run.psf = _resolve_value(run.psf, raw_variables, resolved_variables, set(), defer_functions=defer_functions)
    run.variables = dict(resolved_variables)

    for lens in run.lenses:
        _resolve_lens(lens, raw_variables, resolved_variables, defer_functions=defer_functions)

    for source in run.sources:
        _resolve_source(source, raw_variables, resolved_variables, defer_functions=defer_functions)

    run.mass_to_light_ratios = {
        label: _resolve_value(value, raw_variables, resolved_variables, set())
        for label, value in run.mass_to_light_ratios.items()
    }
    run.luminosities = {
        label: _resolve_value(value, raw_variables, resolved_variables, set())
        for label, value in run.luminosities.items()
    }
    run.sfh_profiles = {
        label: _resolve_value(profile, raw_variables, resolved_variables, set())
        for label, profile in run.sfh_profiles.items()
    }
    run.commands = [
        _resolve_command_line(command_line, raw_variables, resolved_variables)
        for command_line in run.commands
    ]

    return resolved_variables


def _resolve_command_line(command_line: str, raw_variables: dict[str, Any], resolved_variables: dict[str, Any]) -> str:
    tokens = re.findall(r"\{[^{}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|[^\s]+", command_line)
    resolved_tokens: list[str] = []
    for token in tokens:
        if token.startswith("$") and len(token) > 1:
            variable_name = token[1:]
            if variable_name in raw_variables:
                resolved = _resolve_variable(variable_name, raw_variables, resolved_variables, set())
                if isinstance(resolved, RangeSpec):
                    resolved_tokens.append(token)
                else:
                    resolved_tokens.append(str(resolved))
            else:
                resolved_tokens.append(token)
        else:
            resolved_tokens.append(token)
    return " ".join(resolved_tokens)


def _materialize_derived_range_value(value: DerivedRangeSpec, variables: dict[str, Any], axis_samples: dict[int, float] | None = None) -> float:
    if value.expression_text is None:
        return float(value.start)
    if axis_samples is not None:
        return float(_evaluate_expression_text(value.expression_text, variables, axis_samples=axis_samples))
    return float(_evaluate_expression_text(value.expression_text, variables))


def _materialize_deferred_expression(value: DeferredExpression, variables: dict[str, Any], axis_samples: dict[int, float]) -> Any:
    materialized_variables = dict(variables)
    for variable_name, variable_value in materialized_variables.items():
        if isinstance(variable_value, DeferredExpression):
            materialized_variables[variable_name] = _evaluate_expression_text(
                variable_value.expression_text,
                materialized_variables,
                axis_samples=axis_samples,
            )
    return _evaluate_expression_text(value.expression_text, materialized_variables, axis_samples=axis_samples)


def _materialize_range_bound_value(value: Any, axis_samples: dict[int, float], variables: dict[str, Any]) -> Any:
    if isinstance(value, DeferredExpression):
        return _materialize_deferred_expression(value, variables, axis_samples)

    if isinstance(value, DerivedRangeSpec):
        base_range = _root_base_range(value.base_range) or value.base_range
        sample_value = axis_samples.get(id(base_range))
        if value.expression_text is not None:
            return _materialize_derived_range_value(value, variables, axis_samples)
        if sample_value is not None:
            return value.evaluate_at(float(sample_value), variables)
        if value.count == 1:
            return float(value.start)
        return value

    if isinstance(value, RangeSpec):
        base_range = _root_base_range(value) or value
        sample_value = axis_samples.get(id(base_range))
        if sample_value is not None:
            return float(sample_value)
        if value.count == 1:
            return float(value.start)
        return value

    if isinstance(value, list):
        return [_materialize_range_bound_value(item, axis_samples, variables) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize_range_bound_value(item, axis_samples, variables) for item in value)
    if isinstance(value, dict):
        return {key: _materialize_range_bound_value(item, axis_samples, variables) for key, item in value.items()}
    return value


def _materialize_command_line_with_variables(command_line: str, variables: dict[str, Any]) -> str:
    tokens = re.findall(r"\{[^{}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|[^\s]+", command_line)
    resolved_tokens: list[str] = []
    for token in tokens:
        if token.startswith("$") and len(token) > 1:
            variable_name = token[1:]
            if variable_name in variables and not isinstance(variables[variable_name], RangeSpec):
                resolved_tokens.append(str(variables[variable_name]))
            else:
                resolved_tokens.append(token)
        else:
            resolved_tokens.append(token)
    return " ".join(resolved_tokens)


def _axis_sample_key_in_run(axis: ExpansionAxis, run: GlaficRun) -> int | None:
    if not axis.occurrences:
        return None
    entity_kind, entity_index, parameter_index = axis.occurrences[0]
    if entity_kind == "lens":
        parameter_value = run.lenses[entity_index].parameters[parameter_index]
    elif entity_kind == "source":
        parameter_value = run.sources[entity_index].parameters[parameter_index]
    elif entity_kind == "luminosity":
        parameter_value = run.luminosities.get(entity_index)
    elif entity_kind == "mass_to_light":
        parameter_value = run.mass_to_light_ratios.get(entity_index)
    else:
        return None
    base_range = _root_base_range(parameter_value)
    if base_range is None:
        return None
    return id(base_range)


def _parse_lens(tokens: list[str]) -> Lens:
    if len(tokens) < 2:
        raise ValueError("Lens definition requires a model name")

    kind = tokens[1]
    allowed_kinds = {
        "pert",
        "clus3",
        "mpole",
        "crline",
        "point",
        "sie",
        "jaffe",
        "pow",
        "powpot",
        "sers",
        "serspot",
        "hern",
        "ahern",
        "hernpot",
        "nfw",
        "anfw",
        "nfwpot",
        "gnfw",
        "gnfwpot",
        "tnfw",
        "tnfwpot",
        "acnfw",
        "ein",
        "einpot",
        "gaupot",
        "gals",
    }
    if kind not in allowed_kinds:
        raise ValueError(f"Unsupported lens model '{kind}'")

    raw_parameters = [_parse_value(token) for token in tokens[2:]]
    if raw_parameters and isinstance(raw_parameters[-1], str):
        metadata_label = raw_parameters.pop()
    else:
        metadata_label = None
    redshift = raw_parameters[0] if raw_parameters else None
    position = (raw_parameters[2], raw_parameters[3]) if len(raw_parameters) >= 4 else (0.0, 0.0)
    metadata = {f"p{index + 1}": value for index, value in enumerate(raw_parameters)}
    if metadata_label is not None:
        metadata["label"] = _parse_string_token(str(metadata_label))
    return Lens(kind=kind, redshift=redshift, pos=position, parameters=raw_parameters, metadata=metadata)


def _parse_extended_source(tokens: list[str]) -> ExtendedSource:
    if len(tokens) < 2:
        raise ValueError("Extended source definition requires a model name")

    kind = tokens[1]
    allowed_kinds = {"gauss", "sersic", "tophat", "moffat", "jaffe", "point", "srcs"}
    if kind not in allowed_kinds:
        raise ValueError(f"Unsupported extended source model '{kind}'")

    raw_parameters = [_parse_value(token) for token in tokens[2:]]
    if raw_parameters and isinstance(raw_parameters[-1], str):
        metadata_label = raw_parameters.pop()
    else:
        metadata_label = None
    redshift = raw_parameters[0] if raw_parameters else None
    flux = raw_parameters[1] if len(raw_parameters) > 1 else None
    position = (raw_parameters[2], raw_parameters[3]) if len(raw_parameters) >= 4 else (0.0, 0.0)
    ellipticity = raw_parameters[4] if len(raw_parameters) > 4 else 0.0
    position_angle = raw_parameters[5] if len(raw_parameters) > 5 else 0.0

    kwargs: dict[str, Any] = {}
    if kind == "sersic" and len(raw_parameters) >= 8:
        kwargs["re"] = raw_parameters[6]
        kwargs["n"] = raw_parameters[7]
    elif kind in {"gauss", "srcs"} and len(raw_parameters) >= 7:
        kwargs["sigma"] = raw_parameters[6]
    elif kind == "tophat" and len(raw_parameters) >= 7:
        kwargs["r0"] = raw_parameters[6]
    elif kind == "moffat" and len(raw_parameters) >= 8:
        kwargs["alpha"] = raw_parameters[6]
        kwargs["beta"] = raw_parameters[7]
    elif kind == "jaffe" and len(raw_parameters) >= 8:
        kwargs["rtrun"] = raw_parameters[6]
        kwargs["rcore"] = raw_parameters[7]

    metadata = {f"p{index + 1}": value for index, value in enumerate(raw_parameters)}
    if metadata_label is not None:
        metadata["label"] = _parse_string_token(str(metadata_label))
    return ExtendedSource(
        kind=kind,
        redshift=redshift,
        pos=position,
        flux=flux,
        ellipticity=ellipticity,
        PA=position_angle,
        kwargs=kwargs,
        parameters=raw_parameters,
        metadata=metadata,
    )


def _parse_point_source(tokens: list[str]) -> PointSource:
    if len(tokens) < 4:
        raise ValueError("Point source definition requires redshift and two coordinates")

    redshift = _parse_value(tokens[1])
    position = (_parse_value(tokens[2]), _parse_value(tokens[3]))
    metadata = {"p1": redshift, "p2": position[0], "p3": position[1]}
    return PointSource(redshift=redshift, pos=position, metadata=metadata)


def _parse_psf(tokens: list[str]) -> dict[str, Any]:
    return {f"p{index + 1}": _parse_value(token) for index, token in enumerate(tokens[1:])}


def _config_key_from_tokens(tokens: list[str]) -> tuple[str, int]:
    if not tokens:
        raise ValueError("Missing configuration key")

    if len(tokens) >= 2:
        combined_two = f"{tokens[0]}_{tokens[1]}".lower()
        if combined_two in KNOWN_CONFIG_KEYS:
            return combined_two, 2

    single = tokens[0].lower().replace(" ", "_")
    if single in KNOWN_CONFIG_KEYS:
        return single, 1

    if len(tokens) >= 3:
        combined_three = f"{tokens[0]}_{tokens[1]}_{tokens[2]}".lower()
        if combined_three in KNOWN_CONFIG_KEYS:
            return combined_three, 3

    return single, 1


def _reject_blocked_command(command_line: str) -> None:
    tokens = _tokenize(command_line)
    if not tokens:
        return

    command_name = tokens[0].lower()
    if command_name in BLOCKED_COMMANDS:
        _loud_warning(f"Blocked optimization/MCMC command: {command_line}")
        raise ValueError(f"Optimization/MCMC commands are not supported: {command_line}")

    if command_name == "opt" and len(tokens) > 1 and tokens[1].lower() == "explore":
        _loud_warning(f"Blocked optimization/MCMC command: {command_line}")
        raise ValueError(f"Optimization/MCMC commands are not supported: {command_line}")


def _apply_lens_parameter(lens: Lens, parameter_index: int, value: Any) -> None:
    lens.parameters[parameter_index] = value
    lens.metadata[f"p{parameter_index + 1}"] = value
    if parameter_index == 0:
        lens.redshift = value


def _apply_lens_parameter_from_variable(lens: Lens, parameter_index: int, raw_value: Any, resolved_variables: dict[str, Any]) -> None:
    resolved_value = _resolve_value(raw_value, raw_variables, resolved_variables, set()) if False else None
    _apply_lens_parameter(lens, parameter_index, raw_value)


def _apply_source_parameter(source: Source, parameter_index: int, value: Any) -> None:
    source.parameters[parameter_index] = value
    source.metadata[f"p{parameter_index + 1}"] = value

    if parameter_index == 0:
        source.redshift = value
    elif parameter_index == 1 and isinstance(source, ExtendedSource):
        source.flux = value
    elif parameter_index == 2:
        source.pos = (value, source.pos[1])
    elif parameter_index == 3:
        source.pos = (source.pos[0], value)
    elif parameter_index == 4 and isinstance(source, ExtendedSource):
        source.ellipticity = value
    elif parameter_index == 5 and isinstance(source, ExtendedSource):
        source.PA = value
    elif isinstance(source, ExtendedSource):
        extra_index = parameter_index - 6
        if source.kind == "sersic":
            if extra_index == 0:
                source.kwargs["re"] = value
            elif extra_index == 1:
                source.kwargs["n"] = value
        elif source.kind in {"gauss", "srcs"} and extra_index == 0:
            source.kwargs["sigma"] = value
        elif source.kind == "tophat" and extra_index == 0:
            source.kwargs["r0"] = value
        elif source.kind == "moffat":
            if extra_index == 0:
                source.kwargs["alpha"] = value
            elif extra_index == 1:
                source.kwargs["beta"] = value
        elif source.kind == "jaffe":
            if extra_index == 0:
                source.kwargs["rtrun"] = value
            elif extra_index == 1:
                source.kwargs["rcore"] = value


def _default_axis_label(entity_kind: str, entity_index: int, parameter_index: int) -> str:
    if entity_kind == "lens":
        labels = {
            0: "z_lens",
            1: "Mtot",
            2: "lensx",
            3: "lensy",
            4: "ellip",
            5: "PA",
            6: "p7",
            7: "p8",
        }
    else:
        labels = {
            0: "z_source",
            1: "ftot",
            2: "sourcex",
            3: "sourcey",
            4: "ellip",
            5: "PA",
            6: "p7",
            7: "p8",
        }

    return labels.get(parameter_index, f"{entity_kind}{entity_index + 1}_p{parameter_index + 1}")


def _is_log_spaced_lens_parameter(lens_kind: str, parameter_index: int) -> bool:
    return parameter_index in LOG_SPACED_LENS_PARAMETERS.get(lens_kind, set())


def _sample_axis_value(axis: ExpansionAxis, rng: np.random.Generator) -> float:
    start = float(axis.range_spec.start)
    end = float(axis.range_spec.end)
    if axis.log_space:
        if start <= 0.0 or end <= 0.0:
            raise ValueError(f"Log-spaced random sampling requires positive values for axis '{axis.label}'")
        return 10.0 ** rng.uniform(np.log10(start), np.log10(end))
    return rng.uniform(start, end)


def _expand_run(
    run: GlaficRun,
    *,
    mode: str = "interval",
    count: int | None = None,
) -> EnsembleRun | GlaficRun:
    axes_by_object_id: dict[int, ExpansionAxis] = {}

    def _axis_key(parameter: Any) -> int:
        base_range = _root_base_range(parameter)
        if base_range is not None:
            return id(base_range)
        return id(parameter)

    for lens_index, lens in enumerate(run.lenses):
        for parameter_index, parameter in enumerate(lens.parameters):
            if isinstance(parameter, RangeSpec):
                axis_key = _axis_key(parameter)
                axis = axes_by_object_id.get(axis_key)
                log_space = _is_log_spaced_lens_parameter(str(lens.kind), parameter_index)
                if axis is None:
                    axes_by_object_id[axis_key] = ExpansionAxis(
                        label=_default_axis_label("lens", lens_index, parameter_index),
                        range_spec=_root_base_range(parameter) or parameter,
                        log_space=log_space,
                        occurrences=[("lens", lens_index, parameter_index)],
                    )
                else:
                    axis.log_space = axis.log_space or log_space
                    axis.occurrences.append(("lens", lens_index, parameter_index))

    for source_index, source in enumerate(run.sources):
        for parameter_index, parameter in enumerate(source.parameters):
            if isinstance(parameter, RangeSpec):
                axis_key = _axis_key(parameter)
                axis = axes_by_object_id.get(axis_key)
                if axis is None:
                    axes_by_object_id[axis_key] = ExpansionAxis(
                        label=_default_axis_label("source", source_index, parameter_index),
                        range_spec=_root_base_range(parameter) or parameter,
                        log_space=False,
                        occurrences=[("source", source_index, parameter_index)],
                    )
                else:
                    axis.occurrences.append(("source", source_index, parameter_index))

    for label, luminosity_value in run.luminosities.items():
        if isinstance(luminosity_value, RangeSpec):
            axis_key = _axis_key(luminosity_value)
            axis = axes_by_object_id.get(axis_key)
            if axis is None:
                axes_by_object_id[axis_key] = ExpansionAxis(
                    label=f"lum_{label}",
                    range_spec=_root_base_range(luminosity_value) or luminosity_value,
                    log_space=False,
                    occurrences=[("luminosity", label, 0)],
                )
            else:
                axis.occurrences.append(("luminosity", label, 0))

    for label, ml_value in run.mass_to_light_ratios.items():
        if isinstance(ml_value, RangeSpec):
            axis_key = _axis_key(ml_value)
            axis = axes_by_object_id.get(axis_key)
            if axis is None:
                axes_by_object_id[axis_key] = ExpansionAxis(
                    label=f"ml_{label}",
                    range_spec=_root_base_range(ml_value) or ml_value,
                    log_space=False,
                    occurrences=[("mass_to_light", label, 0)],
                )
            else:
                axis.occurrences.append(("mass_to_light", label, 0))

    axes = list(axes_by_object_id.values())
    has_deferred_functions = any(
        isinstance(value, DeferredExpression)
        for value in (
            *run.variables.values(),
            *run.cfg.values(),
            *run.psf.values(),
            *(parameter for lens in run.lenses for parameter in lens.parameters),
            *(parameter for source in run.sources for parameter in source.parameters),
            *(value for source in run.sources if isinstance(source, ExtendedSource) for value in source.kwargs.values()),
            *run.mass_to_light_ratios.values(),
            *run.luminosities.values(),
        )
    )
    if not axes and (mode != "rng" or not has_deferred_functions):
        return run

    if mode == "rng":
        sample_count = int(count) if count is not None else 100
        if sample_count < 1:
            raise ValueError("MODE rng requires a positive integer sample count")

        rng = np.random.default_rng()
        if isinstance(run.cfg.get("ran_seed"), (int, float)) and int(run.cfg["ran_seed"]) != -1:
            rng = np.random.default_rng(int(run.cfg["ran_seed"]))

        expanded_runs: list[GlaficRun] = []
        for _ in range(sample_count):
            expanded_run = deepcopy(run)
            axis_samples: dict[int, float] = {}
            for axis in axes:
                sample_value = _sample_axis_value(axis, rng)
                axis_key = _axis_sample_key_in_run(axis, expanded_run)
                if axis_key is not None:
                    axis_samples[axis_key] = float(sample_value)

                for entity_kind, entity_index, parameter_index in axis.occurrences:
                    if entity_kind == "lens":
                        if isinstance(expanded_run.lenses[entity_index].parameters[parameter_index], DerivedRangeSpec):
                            continue
                        _apply_lens_parameter(expanded_run.lenses[entity_index], parameter_index, sample_value)
                    elif entity_kind == "source":
                        if isinstance(expanded_run.sources[entity_index].parameters[parameter_index], DerivedRangeSpec):
                            continue
                        _apply_source_parameter(expanded_run.sources[entity_index], parameter_index, sample_value)

            merged_variables: dict[str, Any] = dict(expanded_run.variables)
            for variable_name, variable_value in expanded_run.variables.items():
                materialized_value = _materialize_range_bound_value(variable_value, axis_samples, merged_variables)
                merged_variables[variable_name] = materialized_value

            for lens in expanded_run.lenses:
                for idx, parameter in enumerate(lens.parameters):
                    if isinstance(parameter, (DerivedRangeSpec, DeferredExpression)):
                        materialized_value = _materialize_range_bound_value(parameter, axis_samples, merged_variables)
                        _apply_lens_parameter(lens, idx, materialized_value)
                lens.redshift = _materialize_range_bound_value(lens.redshift, axis_samples, merged_variables)
                lens.pos = tuple(
                    _materialize_range_bound_value(value, axis_samples, merged_variables)
                    for value in lens.pos
                )
            for source in expanded_run.sources:
                for idx, parameter in enumerate(source.parameters):
                    if isinstance(parameter, (DerivedRangeSpec, DeferredExpression)):
                        materialized_value = _materialize_range_bound_value(parameter, axis_samples, merged_variables)
                        _apply_source_parameter(source, idx, materialized_value)
                source.redshift = _materialize_range_bound_value(source.redshift, axis_samples, merged_variables)
                source.pos = tuple(
                    _materialize_range_bound_value(value, axis_samples, merged_variables)
                    for value in source.pos
                )
                if isinstance(source, ExtendedSource):
                    source.flux = _materialize_range_bound_value(source.flux, axis_samples, merged_variables)
                    source.ellipticity = _materialize_range_bound_value(source.ellipticity, axis_samples, merged_variables)
                    source.PA = _materialize_range_bound_value(source.PA, axis_samples, merged_variables)
                    for key in ("re", "sigma", "r0", "alpha", "beta", "rtrun", "rcore"):
                        if key in source.kwargs and isinstance(source.kwargs[key], (DerivedRangeSpec, DeferredExpression)):
                            source.kwargs[key] = _materialize_range_bound_value(
                                source.kwargs[key], axis_samples, merged_variables
                            )

            expanded_run.variables = merged_variables
            expanded_run.cfg = _materialize_range_bound_value(expanded_run.cfg, axis_samples, expanded_run.variables)
            expanded_run.psf = _materialize_range_bound_value(expanded_run.psf, axis_samples, expanded_run.variables)

            expanded_run.mass_to_light_ratios = {
                label: _materialize_range_bound_value(value, axis_samples, expanded_run.variables)
                for label, value in expanded_run.mass_to_light_ratios.items()
            }
            expanded_run.luminosities = {
                label: _materialize_range_bound_value(value, axis_samples, expanded_run.variables)
                for label, value in expanded_run.luminosities.items()
            }
            expanded_run.commands = [
                _materialize_command_line_with_variables(command_line, expanded_run.variables)
                for command_line in expanded_run.commands
            ]
            expanded_runs.append(expanded_run)
        return EnsembleRun(runs=expanded_runs, axes=axes, template=run)

    sampled_values = [axis.range_spec.values(log_space=axis.log_space) for axis in axes]

    expanded_runs: list[GlaficRun] = []
    for combination in product(*sampled_values):
        expanded_run = deepcopy(run)
        axis_samples: dict[int, float] = {}
        for axis, value in zip(axes, combination):
            axis_key = _axis_sample_key_in_run(axis, expanded_run)
            if axis_key is not None:
                axis_samples[axis_key] = float(value)
        for axis, value in zip(axes, combination):
            for entity_kind, entity_index, parameter_index in axis.occurrences:
                if entity_kind == "lens":
                    if isinstance(expanded_run.lenses[entity_index].parameters[parameter_index], DerivedRangeSpec):
                        continue
                    _apply_lens_parameter(expanded_run.lenses[entity_index], parameter_index, value)
                elif entity_kind == "source":
                    if isinstance(expanded_run.sources[entity_index].parameters[parameter_index], DerivedRangeSpec):
                        continue
                    _apply_source_parameter(expanded_run.sources[entity_index], parameter_index, value)

        merged_variables: dict[str, Any] = dict(expanded_run.variables)
        for variable_name, variable_value in expanded_run.variables.items():
            materialized_value = _materialize_range_bound_value(variable_value, axis_samples, merged_variables)
            merged_variables[variable_name] = materialized_value

        for lens in expanded_run.lenses:
            for idx, parameter in enumerate(lens.parameters):
                if isinstance(parameter, DerivedRangeSpec):
                    lens.parameters[idx] = _materialize_derived_range_value(parameter, merged_variables, axis_samples)
                    lens.metadata[f"p{idx + 1}"] = lens.parameters[idx]
        for source in expanded_run.sources:
            for idx, parameter in enumerate(source.parameters):
                if isinstance(parameter, DerivedRangeSpec):
                    source.parameters[idx] = _materialize_derived_range_value(parameter, merged_variables, axis_samples)
                    source.metadata[f"p{idx + 1}"] = source.parameters[idx]
            if isinstance(source, ExtendedSource):
                    for key in ("re", "sigma", "r0", "alpha", "beta", "rtrun", "rcore"):
                        if key in source.kwargs and isinstance(source.kwargs[key], (DerivedRangeSpec, DeferredExpression)):
                            source.kwargs[key] = _materialize_range_bound_value(
                                source.kwargs[key], axis_samples, merged_variables
                            )

        expanded_run.variables = merged_variables

        expanded_run.mass_to_light_ratios = {
            label: _materialize_range_bound_value(value, axis_samples, expanded_run.variables)
            for label, value in expanded_run.mass_to_light_ratios.items()
        }
        expanded_run.luminosities = {
            label: _materialize_range_bound_value(value, axis_samples, expanded_run.variables)
            for label, value in expanded_run.luminosities.items()
        }
        expanded_run.commands = [
            _materialize_command_line_with_variables(command_line, expanded_run.variables)
            for command_line in expanded_run.commands
        ]
        expanded_runs.append(expanded_run)

    return EnsembleRun(runs=expanded_runs, axes=axes, template=run)


def parse_input_file(
    path: str | Path,
    warning_callback: Callable[[str], None] | None = None,
    *,
    resolve_only: bool = False,
) -> EnsembleRun | GlaficRun:
    with Path(path).open("r", encoding="utf-8") as input_file:
        return parse_input_lines(input_file.readlines(), warning_callback=warning_callback, resolve_only=resolve_only)


def parse_input_lines(
    lines: list[str],
    warning_callback: Callable[[str], None] | None = None,
    *,
    resolve_only: bool = False,
) -> EnsembleRun | GlaficRun:
    sfh_profiles = _parse_sfh_profiles(lines)
    cfg: dict[str, Any] = {}
    sources: list[Source] = []
    lenses: list[Lens] = []
    commands: list[str] = []
    startup: dict[str, int] = {}
    psf: dict[str, Any] = {}
    raw_variables: dict[str, Any] = {}
    mass_to_light_ratios: dict[str, Any] = {}
    luminosities: dict[str, Any] = {}
    sampling_mode = "interval"
    sampling_count: int | None = None

    mode = "preamble"
    for raw_line in lines:
        stripped = _strip_comment(raw_line)
        if not stripped:
            continue

        variable_definition = _parse_variable_definition(stripped)
        if variable_definition is not None:
            variable_name, variable_value = variable_definition
            raw_variables[variable_name] = variable_value
            continue

        tokens = _tokenize(stripped)
        if not tokens:
            continue

        head = tokens[0].lower()

        if head == "mode":
            if len(tokens) < 2:
                raise ValueError(f"Invalid MODE directive: {stripped}")
            sampling_mode = tokens[1].lower()
            if sampling_mode not in {"interval", "rng", "resolve"}:
                raise ValueError(f"Unsupported MODE '{tokens[1]}'; expected 'interval', 'rng', or 'resolve'")
            if sampling_mode == "rng":
                sampling_count = int(tokens[2]) if len(tokens) > 2 else 100
                if sampling_count < 1:
                    raise ValueError(f"MODE rng requires a positive integer sample count, got {tokens[2] if len(tokens) > 2 else 'default'}")
            else:
                sampling_count = None
            if sampling_mode == "resolve":
                resolve_only = True
            continue

        if mode == "sfh":
            if head == "end":
                mode = "startup"
            continue

        if head in {"lens", "extend", "point", "psf"} and mode != "startup":
            raise ValueError(f"Model definition found outside startup block: {stripped}")

        if head == "sfh":
            mode = "sfh"
            continue

        if head == "startup":
            startup = {
                "lenses": int(tokens[1]) if len(tokens) > 1 else 0,
                "extended_sources": int(tokens[2]) if len(tokens) > 2 else 0,
                "point_sources": int(tokens[3]) if len(tokens) > 3 else 0,
            }
            mode = "startup"
            continue

        if head == "end_startup":
            mode = "post_startup"
            continue

        if head == "start_setopt" or head == "end_setopt":
            _reject_blocked_command(stripped)
            continue

        if head == "start_command":
            mode = "command"
            continue

        if head in {"ml", "lum"}:
            label, value = _parse_label_value_directive(tokens, head)
            target = mass_to_light_ratios if head == "ml" else luminosities
            if label in target:
                raise ValueError(f"Duplicate {head} directive for label '{label}'")
            target[label] = value
            continue

        if mode == "startup":
            if head == "lens":
                lenses.append(_parse_lens(tokens))
                continue
            if head == "extend":
                sources.append(_parse_extended_source(tokens))
                continue
            if head == "point":
                sources.append(_parse_point_source(tokens))
                continue
            if head == "psf":
                psf = _parse_psf(tokens)
                continue
            raise ValueError(f"Unrecognized startup line: {stripped}")

        if mode == "command":
            _reject_blocked_command(stripped)
            commands.append(stripped)
            continue

        config_key, value_token_count = _config_key_from_tokens(tokens)
        if len(tokens) > value_token_count:
            if config_key not in KNOWN_CONFIG_KEYS:
                if warning_callback is not None:
                    warning_callback(f"Unrecognized configuration key '{config_key}' in line: {stripped}")
                else:
                    _loud_warning(f"Unrecognized configuration key '{config_key}' in line: {stripped}")
            cfg[config_key] = _parse_value(" ".join(tokens[value_token_count:]))
        else:
            if config_key not in KNOWN_CONFIG_KEYS:
                if warning_callback is not None:
                    warning_callback(f"Unrecognized configuration key '{config_key}' in line: {stripped}")
                else:
                    _loud_warning(f"Unrecognized configuration key '{config_key}' in line: {stripped}")
            cfg[config_key] = True

    run = GlaficRun(
        cfg=cfg,
        sources=sources,
        lenses=lenses,
        commands=commands,
        startup=startup,
        psf=psf,
        variables=raw_variables,
        sfh_profiles=sfh_profiles,
        mass_to_light_ratios=mass_to_light_ratios,
        luminosities=luminosities,
    )
    _resolve_run_variables(run, raw_variables, defer_functions=sampling_mode == "rng")
    if startup:
        lens_count = len(lenses)
        extended_count = sum(isinstance(source, ExtendedSource) for source in sources)
        point_count = sum(isinstance(source, PointSource) for source in sources)
        if startup.get("lenses", lens_count) != lens_count:
            message = f"Startup lens count mismatch: declared {startup.get('lenses')}, parsed {lens_count}"
            (warning_callback or _loud_warning)(message)
        if startup.get("extended_sources", extended_count) != extended_count:
            message = f"Startup extended source count mismatch: declared {startup.get('extended_sources')}, parsed {extended_count}"
            (warning_callback or _loud_warning)(message)
        if startup.get("point_sources", point_count) != point_count:
            message = f"Startup point source count mismatch: declared {startup.get('point_sources')}, parsed {point_count}"
            (warning_callback or _loud_warning)(message)
    if resolve_only:
        return run
    return _expand_run(run, mode=sampling_mode, count=sampling_count)