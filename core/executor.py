from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from parser import ExpansionAxis
from runs import EnsembleRun, GlaficRun
from ui import ConsoleReporter, LogLevel


MAX_RUN_NAME_LENGTH = 72
MAX_RUN_NAME_COMPONENT_LENGTH = 16
MAX_READABLE_AXES_IN_NAME = 4


@dataclass
class RunResult:
    name: str
    returncode: int
    input_path: Path
    output_dir: Path
    stdout_path: Path
    stderr_path: Path


def _sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", text)
    return cleaned.strip("_") or "run"


def _format_value_for_name(value: Any) -> str:
    if isinstance(value, float):
        return format(value, ".2e") # round to 2 dp for the run name, to avoid overly long names
    return str(value)


def _compact_name_with_hash(name: str, fingerprint: str) -> str:
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10]
    sanitized = _sanitize_component(name)
    suffix = f"_{digest}"
    if len(sanitized) + len(suffix) <= MAX_RUN_NAME_LENGTH:
        return f"{sanitized}{suffix}"

    head_budget = max(1, MAX_RUN_NAME_LENGTH - len(suffix))
    return f"{sanitized[:head_budget]}{suffix}"


def _axis_value(run: GlaficRun, axis: ExpansionAxis) -> Any:
    entity_kind, entity_index, parameter_index = axis.occurrences[0]
    if entity_kind == "lens":
        return run.lenses[entity_index].parameters[parameter_index]
    if entity_kind == "source":
        return run.sources[entity_index].parameters[parameter_index]
    if entity_kind == "luminosity":
        return run.luminosities[entity_index]
    if entity_kind == "mass_to_light":
        return run.mass_to_light_ratios[entity_index]
    raise ValueError(f"Unsupported axis entity kind '{entity_kind}'")


def _has_invalid_redshift_order(run: GlaficRun) -> bool:
    if not run.lenses or not run.sources:
        return False

    lens_redshift = run.lenses[0].redshift
    source_redshift = run.sources[0].redshift
    if lens_redshift is None or source_redshift is None:
        return False

    return float(lens_redshift) > float(source_redshift)


def build_run_name(run: GlaficRun, axes: list[ExpansionAxis] | None = None, fallback_name: str | None = None) -> str:
    if not axes:
        base_name = _sanitize_component(fallback_name or "run")
        if len(base_name) <= MAX_RUN_NAME_LENGTH:
            return base_name
        return _compact_name_with_hash(base_name, base_name)

    raw_pairs: list[str] = []
    readable_parts: list[str] = []
    for axis in axes:
        axis_value = _format_value_for_name(_axis_value(run, axis))
        raw_pairs.append(f"{axis.label}={axis_value}")

        if len(readable_parts) < MAX_READABLE_AXES_IN_NAME:
            label = _sanitize_component(axis.label)[:MAX_RUN_NAME_COMPONENT_LENGTH]
            value = _sanitize_component(axis_value)[:MAX_RUN_NAME_COMPONENT_LENGTH]
            readable_parts.append(f"{label}_{value}")

    readable_stem = "_".join(readable_parts) if readable_parts else _sanitize_component(fallback_name or "run")
    return _compact_name_with_hash(readable_stem, "|".join(raw_pairs))


def _resolve_glafic_executable(workspace_root: Path) -> str:
    local_binary = workspace_root / "glafic"
    if local_binary.exists():
        return str(local_binary)

    which_binary = shutil.which("glafic")
    if which_binary is not None:
        return which_binary

    raise FileNotFoundError("Could not find a glafic executable in the workspace or PATH")


def _write_run_input(run: GlaficRun, input_path: Path) -> None:
    input_path.write_text(run.to_input_text(), encoding="utf-8")


def _write_run_header(run: GlaficRun, header_path: Path) -> None:
    header_path.write_text(
        json.dumps(
            {
                "config": run.cfg,
                "variables": run.variables,
                "lenses": [lens.evaluate() for lens in run.lenses],
                "sources": [source.evaluate() for source in run.sources],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _copy_unstretched_fits(run_out_dir: Path) -> None:
    source_names = ("_image.fits", "_source.fits")
    missing = [source_name for source_name in source_names if not (run_out_dir / source_name).is_file()]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(f"glafic did not produce required unstretched FITS product(s): {missing_names}")
    shutil.copyfile(run_out_dir / "_image.fits", run_out_dir / "lensed_image_only.fits")
    shutil.copyfile(run_out_dir / "_source.fits", run_out_dir / "lensing_galaxy_light.fits")
    with fits.open(run_out_dir / "_image.fits", memmap=False) as image_hdul, fits.open(
        run_out_dir / "_source.fits", memmap=False
    ) as source_hdul:
        image_data = image_hdul[0].data
        source_data = source_hdul[0].data
        if image_data is None or source_data is None:
            raise RuntimeError("glafic FITS products must contain primary image data")
        if image_data.shape != source_data.shape:
            raise RuntimeError(
                f"Cannot combine FITS products with different shapes: {image_data.shape} and {source_data.shape}"
            )
        combined = np.asarray(image_data, dtype=np.float64) + np.asarray(source_data, dtype=np.float64)
        header = image_hdul[0].header.copy()
    fits.PrimaryHDU(data=combined, header=header).writeto(run_out_dir / "lens_image.fits", overwrite=True)


def prepare_run_inputs(
    parsed: GlaficRun | EnsembleRun,
    *,
    out_root: str | Path,
    fallback_name: str | None = None,
) -> list[Path]:
    """Write expanded run inputs without launching glafic."""
    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)

    if isinstance(parsed, GlaficRun):
        runs_and_names = [(parsed, build_run_name(parsed, None, fallback_name=fallback_name))]
    else:
        runs_and_names = [
            (run, build_run_name(run, parsed.axes, fallback_name=fallback_name))
            for run in parsed.runs
        ]

    input_paths: list[Path] = []
    for run, run_name in runs_and_names:
        run_out_dir = out_path / run_name
        run_out_dir.mkdir(parents=True, exist_ok=True)
        run_copy = run.clone()
        run_copy.cfg["prefix"] = str(run_out_dir.resolve()) + os.sep
        input_path = run_out_dir / "run.input"
        _write_run_input(run_copy, input_path)
        _write_run_header(run_copy, run_out_dir / "header.json")
        input_paths.append(input_path)

    return input_paths


def _run_glafic_job(
    run: GlaficRun,
    run_name: str,
    workspace_root: str,
    out_root: str,
    glafic_executable: str,
) -> RunResult:
    workspace_path = Path(workspace_root)
    out_path = Path(out_root)

    run_out_dir = out_path / run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    run_copy = run.clone()
    run_copy.cfg["prefix"] = str(run_out_dir.resolve()) + os.sep

    input_path = run_out_dir / "run.input"
    stdout_path = run_out_dir / "stdout.log"
    stderr_path = run_out_dir / "stderr.log"

    _write_run_input(run_copy, input_path)

    completed = subprocess.run(
        [glafic_executable, str(input_path)],
        cwd=str(workspace_path),
        capture_output=True,
        text=True,
        check=False,
    )

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"glafic failed for '{run_name}' with exit code {completed.returncode}. See {stderr_path}"
        )

    _copy_unstretched_fits(run_out_dir)

    return RunResult(
        name=run_name,
        returncode=completed.returncode,
        input_path=input_path,
        output_dir=run_out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def execute_runs(
    parsed: GlaficRun | EnsembleRun,
    *,
    workspace_root: str | Path | None = None,
    out_root: str | Path | None = None,
    max_workers: int | None = None,
    fallback_name: str | None = None,
    reporter: ConsoleReporter | None = None,
    result_callback: Any | None = None,
) -> list[RunResult]:
    runs = [parsed] if isinstance(parsed, GlaficRun) else parsed.runs
    valid_runs = [run for run in runs if not _has_invalid_redshift_order(run)]
    skipped_runs = len(runs) - len(valid_runs)
    if skipped_runs:
        if reporter is not None:
            reporter.stage(f"Skipping {skipped_runs} run(s) with lens redshift above source redshift")
        if not valid_runs:
            return []
        parsed = valid_runs[0] if isinstance(parsed, GlaficRun) else EnsembleRun(valid_runs, parsed.axes, parsed.template)

    workspace_path = Path(workspace_root) if workspace_root is not None else Path.cwd()
    out_path = Path(out_root) if out_root is not None else workspace_path / "out"

    out_path.mkdir(parents=True, exist_ok=True)

    glafic_executable = _resolve_glafic_executable(workspace_path)
    reporter = reporter or ConsoleReporter(LogLevel.VERBOSE)

    if isinstance(parsed, GlaficRun):
        run_name = build_run_name(parsed, None, fallback_name=fallback_name)
        reporter.stage(f"Running {run_name}")
        reporter.progress(0, 1, run_name)
        result = _run_glafic_job(
            parsed,
            run_name,
            str(workspace_path),
            str(out_path),
            glafic_executable,
        )
        if result_callback is not None:
            result_callback(result, 1, 1)
        reporter.progress(1, 1, run_name)
        reporter.progress_done()
        reporter.stage(f"Finished {run_name}")
        return [
            result
        ]

    runs = parsed.runs
    axes = parsed.axes
    futures = []
    results: list[RunResult] = []
    total_runs = len(runs)

    reporter.stage(f"Preparing {total_runs} run input files")
    reporter.stage("Running jobs")
    reporter.progress(0, total_runs, "starting")

    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count() or 1) as executor:
        for index, run in enumerate(runs, start=1):
            run_fallback = fallback_name
            if not axes and fallback_name:
                run_fallback = f"{fallback_name}_{index:04d}"
            run_name = build_run_name(run, axes, fallback_name=run_fallback)
            futures.append(
                executor.submit(
                    _run_glafic_job,
                    run,
                    run_name,
                    str(workspace_path),
                    str(out_path),
                    glafic_executable,
                )
            )

        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                reporter.progress_done()
                raise

            results.append(result)
            completed += 1
            reporter.progress(completed, total_runs, result.name)
            if result_callback is not None:
                result_callback(result, completed, total_runs)

    reporter.progress_done()
    reporter.stage(f"Finished {total_runs} runs")

    return sorted(results, key=lambda item: item.name)