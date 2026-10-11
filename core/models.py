from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RangeSpec:
    start: float
    end: float
    count: int

    def __str__(self) -> str:
        return f"{{{self.start}, {self.end}, {self.count}}}"

    def values(self, *, log_space: bool = False) -> list[float]:
        if self.count < 1:
            raise ValueError("RangeSpec.count must be at least 1")

        if self.count == 1:
            return [float(self.start)]

        start = float(self.start)
        end = float(self.end)

        if log_space:
            if start <= 0 or end <= 0:
                raise ValueError("Log-spaced sampling requires positive start and end values")
            ratio = (end / start) ** (1.0 / (self.count - 1))
            return [start * (ratio ** index) for index in range(self.count)]

        step = (end - start) / float(self.count - 1)
        values = [start + step * index for index in range(self.count)]

        if self.count > 1 and abs(values[-1] - end) > 1.0e-9 * max(1.0, abs(end)):
            values[-1] = end

        return values


class Source:
    def __init__(
        self,
        kind: str = "",
        redshift: float | None = None,
        pos: tuple[float, float] | None = None,
        parameters: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.kind = kind
        self.redshift = redshift
        self.pos = tuple(pos) if pos is not None else (0.0, 0.0)
        self.parameters = list(parameters) if parameters is not None else []
        self.metadata = dict(metadata) if metadata is not None else {}

    def clone(self):
        return deepcopy(self)


class ExtendedSource(Source):
    def __init__(
        self,
        kind: str = "",
        redshift: float | None = None,
        pos: tuple[float, float] | None = None,
        flux: float | None = None,
        ellipticity: float = 0.0,
        PA: float = 0.0,
        kwargs: dict[str, Any] | None = None,
        parameters: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(kind=kind, redshift=redshift, pos=pos, parameters=parameters, metadata=metadata)
        self.flux = flux
        self.ellipticity = ellipticity
        self.PA = PA
        self.kwargs = dict(kwargs) if kwargs is not None else {}

    def evaluate(self):
        label = self.metadata.get("label")
        label_text = f' "{label}"' if label is not None else ""
        if self.kind == "sersic":
            return f"extend {self.kind} {self.redshift} {self.flux} {self.pos[0]} {self.pos[1]} {self.ellipticity} {self.PA} {self.kwargs.get('re')} {self.kwargs.get('n')}{label_text}"
        if self.kind in {"gauss", "srcs"}:
            return f"extend {self.kind} {self.redshift} {self.flux} {self.pos[0]} {self.pos[1]} {self.ellipticity} {self.PA} {self.kwargs.get('sigma')} 0{label_text}"
        if self.kind == "tophat":
            return f"extend {self.kind} {self.redshift} {self.flux} {self.pos[0]} {self.pos[1]} {self.ellipticity} {self.PA} {self.kwargs.get('r0')} 0{label_text}"
        if self.kind == "moffat":
            return f"extend {self.kind} {self.redshift} {self.flux} {self.pos[0]} {self.pos[1]} {self.ellipticity} {self.PA} {self.kwargs.get('alpha')} {self.kwargs.get('beta')}{label_text}"
        if self.kind == "jaffe":
            return f"extend {self.kind} {self.redshift} {self.flux} {self.pos[0]} {self.pos[1]} {self.ellipticity} {self.PA} {self.kwargs.get('rtrun')} {self.kwargs.get('rcore')}{label_text}"
        return f"extend {self.kind}{label_text}"

    def get_light_profile(self):
        if self.kind == "sersic":
            return f"{self.kind} {self.kwargs.get('re')} {self.kwargs.get('n')}"
        if self.kind in {"gauss", "srcs"}:
            return f"{self.kind} {self.kwargs.get('sigma')} 0"
        if self.kind == "tophat":
            return f"{self.kind} {self.kwargs.get('r0')} 0"
        if self.kind == "moffat":
            return f"{self.kind} {self.kwargs.get('alpha')} {self.kwargs.get('beta')}"
        if self.kind == "jaffe":
            return f"{self.kind} {self.kwargs.get('rtrun')} {self.kwargs.get('rcore')}"
        return self.kind


class PointSource(Source):
    def __init__(
        self,
        redshift: float | None = None,
        pos: tuple[float, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(kind="point", redshift=redshift, pos=pos, metadata=metadata)

    def evaluate(self):
        return f"{self.kind} {self.redshift} {self.pos[0]} {self.pos[1]}"


class Lens:
    def __init__(
        self,
        kind: str = "",
        redshift: float | None = None,
        pos: tuple[float, float] | None = None,
        parameters: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.kind = kind
        self.redshift = redshift
        self.pos = tuple(pos) if pos is not None else (0.0, 0.0)
        self.parameters = list(parameters) if parameters is not None else []
        self.metadata = dict(metadata) if metadata is not None else {}

    def clone(self):
        return deepcopy(self)

    def evaluate(self):
        parameter_text = " ".join(str(value) for value in self.parameters[:8])
        label = self.metadata.get("label")
        label_text = f' "{label}"' if label is not None else ""
        return f"{self.kind} {parameter_text}{label_text}".strip()