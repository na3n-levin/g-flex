from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from models import ExtendedSource, Lens, PointSource, Source


class GlaficRun:
    defaultPrimaryConfig = {
        "omega": 0.315,
        "lambda": 0.685,
        "weos": -1.03,
        "hubble": 0.674,
        "xmin": -5,
        "xmax": 5,
        "ymin": -5,
        "ymax": 5,
        "pix_poi": 0.01,
        "pix_ext": 0.01,
        "ran_seed": -1,
    }

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        sources: list[Source] | None = None,
        lenses: list[Lens] | None = None,
        commands: list[str] | None = None,
        startup: dict[str, int] | None = None,
        setopt: list[list[int]] | None = None,
        psf: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        sfh_profiles: dict[str, dict[str, Any]] | None = None,
        mass_to_light_ratios: dict[str, Any] | None = None,
        luminosities: dict[str, Any] | None = None,
    ):
        self.cfg = dict(self.defaultPrimaryConfig)
        for override_key, override_value in (cfg or {}).items():
            self.cfg[override_key] = override_value

        self.sources = list(sources) if sources is not None else []
        self.lenses = list(lenses) if lenses is not None else []
        self.commands = list(commands) if commands is not None else []
        self.startup = dict(startup) if startup is not None else {}
        self.setopt = [list(row) for row in setopt] if setopt is not None else []
        self.psf = dict(psf) if psf is not None else {}
        self.variables = dict(variables) if variables is not None else {}
        self.sfh_profiles = dict(sfh_profiles) if sfh_profiles is not None else {}
        self.mass_to_light_ratios = dict(mass_to_light_ratios) if mass_to_light_ratios is not None else {}
        self.luminosities = dict(luminosities) if luminosities is not None else {}

    def clone(self) -> "GlaficRun":
        return deepcopy(self)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return format(value, ".12g")
        return str(value)

    def _format_config_key(self, key: str) -> str:
        return key

    def _render_config_lines(self) -> list[str]:
        lines: list[str] = []
        for key, value in self.cfg.items():
            lines.append(f"{self._format_config_key(key)} {self._format_value(value)}")
        return lines

    def _render_startup_lines(self) -> list[str]:
        lens_lines = [f"lens {lens.evaluate()}" for lens in self.lenses]
        source_lines = [source.evaluate() for source in self.sources]
        psf_lines = ["psf " + " ".join(self._format_value(value) for value in self.psf.values())] if self.psf else []
        ml_lines = [f'ml "{label}" {self._format_value(value)}' for label, value in self.mass_to_light_ratios.items()]
        lum_lines = [f'lum "{label}" {self._format_value(value)}' for label, value in self.luminosities.items()]
        extended_source_count = sum(isinstance(source, ExtendedSource) for source in self.sources)
        point_source_count = sum(isinstance(source, PointSource) for source in self.sources)
        return [
            f"startup {len(self.lenses)} {extended_source_count} {point_source_count}",
            *lens_lines,
            *source_lines,
            *psf_lines,
            *ml_lines,
            *lum_lines,
            "end_startup",
        ]

    def to_input_text(self) -> str:
        lines = self._render_config_lines()
        lines.extend(self._render_startup_lines())
        if self.commands:
            lines.append("start_command")
            lines.extend(self.commands if self.commands[-1].strip().lower() == "quit" else [*self.commands, "quit"])
        else:
            lines.extend(["start_command", "quit"])
        return "\n".join(lines) + "\n"

    @classmethod
    def from_input_file(cls, path: str | Path):
        from parser import parse_input_file

        return parse_input_file(path)


class EnsembleRun:
    def __init__(self, runs: list[GlaficRun] | None = None, axes: list[tuple[int, int, list[Any]]] | None = None, template: GlaficRun | None = None):
        self.runs = list(runs) if runs is not None else []
        self.axes = list(axes) if axes is not None else []
        self.template = template

    def __iter__(self):
        return iter(self.runs)

    def __len__(self):
        return len(self.runs)