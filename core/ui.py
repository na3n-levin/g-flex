from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(str, Enum):
    SILENT = "SILENT"
    QUIET = "QUIET"
    VERBOSE = "VERBOSE"


@dataclass
class ConsoleReporter:
    level: LogLevel = LogLevel.VERBOSE
    _progress_started_at: float | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes}m{secs}s"
        if minutes:
            return f"{minutes}m{secs}s"
        return f"{secs}s"

    def warning(self, message: str) -> None:
        if self.level == LogLevel.SILENT:
            return
        print(f"[WARNING] {message}")

    def stage(self, message: str) -> None:
        if self.level == LogLevel.VERBOSE or self.level == LogLevel.QUIET:
            print(message)

    def progress(self, current: int, total: int, label: str = "") -> None:
        if current <= 0 or self._progress_started_at is None:
            self._progress_started_at = time.monotonic()

        bar_width = 28
        completed_width = bar_width if total <= 0 else int(round((current / total) * bar_width))
        completed_width = max(0, min(bar_width, completed_width))
        remaining_width = bar_width - completed_width
        bar = "[" + ("=" * completed_width) + (" " * remaining_width) + "]"
        suffix = f" {current}/{total}"
        if label:
            suffix += f" {label}"

        elapsed = time.monotonic() - self._progress_started_at
        eta = "--:--"
        if total > 0 and current > 0 and elapsed > 0:
            rate = current / elapsed
            remaining_units = max(total - current, 0)
            if rate > 0:
                eta_seconds = remaining_units / rate
                eta = self._format_duration(eta_seconds)
            suffix += f" | elapsed {self._format_duration(elapsed)} | eta {eta}"
        else:
            suffix += f" | elapsed {self._format_duration(elapsed)} | eta {eta}"

        sys.stdout.write("\r" + bar + suffix)
        sys.stdout.flush()

    def progress_done(self) -> None:
        self._progress_started_at = None
        sys.stdout.write("\n")
        sys.stdout.flush()