"""CLI progress bars + JSON status snapshots for long-running setup jobs."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bar(pct: float, width: int = 32) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100.0)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def fmt_eta(seconds: float) -> str:
    """Human-readable duration for dashboards."""
    if seconds <= 0 or seconds > 86400 * 30:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def rate_eta_s(elapsed_s: float, pct: float) -> float:
    """ETA from elapsed time and completion fraction."""
    if pct <= 0 or elapsed_s <= 0:
        return 0.0
    if pct >= 100:
        return 0.0
    return elapsed_s * (100.0 - pct) / pct


@dataclass
class ProgressReporter:
    """Progress bar (tqdm if installed) + optional JSON status file for `watch`."""

    job: str
    phase: str
    total: int
    unit: str = "step"
    status_path: Path | None = None
    desc: str = ""
    _current: int = 0
    _start: float = field(default_factory=time.monotonic)
    _extra: dict[str, Any] = field(default_factory=dict)
    _tqdm: Any = None

    def __post_init__(self) -> None:
        label = self.desc or f"{self.job}: {self.phase}"
        try:
            from tqdm import tqdm

            self._tqdm = tqdm(
                total=max(self.total, 1),
                desc=label,
                unit=self.unit,
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        except ImportError:
            self._tqdm = None
            sys.stderr.write(f"{label} (0/{self.total})\n")
            sys.stderr.flush()
        self._write_status()

    def update(self, n: int = 1, **extra: Any) -> None:
        self._current = min(self._current + n, self.total)
        self._extra.update(extra)
        if self._tqdm is not None:
            self._tqdm.update(n)
            if extra:
                self._tqdm.set_postfix(**{k: _fmt(v) for k, v in extra.items()}, refresh=False)
        else:
            self._print_fallback()
        self._write_status()

    def set(self, value: int, **extra: Any) -> None:
        delta = max(0, min(value, self.total) - self._current)
        if delta:
            self.update(delta, **extra)
        elif extra:
            self._extra.update(extra)
            if self._tqdm is not None:
                self._tqdm.set_postfix(**{k: _fmt(v) for k, v in extra.items()}, refresh=False)
            self._write_status()

    def close(self, final: bool = True, **extra: Any) -> None:
        if extra:
            self._extra.update(extra)
        if self._tqdm is not None:
            if extra:
                self._tqdm.set_postfix(**{k: _fmt(v) for k, v in extra.items()})
            self._tqdm.close()
        else:
            elapsed = time.monotonic() - self._start
            sys.stderr.write(
                f"done {self.job}/{self.phase}: {self._current}/{self.total} "
                f"in {elapsed:.1f}s\n"
            )
            sys.stderr.flush()
        self._current = self.total
        self._write_status(done=final)

    def _print_fallback(self) -> None:
        pct = 100.0 * self._current / max(self.total, 1)
        elapsed = time.monotonic() - self._start
        rate = self._current / max(elapsed, 1e-6)
        eta = (self.total - self._current) / max(rate, 1e-6)
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in self._extra.items())
        line = (
            f"\r{_bar(pct)} {pct:5.1f}% {self._current}/{self.total} "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s {extra}"
        )
        sys.stderr.write(line[:120].ljust(120))
        sys.stderr.flush()

    def _write_status(self, done: bool = False) -> None:
        if self.status_path is None:
            return
        elapsed = time.monotonic() - self._start
        pct = 100.0 * self._current / max(self.total, 1)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "job": self.job,
            "phase": self.phase,
            "desc": self.desc,
            "current": self._current,
            "total": self.total,
            "unit": self.unit,
            "pct": round(pct, 2),
            "elapsed_s": round(elapsed, 1),
            "done": done,
            **self._extra,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.status_path)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
