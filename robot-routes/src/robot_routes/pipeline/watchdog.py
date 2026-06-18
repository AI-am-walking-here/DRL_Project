"""Stage watchdog — detect true hangs without killing long-running work (§11.7.4)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from robot_routes.pipeline.stage_progress import STAGE_LIVE


class StageWatchdog:
    """Kill only when no heartbeat, no stage_live, and no child subprocess activity."""

    def __init__(
        self,
        run_dir: Path,
        timeout_s: int = 1800,
        *,
        stage: str = "",
    ) -> None:
        self.run_dir = run_dir
        self.timeout_s = timeout_s
        self.stage = stage
        self._child: subprocess.Popen[Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach_child(self, proc: subprocess.Popen[Any]) -> None:
        self._child = proc

    def _child_running(self) -> bool:
        return self._child is not None and self._child.poll() is None

    def _signal_ages_s(self) -> list[float]:
        ages: list[float] = []
        now = time.time()
        hb = self.run_dir / "heartbeat"
        if hb.exists():
            try:
                ages.append(now - float(hb.read_text().strip()))
            except (ValueError, OSError):
                ages.append(now - hb.stat().st_mtime)
        live = self.run_dir / STAGE_LIVE
        if live.exists():
            ages.append(now - live.stat().st_mtime)
        return ages

    def _log_kill(self, reason: str) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "watchdog_kill",
            "stage": self.stage,
            "reason": reason,
            "timeout_s": self.timeout_s,
            "child_running": self._child_running(),
            "signal_ages_s": self._signal_ages_s(),
        }
        log = self.run_dir / "events.log"
        try:
            with log.open("a") as f:
                f.write(json.dumps(payload) + "\n")
        except OSError:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(60):
            if self._child_running():
                continue
            ages = self._signal_ages_s()
            if not ages:
                continue
            if min(ages) > self.timeout_s:
                reason = (
                    f"no liveness for {min(ages):.0f}s "
                    f"(limit {self.timeout_s}s, stage={self.stage})"
                )
                self._log_kill(reason)
                os.kill(os.getpid(), signal.SIGKILL)

    def __enter__(self) -> "StageWatchdog":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def run_with_heartbeat(
    fn: Callable[[], None],
    heartbeat: Callable[[], None],
    *,
    tick_s: float = 15.0,
) -> None:
    """Run in-process stage work while refreshing heartbeat (calibration, eval, ...)."""
    done = threading.Event()
    exc: list[BaseException] = []

    def runner() -> None:
        try:
            fn()
        except BaseException as e:
            exc.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    heartbeat()
    while not done.wait(tick_s):
        heartbeat()
    thread.join()
    if exc:
        raise exc[0]
