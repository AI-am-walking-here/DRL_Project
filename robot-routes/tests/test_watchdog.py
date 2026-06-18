"""Watchdog must not kill during active child subprocesses."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from robot_routes.pipeline.stage_progress import write_stage_live
from robot_routes.pipeline.watchdog import StageWatchdog


def test_watchdog_skips_kill_while_child_running(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "heartbeat").write_text(str(time.time() - 9999))
    wd = StageWatchdog(run_dir, timeout_s=1, stage="dagger_rac")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
    with wd:
        wd.attach_child(proc)
        time.sleep(2.5)
    proc.wait(timeout=5)
    assert proc.returncode == 0


def test_watchdog_uses_stage_live_as_liveness(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "heartbeat").write_text(str(time.time() - 9999))
    write_stage_live(run_dir, job="dagger_rac", phase="collect", current=1, total=10)
    wd = StageWatchdog(run_dir, timeout_s=1, stage="dagger_rac")
    with wd:
        time.sleep(2.5)
    # no exception / kill — stage_live is fresh
