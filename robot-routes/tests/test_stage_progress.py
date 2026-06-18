"""Stage progress snapshot tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

from robot_routes.pipeline.stage_progress import liveness, snapshot_run_progress, write_stage_live


def test_snapshot_detects_orphaned_dagger(tmp_path: Path) -> None:
    run_dir = tmp_path / "rac_noreroute_seed0"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(
        json.dumps({"condition": "rac_noreroute", "seed": 0, "profile": "full"})
    )
    (run_dir / "configs").mkdir()
    (run_dir / "configs" / "dagger_rac.yaml").write_text("rounds: 6\nbudget: 40000\n")
    state = {
        "stages": {
            "setup": {"status": "COMPLETED"},
            "train_bc": {"status": "COMPLETED"},
            "dagger_rac": {"status": "RUNNING"},
        }
    }
    (run_dir / "pipeline_state.json").write_text(json.dumps(state))
    (run_dir / "train_bc.stamp").touch()
    (run_dir / "dagger").mkdir()
    (run_dir / "dagger" / "round_0.h5").write_bytes(b"x" * 100)
    (run_dir / "heartbeat").write_text(str(time.time() - 600))

    snap = snapshot_run_progress(run_dir)
    assert snap["current_stage"] == "dagger_rac"
    assert snap["liveness"] == "orphaned"
    assert "dagger_round" in snap


def test_write_stage_live_updates_heartbeat(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_stage_live(run_dir, job="dagger_rac", phase="round_0_collect", current=10, total=100)
    live = json.loads((run_dir / "stage_live.json").read_text())
    assert live["current"] == 10
    info = liveness(run_dir, "RUNNING")
    assert info["liveness"] in ("alive", "stale")
