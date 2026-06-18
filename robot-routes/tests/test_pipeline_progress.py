"""Pipeline-level progress tests."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.pipeline.progress import PipelineProgress, ordered_stages


def test_ordered_stages_rac_noreroute() -> None:
    stages = ordered_stages({"collect_bc", "train_bc", "dagger_rac", "curriculum", "evaluate_val"})
    assert stages == [
        "setup",
        "scene_sets",
        "calibrate_delta",
        "collect_bc",
        "train_bc",
        "dagger_rac",
        "curriculum",
        "evaluate_val",
    ]


def test_pipeline_progress_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = run_dir / ".pipeline_progress.json"
    stages = ["setup", "collect_bc", "train_bc"]
    prog = PipelineProgress(
        run_dir,
        "bc",
        0,
        "smoke",
        stages,
        status_path=status,
    )
    prog.stage_running("setup")
    prog.stage_done("setup")
    data = json.loads(status.read_text())
    assert data["stages_completed"] == 1
    assert data["stages_total"] == 3
    assert data["current_stage"] == "setup"
    assert data["stages"][0]["status"] == "COMPLETED"
    prog.stage_skipped("collect_bc")
    prog.close(ok=True)
    done = json.loads(status.read_text())
    assert done["done"] is True
    assert done["stages_completed"] == 2
