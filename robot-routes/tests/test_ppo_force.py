"""Tests for forced PPO resume."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.pipeline.ppo_force import (
    arm_ppo_force,
    grid_job_complete,
    needs_ppo_force_rerun,
)


def _write_state(run_dir: Path, ppo_status: str, *, completed: bool = True) -> None:
    run_dir.mkdir(parents=True)
    state = {
        "stages": {
            "ppo": {"status": ppo_status},
            "pipeline": {"status": "COMPLETED" if completed else "PENDING"},
        },
        "events": [],
    }
    (run_dir / "pipeline_state.json").write_text(json.dumps(state))
    if completed:
        (run_dir / "COMPLETED").touch()
        (run_dir / "evaluate_test.stamp").touch()


def test_grid_job_complete_when_ppo_skipped_but_forced(tmp_path: Path) -> None:
    run_dir = tmp_path / "full_seed0"
    _write_state(run_dir, "SKIPPED")
    assert not grid_job_complete(run_dir, ppo_force=True, has_ppo_stage=True)
    assert grid_job_complete(run_dir, ppo_force=False, has_ppo_stage=True)


def test_arm_ppo_force_clears_skip(tmp_path: Path) -> None:
    run_dir = tmp_path / "full_seed0"
    _write_state(run_dir, "SKIPPED")
    assert needs_ppo_force_rerun(run_dir, ppo_force=True, has_ppo_stage=True)
    assert arm_ppo_force(run_dir)
    assert not (run_dir / "COMPLETED").exists()
    assert not (run_dir / "evaluate_test.stamp").exists()
    state = json.loads((run_dir / "pipeline_state.json").read_text())
    assert state["stages"]["ppo"]["status"] == "PENDING"
