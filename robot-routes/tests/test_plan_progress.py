"""Pre-travel plan progress tests."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.pipeline.plan_progress import (
    render_plan_dashboard,
    scene_profile_progress,
    snapshot_plan_progress,
)
from robot_routes.pipeline.stage_progress import snapshot_run_progress


def test_scene_profile_progress_from_status(tmp_path: Path) -> None:
    scenes = tmp_path / "data" / "scenes"
    scenes.mkdir(parents=True)
    status = scenes / ".generation_status.json"
    status.write_text(
        json.dumps(
            {
                "profile": "medium",
                "phase": "test_L2",
                "set_index": 8,
                "set_total": 10,
                "current": 20,
                "total": 40,
                "done": False,
                "desc": "test_L2 (40 scenes)",
                "elapsed_s": 1000,
            }
        )
    )
    prog = scene_profile_progress(tmp_path, "medium")
    assert prog["status"] == "running"
    assert 60 < prog["pct"] < 75
    assert "test_L2" in prog["detail"]
    assert prog["eta_s"] > 0


def test_snapshot_plan_progress_sections(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "delta.json").write_text('{"delta_distinct_m": 0.32}')
    data = snapshot_plan_progress(tmp_path)
    assert len(data["sections"]) == 9
    assert data["scope"] == "medium"
    assert "scope_eta_s" in data
    text = render_plan_dashboard(data)
    assert "Medium preflight" in text
    assert "eta" in text
    assert "[#" in text or "[-" in text

def test_snapshot_plan_eta_sums_pending(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "delta.json").write_text('{"delta_distinct_m": 0.32}')
    data = snapshot_plan_progress(tmp_path)
    assert data["scope"] == "medium"
    assert data["scope_eta_s"] > 0
    assert data["scope_completion"]
    scope_pending = [s for s in data["sections"] if s["in_scope"] and s["status"] != "done"]
    assert sum(s["eta_s"] for s in scope_pending) == data["scope_eta_s"]


def test_snapshot_run_progress_stage_pct(tmp_path: Path) -> None:
    run_dir = tmp_path / "rac_noreroute_seed0"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(
        json.dumps({"condition": "rac_noreroute", "seed": 0, "profile": "medium"})
    )
    (run_dir / "configs").mkdir()
    (run_dir / "configs" / "dagger_rac.yaml").write_text("rounds: 2\nbudget: 2000\n")
    state = {
        "stages": {
            "setup": {"status": "COMPLETED"},
            "collect_bc": {"status": "COMPLETED"},
            "train_bc": {"status": "COMPLETED"},
            "dagger_rac": {"status": "RUNNING"},
        }
    }
    (run_dir / "pipeline_state.json").write_text(json.dumps(state))
    (run_dir / "setup.stamp").touch()
    (run_dir / "collect_bc.stamp").touch()
    (run_dir / "train_bc.stamp").touch()
    (run_dir / "collect").mkdir()
    (run_dir / "collect" / "demos_w0.h5").write_bytes(b"x")
    (run_dir / "dagger").mkdir()

    snap = snapshot_run_progress(run_dir)
    by_name = {s["name"]: s for s in snap["stages"]}
    assert by_name["setup"]["pct"] == 100.0
    assert by_name["collect_bc"]["pct"] > 0
    assert by_name["dagger_rac"]["status"] == "RUNNING"
