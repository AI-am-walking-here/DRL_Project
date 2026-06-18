"""Handoff remediation tests."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.pipeline.handoff_remediate import (
    Finding,
    FULL_PROFILE_DEFAULTS,
    analyze_day,
    analyze_medium,
    apply_full_grid_remediations,
    apply_remediations,
    project_grid_eta_from_day,
    remediation_to_dict,
)


def test_analyze_flags_low_expert_replay(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = {
        "pipeline_state": {
            "events": [{"event": "G-BC_ladder", "expert_replay_rate": 0.6}],
        },
        "eval": {"success_rate": 0.05},
    }
    findings = analyze_medium(root, run_dir, report)
    ids = {f.id for f in findings}
    assert "expert_replay_low" in ids
    assert "val_success_low" in ids


def test_apply_expert_and_day_overrides(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs/expert").mkdir(parents=True)
    (root / "configs/expert/rrt_connect.yaml").write_text(
        "t_validate_s: 10.0\nt_label_s: 3.0\n"
    )
    findings = [
        Finding("expert_replay_low", "warn", "low", 0.6),
        Finding("val_success_low", "warn", "low val", 0.05),
    ]
    result = apply_remediations(root, findings)
    expert = (root / "configs/expert/rrt_connect.yaml").read_text()
    assert "t_label_s: 4.0" in expert or "t_label_s: 4" in expert
    overrides = root / "configs/handoff/day_overrides.yaml"
    assert overrides.exists()
    data = remediation_to_dict(result)
    assert data["actions"]


def test_project_grid_eta_from_day(tmp_path: Path) -> None:
    run_dir = tmp_path / "day"
    run_dir.mkdir()
    (run_dir / "COMPLETED").touch()
    (run_dir / ".pipeline_progress.json").write_text(
        json.dumps({"elapsed_s": 20 * 3600, "done": True})
    )
    (tmp_path / "configs/handoff").mkdir(parents=True)
    (tmp_path / "configs/handoff/day_overrides.yaml").write_text("bc_epochs: 175\ndagger_epochs: 90\n")
    (tmp_path / "configs/pipeline.yaml").write_text(
        "profile:\n  day:\n    dagger_rounds: 5\n    dagger_budget: 35000\n    ppo_steps: 500000\n"
    )
    eta = project_grid_eta_from_day(
        tmp_path,
        run_dir,
        FULL_PROFILE_DEFAULTS,
        conditions=["full", "rac_noreroute", "bc_dagger"],
        seeds=3,
    )
    assert eta["ok"]
    assert eta["day_wall_h"] == 20.0
    assert eta["grid_jobs"] == 9
    assert eta["sequential_h"] > 0


def test_apply_full_trim_when_high_val(tmp_path: Path) -> None:
    findings = [
        Finding("val_success", "info", "high", 0.45),
        Finding("val_success_high", "info", "high", 0.45),
        Finding("dagger_plateau", "info", "plateau", 0.42),
    ]
    result, overrides = apply_full_grid_remediations(tmp_path, findings)
    assert overrides["bc_epochs"] == 160
    assert overrides["ppo_steps"] == 750_000
    assert result.actions


def test_apply_full_overrides_on_weak_val(tmp_path: Path) -> None:
    run_dir = tmp_path / "day"
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "eval/val_eval.json").write_text(json.dumps({"success_rate": 0.12}))
    (run_dir / "dagger").mkdir()
    (run_dir / "dagger/round_stats.json").write_text(json.dumps([0.05, 0.1, 0.18]))
    findings = analyze_day(tmp_path, run_dir)
    result, overrides = apply_full_grid_remediations(tmp_path, findings)
    assert overrides["bc_epochs"] == 220
    assert overrides["dagger_epochs"] == 110
    assert (tmp_path / "configs/handoff/full_overrides.yaml").exists()
    assert result.actions
