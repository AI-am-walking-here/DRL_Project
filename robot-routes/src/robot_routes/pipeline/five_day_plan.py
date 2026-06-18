"""Five-day grid planner from measured day-preflight wall times."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_routes.pipeline.conditions import condition_spec, load_grid, stage_list

# Measured from runs/preflight/full_seed0 stamp deltas (full condition, profile=day).
DAY_STAGE_HOURS: dict[str, float] = {
    "collect_bc": 0.37,
    "train_bc": 0.35,
    "dagger_rac": 6.23,
    "curriculum": 6.82,
    "ppo": 1.48,
    "evaluate_val": 0.93,
    "evaluate_test": 0.68,
    "overhead": 0.10,
}

# Day run effective training knobs (pipeline day + handoff day_overrides).
DAY_BASELINE_TRAIN: dict[str, float] = {
    "bc_epochs": 175.0,
    "dagger_rounds": 5.0,
    "dagger_budget": 35000.0,
    "dagger_epochs": 90.0,
    "ppo_steps": 500000.0,
}

GRID_CONDITION_FRAC: dict[str, float] = {
    "full": 1.0,
    "rac_noreroute": 0.72,
    "bc_dagger": 0.55,
}

DEFAULT_GRID_CONDITIONS = ("full", "rac_noreroute", "bc_dagger")
BUDGET_HOURS = 118.0  # 5d − small buffer for scene verify / handoff


@dataclass(frozen=True)
class FiveDayPlan:
    full_overrides: dict[str, Any]
    expert_patches: dict[str, Any]
    grid_conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    per_job_h: dict[str, float]
    total_h: float
    findings: list[str]
    actions: list[str]


def _train_scale(overrides: dict[str, Any], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "bc": overrides["bc_epochs"] / baseline["bc_epochs"],
        "dagger": (
            overrides["dagger_rounds"]
            / baseline["dagger_rounds"]
            * overrides["dagger_budget"]
            / baseline["dagger_budget"]
            * overrides["dagger_epochs"]
            / baseline["dagger_epochs"]
        ),
        "ppo": overrides["ppo_steps"] / baseline["ppo_steps"],
    }


def estimate_condition_hours(
    condition: str,
    overrides: dict[str, Any],
    *,
    baseline: dict[str, float] | None = None,
    stage_hours: dict[str, float] | None = None,
    grid_path: Path | None = None,
) -> float:
    baseline = baseline or DAY_BASELINE_TRAIN
    stage_hours = stage_hours or DAY_STAGE_HOURS
    if grid_path and grid_path.exists():
        import yaml

        grid = yaml.safe_load(grid_path.read_text())
        spec = condition_spec(grid, condition)
        stages = set(stage_list(spec))
    else:
        stages = {
            "full": {
                "collect_bc",
                "train_bc",
                "dagger_rac",
                "curriculum",
                "evaluate_val",
                "ppo",
                "evaluate_test",
            },
            "rac_noreroute": {
                "collect_bc",
                "train_bc",
                "dagger_rac",
                "curriculum",
                "evaluate_val",
            },
            "bc_dagger": {"collect_bc", "train_bc", "dagger_rac", "evaluate_val"},
        }[condition]

    sc = _train_scale(overrides, baseline)
    hours = stage_hours["overhead"]
    if "collect_bc" in stages or "train_bc" in stages:
        hours += (stage_hours["collect_bc"] + stage_hours["train_bc"]) * sc["bc"]
    if "dagger_rac" in stages:
        hours += stage_hours["dagger_rac"] * sc["dagger"]
    if "curriculum" in stages:
        hours += stage_hours["curriculum"] * sc["dagger"] * 0.95
    if "ppo" in stages:
        hours += stage_hours["ppo"] * sc["ppo"]
    if "evaluate_val" in stages:
        hours += stage_hours["evaluate_val"]
    if "evaluate_test" in stages:
        hours += stage_hours["evaluate_test"]
    return hours


def _grid_total_h(
    overrides: dict[str, Any],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    grid_file: Path,
) -> tuple[float, dict[str, float]]:
    per_job = {
        c: estimate_condition_hours(c, overrides, grid_path=grid_file)
        for c in conditions
    }
    total = sum(per_job[c] * len(seeds) for c in conditions)
    return total, per_job


def _fit_overrides_to_budget(
    overrides: dict[str, Any],
    *,
    budget_h: float,
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    grid_file: Path,
    actions: list[str],
) -> tuple[dict[str, Any], float, dict[str, float]]:
    """Trim lowest-ROI knobs first; keep expert/DAgger depth as long as possible."""
    trims: list[tuple[str, str, int, int, int]] = [
        # key, step label, step, floor, priority (lower = trim first)
        ("ppo_steps", "ppo_steps", 50_000, 350_000, 0),
        ("dagger_epochs", "dagger_epochs", 5, 85, 1),
        ("dagger_budget", "dagger_budget", 2_000, 30_000, 2),
        ("bc_epochs", "bc_epochs", 10, 170, 3),
        ("dagger_rounds", "dagger_rounds", 1, 5, 4),
    ]
    start_vals = {k: int(overrides[k]) for k, *_ in trims}
    total, per_job = _grid_total_h(overrides, conditions, seeds, grid_file)
    if total <= budget_h:
        return overrides, total, per_job

    for key, label, step, floor, _prio in sorted(trims, key=lambda x: x[4]):
        while total > budget_h and int(overrides[key]) > floor:
            overrides[key] = int(overrides[key]) - step
            total, per_job = _grid_total_h(overrides, conditions, seeds, grid_file)
        if int(overrides[key]) < start_vals[key]:
            actions.append(f"{label}→{overrides[key]} to fit 5d budget")

    return overrides, total, per_job


def plan_five_day_grid(
    root: Path,
    *,
    budget_h: float = BUDGET_HOURS,
    seeds: tuple[int, ...] = (0, 1, 2),
    conditions: tuple[str, ...] = DEFAULT_GRID_CONDITIONS,
    grid_config: str = "configs/grid_7day.yaml",
) -> FiveDayPlan:
    """Tune PROFILE=full overrides to maximize learning within wall-clock budget."""
    findings: list[str] = []
    actions: list[str] = []

    findings.append(
        "day preflight: 16.9h wall; val 0% / test 2%; expert replay 60%; G-BC failed (12%)"
    )
    findings.append("dagger round SR peaked 20% then collapsed — need expert + train budget")

    # Start from spec defaults; day data says imitation ceiling is expert-limited.
    overrides: dict[str, Any] = {
        "n_demos": 2000,
        "bc_epochs": 200,
        "dagger_rounds": 6,
        "dagger_budget": 40000,
        "dagger_epochs": 100,
        "ppo_steps": 700000,
    }
    expert = {"t_label_s": 5.0, "t_validate_s": 13.0}

    actions.append("expert t_label_s→5.0, t_validate_s→13 (60% replay on day)")
    actions.append("full: spec BC/DAgger (200ep, 6×40k, 100 retrain) — learning-limited not compute")

    grid_file = root / grid_config
    overrides, total, per_job = _fit_overrides_to_budget(
        overrides,
        budget_h=budget_h,
        conditions=conditions,
        seeds=seeds,
        grid_file=grid_file,
        actions=actions,
    )

    if total > budget_h:
        findings.append(
            f"WARN: est {total:.0f}h still > {budget_h:.0f}h after trims — launch may spill ~{total - budget_h:.0f}h"
        )

    actions.append(
        f"grid: {len(conditions)} conditions × {len(seeds)} seeds = "
        f"{len(conditions)*len(seeds)} jobs, est {total:.0f}h / {budget_h:.0f}h budget"
    )

    return FiveDayPlan(
        full_overrides=overrides,
        expert_patches=expert,
        grid_conditions=conditions,
        seeds=seeds,
        per_job_h={k: round(v, 2) for k, v in per_job.items()},
        total_h=round(total, 1),
        findings=findings,
        actions=actions,
    )


def apply_five_day_plan(root: Path, plan: FiveDayPlan) -> list[str]:
    import yaml

    changed: list[str] = []
    handoff = root / "configs" / "handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    full_path = handoff / "full_overrides.yaml"
    full_path.write_text(yaml.dump(plan.full_overrides, default_flow_style=False, sort_keys=False))
    changed.append(str(full_path))

    expert_path = root / "configs" / "expert" / "rrt_connect.yaml"
    data = yaml.safe_load(expert_path.read_text())
    for k, v in plan.expert_patches.items():
        if data.get(k) != v:
            data[k] = v
    expert_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    changed.append(str(expert_path))

    report = {
        "budget_h": BUDGET_HOURS,
        "total_h": plan.total_h,
        "per_job_h": plan.per_job_h,
        "grid_jobs": len(plan.grid_conditions) * len(plan.seeds),
        "conditions": list(plan.grid_conditions),
        "seeds": list(plan.seeds),
        "findings": plan.findings,
        "actions": plan.actions,
        "full_overrides": plan.full_overrides,
        "expert_patches": plan.expert_patches,
        "launch": "git tag prereg-v1 && PIPELINE_SKIP_PREREG=1 make grid-5day OUT=runs/grid",
    }
    report_path = root / "runs" / "preflight" / "five_day_plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    changed.append(str(report_path))
    return changed
