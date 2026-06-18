#!/usr/bin/env python3
"""Analyze day preflight and write 5-day grid training plan."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.five_day_plan import apply_five_day_plan, plan_five_day_grid
from robot_routes.pipeline.handoff_remediate import analyze_day, remediation_to_dict


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "runs" / "preflight" / "full_seed0"
    if not (run_dir / "COMPLETED").exists():
        print("ABORT: day preflight not COMPLETED")
        return 1

    print("=== day analysis ===")
    findings = analyze_day(root, run_dir)
    for f in findings:
        print(f"  [{f.severity}] {f.id}: {f.detail}")

    print("\n=== 5-day plan ===")
    plan = plan_five_day_grid(root)
    for f in plan.findings:
        print(f"  finding: {f}")
    for a in plan.actions:
        print(f"  action: {a}")
    print(f"  per_job_h: {plan.per_job_h}")
    print(f"  total: {plan.total_h}h (budget {118}h)")
    print(f"  full_overrides: {plan.full_overrides}")

    paths = apply_five_day_plan(root, plan)
    print("\nwritten:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
