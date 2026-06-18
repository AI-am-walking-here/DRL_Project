#!/usr/bin/env python3
"""Wait for day preflight → analyze → write full_overrides.yaml (no grid launch by default)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.handoff_remediate import (
    day_report,
    remediation_to_dict,
    run_day_to_full_cycle,
)
from robot_routes.pipeline.stage_progress import snapshot_run_progress
from robot_routes.utils.progress import fmt_eta

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "runs" / "preflight"
DAY_RUN = PREFLIGHT / "full_seed0"
LOG_PATH = PREFLIGHT / "day_to_grid.log"
REPORT_PATH = PREFLIGHT / "day_to_grid_report.json"
REMEDIATION_PATH = PREFLIGHT / "day_to_grid_remediation.json"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def day_finished(run_dir: Path) -> bool:
    return (run_dir / "COMPLETED").exists()


def day_failed(run_dir: Path) -> bool:
    st = run_dir / "pipeline_state.json"
    if not st.exists():
        return False
    data = json.loads(st.read_text())
    return any(v.get("status") == "FAILED" for v in data.get("stages", {}).values())


def wait_day(run_dir: Path, poll_s: float = 120.0, timeout_h: float = 36.0) -> bool:
    deadline = time.time() + timeout_h * 3600
    log(f"waiting for day preflight: {run_dir}")
    while time.time() < deadline:
        if day_finished(run_dir):
            log("day COMPLETED")
            return True
        if day_failed(run_dir):
            log("day FAILED")
            return False
        snap = snapshot_run_progress(run_dir)
        log(
            f"day in progress: {snap.get('overall_pct', 0):.1f}% "
            f"stage={snap.get('current_stage')} eta={fmt_eta(snap.get('eta_s', 0))}"
        )
        time.sleep(poll_s)
    log("day wait timed out")
    return False


def analyze_and_write(run_dir: Path) -> int:
    log("=== day → full grid: analyze results ===")
    report = day_report(run_dir)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"artifact report → {REPORT_PATH}")

    result, overrides, eta = run_day_to_full_cycle(ROOT, run_dir)
    payload = remediation_to_dict(result)
    payload["full_overrides"] = overrides
    payload["grid_eta"] = eta
    payload["grid_config"] = "configs/grid_7day.yaml"
    payload["launch_when_ready"] = (
        "Review day_to_grid_report.json + full_overrides.yaml, then:\n"
        "  git tag prereg-v1\n"
        "  PIPELINE_SKIP_PREREG=1 make grid-7day OUT=runs/grid"
    )
    REMEDIATION_PATH.write_text(json.dumps(payload, indent=2))
    log(f"remediation → {REMEDIATION_PATH}")

    for f in result.findings:
        log(f"  finding [{f.severity}] {f.id}: {f.detail}")
    for action in result.actions:
        log(f"  action: {action}")
    log(f"  full_overrides: {overrides}")
    if eta.get("ok"):
        log(
            f"  grid ETA (from day wall {eta['day_wall_h']}h): "
            f"{eta['sequential_h']}h sequential / {eta['sequential_d']}d "
            f"for {eta['grid_jobs']} jobs"
        )
        log(f"  per_job_h: {eta.get('per_job_h')}")
    log(f"  verify: {'OK' if result.verify_ok else 'advisory FAIL (see log)'}")
    if not result.verify_ok:
        log(f"  verify log:\n{result.verify_log[-600:]}")
    log("=== done — grid NOT launched (review overrides first) ===")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Day preflight → full grid param handoff")
    p.add_argument("--day-run", type=Path, default=DAY_RUN)
    p.add_argument("--skip-wait", action="store_true", help="Analyze immediately if day done")
    p.add_argument("--timeout-h", type=float, default=36.0)
    args = p.parse_args()

    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    log("=== day → full grid handoff started ===")

    if not args.skip_wait:
        if not wait_day(args.day_run):
            log("ABORT: day did not complete — no full_overrides written")
            return 1

    if not day_finished(args.day_run):
        log("ABORT: day not COMPLETED")
        return 1

    return analyze_and_write(args.day_run)


if __name__ == "__main__":
    sys.exit(main())
