#!/usr/bin/env python3
"""Wait for medium preflight → review → launch day preflight (unattended)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.data.scene_sets import load_manifest, verify_scene_sets
from robot_routes.pipeline.handoff_remediate import remediation_to_dict, run_remediation_cycle
from robot_routes.pipeline.stage_progress import snapshot_run_progress
from robot_routes.utils.device import project_python
from robot_routes.utils.progress import fmt_eta

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "runs" / "preflight"
MEDIUM_RUN = PREFLIGHT / "rac_noreroute_seed0"
DAY_RUN = PREFLIGHT / "full_seed0"
LOG_PATH = PREFLIGHT / "handoff.log"
REPORT_PATH = PREFLIGHT / "handoff_report.json"
REMEDIATION_PATH = PREFLIGHT / "handoff_remediation.json"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def medium_finished(run_dir: Path) -> bool:
    if (run_dir / "COMPLETED").exists():
        return True
    st = run_dir / "pipeline_state.json"
    if not st.exists():
        return False
    try:
        data = json.loads(st.read_text())
        return data.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED"
    except (json.JSONDecodeError, OSError):
        return False


def medium_failed(run_dir: Path) -> bool:
    st = run_dir / "pipeline_state.json"
    if not st.exists():
        return False
    data = json.loads(st.read_text())
    for name, entry in data.get("stages", {}).items():
        if entry.get("status") == "FAILED":
            return True
    return False


def wait_medium(run_dir: Path, poll_s: float = 60.0, timeout_h: float = 6.0) -> bool:
    deadline = time.time() + timeout_h * 3600
    log(f"waiting for medium preflight: {run_dir}")
    while time.time() < deadline:
        snap = snapshot_run_progress(run_dir)
        if medium_finished(run_dir):
            log(f"medium COMPLETED ({snap['overall_pct']:.1f}%)")
            return True
        if medium_failed(run_dir):
            log(f"medium FAILED at stage {snap.get('current_stage')}")
            return False
        log(
            f"medium in progress: {snap['overall_pct']:.1f}% "
            f"stage={snap.get('current_stage')} eta={fmt_eta(snap['eta_s'])}"
        )
        time.sleep(poll_s)
    log("medium wait timed out")
    return False


def review_medium(run_dir: Path) -> dict:
    """Summarize medium run artifacts for handoff report."""
    report: dict = {"run_dir": str(run_dir), "ts": datetime.now(timezone.utc).isoformat()}
    st_path = run_dir / "pipeline_state.json"
    if st_path.exists():
        report["pipeline_state"] = json.loads(st_path.read_text())
    snap = snapshot_run_progress(run_dir)
    report["overall_pct"] = snap.get("overall_pct")
    report["stages_completed"] = snap.get("stages_completed")
    report["stages_total"] = snap.get("stages_total")

    eval_path = run_dir / "eval" / "val_eval.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        report["eval"] = {
            k: ev.get(k)
            for k in (
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "validity_frac",
                "n_scenes",
            )
            if k in ev
        }

    events = report.get("pipeline_state", {}).get("events", [])
    report["warnings"] = [
        e for e in events if e.get("event") in ("G-BC", "G-BC_FAILED", "G-BC_preflight_baseline")
    ]
    report["ok"] = medium_finished(run_dir) and not medium_failed(run_dir)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log(f"handoff report → {REPORT_PATH}")
    if report.get("eval"):
        log(f"medium val success_rate={report['eval'].get('success_rate')}")
    if report.get("warnings"):
        log(f"medium warnings: {len(report['warnings'])} gate events (see report)")
    return report


def remediate_from_medium(run_dir: Path, report: dict) -> bool:
    """Analyze medium data, apply bounded fixes; verify is advisory (never blocks launch)."""
    log("=== remediation: analyze medium results ===")
    result = run_remediation_cycle(ROOT, run_dir, report)
    REMEDIATION_PATH.write_text(json.dumps(remediation_to_dict(result), indent=2))
    for f in result.findings:
        log(f"  finding [{f.severity}] {f.id}: {f.detail}")
    for action in result.actions:
        log(f"  action: {action}")
    if result.files_changed:
        log(f"  files changed: {', '.join(result.files_changed)}")
    if result.verify_ok:
        log("  verify: OK")
    else:
        log("  verify: FAILED (advisory only — day launch will proceed)")
        log(f"  verify log tail:\n{result.verify_log[-800:]}")
    return True


def scenes_ready(profile: str) -> bool:
    try:
        verify_scene_sets(ROOT, profile=profile)
        return True
    except Exception:
        return False


def wait_scenes(profile: str, poll_s: float = 120.0, timeout_h: float = 8.0) -> bool:
    if scenes_ready(profile):
        log(f"scene sets profile={profile} already verified")
        return True
    deadline = time.time() + timeout_h * 3600
    status = ROOT / "data" / "scenes" / ".generation_status.json"
    log(f"waiting for scene-sets PROFILE={profile} (up to {timeout_h:.0f}h)")
    while time.time() < deadline:
        if scenes_ready(profile):
            log(f"scene sets profile={profile} verified")
            return True
        if status.exists():
            try:
                live = json.loads(status.read_text())
                if live.get("profile") == profile:
                    log(
                        f"scene gen: {live.get('desc', live.get('phase'))} "
                        f"{live.get('current', '?')}/{live.get('total', '?')} "
                        f"({live.get('pct', 0):.1f}%)"
                    )
            except (json.JSONDecodeError, OSError):
                pass
        else:
            manifest = load_manifest(ROOT)
            if profile in manifest.get("profiles", {}):
                log(f"scene manifest has {profile}; verifying pool_rl…")
        time.sleep(poll_s)
    log(f"scene-sets PROFILE={profile} timed out")
    return False


def gpu_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_pipeline.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        pids = [p for p in out.split() if p]
        return len(pids) > 0
    except (subprocess.CalledProcessError, OSError):
        return False


def launch_day(seed: int = 0) -> int:
    if DAY_RUN.exists() and (DAY_RUN / "COMPLETED").exists():
        log(f"day run already COMPLETED at {DAY_RUN} — skipping launch")
        return 0
    if gpu_busy():
        log("GPU still busy with another pipeline — waiting 5 min")
        time.sleep(300)
        if gpu_busy():
            log("WARNING: GPU still busy — launching day anyway (preflight handoff)")

    py = project_python(ROOT)
    log_path = PREFLIGHT / "day.log"
    env = os.environ.copy()
    env["PIPELINE_SKIP_PREREG"] = "1"
    env["PYTHONPATH"] = "src"
    env.setdefault("MUJOCO_GL", "egl")

    cmd = [
        py,
        "scripts/run_pipeline.py",
        "--seed",
        str(seed),
        "--condition",
        "full",
        "--profile",
        "day",
        "--out",
        str(PREFLIGHT),
        "--device",
        "auto",
    ]
    log(f"launching day preflight: {' '.join(cmd)}")
    log(f"log → {log_path}")
    with log_path.open("a") as lf:
        lf.write(f"\n--- handoff launch {_ts()} ---\n")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (PREFLIGHT / "day.pid").write_text(str(proc.pid))
    log(f"day preflight started pid={proc.pid}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Unattended medium→day handoff")
    p.add_argument("--medium-run", type=Path, default=MEDIUM_RUN)
    p.add_argument("--skip-medium-wait", action="store_true")
    p.add_argument("--skip-scenes-wait", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    log("=== autonomous handoff started ===")

    if not args.skip_medium_wait:
        if not wait_medium(args.medium_run):
            log("ABORT: medium did not complete successfully — day test not started")
            review_medium(args.medium_run)
            return 1

    report = review_medium(args.medium_run)
    if not report.get("ok"):
        log("ABORT: medium review failed")
        return 1

    if not remediate_from_medium(args.medium_run, report):
        log("WARNING: remediation cycle error — continuing to day launch")

    if not args.skip_scenes_wait:
        if not wait_scenes("day"):
            log("ABORT: day scene sets not ready")
            return 1

    st = os.statvfs(ROOT)
    free_gb = (st.f_bavail * st.f_frsize) / 2**30
    if free_gb < 80:
        log(f"WARNING: only {free_gb:.0f} GB free (day run needs headroom)")

    return launch_day(seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
