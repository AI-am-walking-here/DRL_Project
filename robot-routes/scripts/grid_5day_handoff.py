#!/usr/bin/env python3
"""Generate full scenes if needed → wait → launch 5-day grid (unattended)."""

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
from robot_routes.utils.device import project_python
from robot_routes.utils.progress import fmt_eta

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "runs" / "preflight"
LOG_PATH = PREFLIGHT / "grid_5day_handoff.log"
SCENES_LOG = PREFLIGHT / "full_scenes.log"
GRID_LOG = PREFLIGHT / "grid_5day.log"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def scenes_ready(profile: str) -> bool:
    try:
        verify_scene_sets(ROOT, profile=profile)
        return True
    except Exception:
        return False


def scene_gen_running() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "generate_scene_sets.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(out)
    except (subprocess.CalledProcessError, OSError):
        return False


def start_scene_gen(profile: str = "full") -> None:
    if scenes_ready(profile):
        log(f"scene sets profile={profile} already verified")
        return
    if scene_gen_running():
        log("scene generation already running")
        return
    py = project_python(ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env.setdefault("MUJOCO_GL", "egl")
    cmd = [py, "scripts/generate_scene_sets.py", "--profile", profile]
    log(f"starting scene generation: {' '.join(cmd)}")
    with SCENES_LOG.open("a") as lf:
        lf.write(f"\n--- handoff launch {_ts()} ---\n")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=open(SCENES_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (PREFLIGHT / "full_scenes.pid").write_text(str(proc.pid))
    log(f"scene gen pid={proc.pid} log={SCENES_LOG}")


def wait_scenes(profile: str, poll_s: float = 120.0, timeout_h: float = 8.0) -> bool:
    if scenes_ready(profile):
        log(f"scene sets profile={profile} verified")
        return True
    deadline = time.time() + timeout_h * 3600
    status = ROOT / "data" / "scenes" / ".generation_status.json"
    log(f"waiting for PROFILE={profile} scenes (up to {timeout_h:.0f}h)")
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
        elif profile in load_manifest(ROOT).get("profiles", {}):
            log(f"manifest has {profile}; waiting for pool_rl verification")
        time.sleep(poll_s)
    log(f"scene-sets PROFILE={profile} timed out")
    return False


def gpu_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_pipeline.py|07_launch_grid.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(out)
    except (subprocess.CalledProcessError, OSError):
        return False


def apply_plan() -> None:
    py = project_python(ROOT)
    log("running plan_5day_grid.py")
    subprocess.run([py, "scripts/plan_5day_grid.py"], cwd=ROOT, check=False)


def launch_grid() -> int:
    if gpu_busy():
        log("GPU busy — waiting 5 min before grid launch")
        time.sleep(300)
    py = project_python(ROOT)
    env = os.environ.copy()
    env["PIPELINE_SKIP_PREREG"] = "1"
    env["PYTHONPATH"] = "src"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [
        py,
        "scripts/07_launch_grid.py",
        "--config",
        "configs/grid_7day.yaml",
        "--profile",
        "full",
        "--runs-root",
        "runs/grid",
    ]
    log(f"launching 5-day grid: {' '.join(cmd)}")
    with GRID_LOG.open("a") as lf:
        lf.write(f"\n--- handoff launch {_ts()} ---\n")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=open(GRID_LOG, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (PREFLIGHT / "grid_5day.pid").write_text(str(proc.pid))
    log(f"grid launcher pid={proc.pid} log={GRID_LOG}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Full scenes → 5-day grid handoff")
    p.add_argument("--skip-scenes", action="store_true")
    p.add_argument("--skip-plan", action="store_true")
    p.add_argument("--timeout-h", type=float, default=8.0)
    args = p.parse_args()

    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    log("=== 5-day grid handoff started ===")

    if not args.skip_plan:
        apply_plan()

    if not args.skip_scenes:
        start_scene_gen("full")
        if not wait_scenes("full", timeout_h=args.timeout_h):
            log("ABORT: full scene sets not ready")
            return 1

    st = os.statvfs(ROOT)
    free_gb = (st.f_bavail * st.f_frsize) / 2**30
    if free_gb < 100:
        log(f"WARNING: only {free_gb:.0f} GB free (grid needs headroom)")

    plan_path = PREFLIGHT / "five_day_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        log(
            f"plan: {plan.get('grid_jobs')} jobs, est {plan.get('total_h')}h, "
            f"overrides={plan.get('full_overrides')}"
        )

    return launch_grid()


if __name__ == "__main__":
    sys.exit(main())
