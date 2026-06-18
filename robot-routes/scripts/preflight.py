#!/usr/bin/env python3
"""Preflight checks before unattended training (§11.7.4 ops)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.data.scene_sets import verify_scene_sets
from robot_routes.pipeline.calibration import load_delta
from robot_routes.pipeline.plan_progress import render_plan_dashboard, snapshot_plan_progress
from robot_routes.utils.device import resolve_device


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    line = f"[{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Preflight checks + plan progress")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--no-plan", action="store_true", help="skip plan progress bars")
    args = p.parse_args()

    root = Path(args.root).resolve()
    ok_all = True

    if not args.no_plan:
        plan = snapshot_plan_progress(root)
        print(render_plan_dashboard(plan))
        print()
        print("checks:")
        print()

    venv_py = root / ".venv" / "bin" / "python"
    ok_all &= check("venv", venv_py.is_file(), str(venv_py))

    try:
        import torch  # noqa: F401
        import mujoco  # noqa: F401

        dev = resolve_device("auto")
        cuda = dev.type == "cuda"
        ok_all &= check("imports", True, f"torch+mujoco; device={dev}")
        ok_all &= check("gpu", cuda, "CUDA required for full/day runs")
    except Exception as e:
        ok_all &= check("imports", False, str(e))

    st = os.statvfs(root)
    free_gb = (st.f_bavail * st.f_frsize) / 2**30
    ok_all &= check("disk", free_gb >= 50, f"{free_gb:.0f} GB free (need ≥50)")

    try:
        delta = load_delta(root)
        ok_all &= check("calibration", True, f"delta={delta:.4f}")
    except Exception as e:
        ok_all &= check("calibration", False, str(e))

    manifest = root / "data/scenes/manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        profiles = list(data.get("profiles", {}).keys())
        ok_all &= check("scene manifest", True, f"profiles={profiles}")
        for prof in ("smoke", "medium", "day", "full"):
            if prof in profiles:
                try:
                    verify_scene_sets(root, profile=prof)
                    check(f"scenes:{prof}", True)
                except Exception as e:
                    ok_all &= check(f"scenes:{prof}", False, str(e))
            else:
                check(f"scenes:{prof}", False, "missing — run make scene-sets PROFILE=" + prof)
                if prof in ("medium", "full"):
                    ok_all = False
    else:
        ok_all &= check("scene manifest", False, "missing")

    try:
        out = subprocess.check_output(["git", "tag", "-l", "prereg-v1"], cwd=root, text=True).strip()
        check("prereg tag", bool(out), "optional for preflight; required for PROFILE=full grid")
    except Exception:
        check("prereg tag", False, "git unavailable")

    if shutil.which("nvidia-smi"):
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
        check("nvidia-smi", True, smi)
    else:
        ok_all &= check("nvidia-smi", False)

    print()
    if ok_all:
        print("preflight: READY for medium/day preflight runs")
        print("  make test-medium   # ~3–5 h  rac_noreroute")
        print("  make test-day      # ~14–18 h full condition")
        print("  make prep-full     # scene-sets PROFILE=full (hours) + grid before travel")
        print("  make watch-plan    # live section progress bars")
    else:
        print("preflight: fix FAIL items before unattended runs")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
