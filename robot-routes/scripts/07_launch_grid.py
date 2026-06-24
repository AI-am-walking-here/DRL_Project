#!/usr/bin/env python3
"""Sequential single-GPU grid launcher.

Runs each (condition, seed) job one at a time on the local GPU via
``run_pipeline.py``. Completed jobs are skipped; jobs whose cross-condition
dependencies are unmet are deferred until satisfied (with a timeout). This
replaces the former multi-GPU NVML-leasing scheduler: a class project on a
single 4090 does not benefit from the added complexity, and sequential runs
are far easier to reason about and reproduce.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.conditions import condition_spec, load_grid, stage_list
from robot_routes.utils.device import project_python


@dataclass(frozen=True)
class GridJob:
    name: str
    seed: int
    run_dir: Path
    requires: list[str] = field(default_factory=list)


def job_completed(run_dir: Path) -> bool:
    state_path = run_dir / "pipeline_state.json"
    if (run_dir / "COMPLETED").exists():
        return True
    if not state_path.exists():
        return False
    state = json.loads(state_path.read_text())
    return state.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED"


def dep_satisfied(dep: str, seed: int, runs_root: Path) -> bool:
    # dep format: "<condition>.curriculum.eval_val"
    cond = dep.split(".")[0]
    run_dir = runs_root / f"{cond}_seed{seed}"
    if "eval_val" in dep:
        return (run_dir / "eval/val_eval.json").exists() and (
            run_dir / "evaluate_val.stamp"
        ).exists()
    return (run_dir / "COMPLETED").exists()


def deps_ready(job: GridJob, runs_root: Path, profile: str) -> bool:
    if profile == "smoke" or not job.requires:
        return True
    return all(dep_satisfied(dep, job.seed, runs_root) for dep in job.requires)


def build_jobs(grid: dict, runs_root: Path) -> list[GridJob]:
    jobs: list[GridJob] = []
    for cond in grid.get("priority", []):
        name = cond if isinstance(cond, str) else cond["name"]
        spec = condition_spec(grid, name)
        stages = stage_list(spec)
        requires = list(spec.get("requires") or [])
        if not requires and "ppo" in stages:
            requires = list(grid.get("ppo", {}).get("requires", []) or [])
        for seed in grid.get("seeds", [0, 1, 2]):
            jobs.append(
                GridJob(
                    name=name,
                    seed=seed,
                    run_dir=runs_root / f"{name}_seed{seed}",
                    requires=requires,
                )
            )
    return jobs


def run_job(root: Path, runs_root: Path, profile: str, py: str, job: GridJob, device: str) -> int:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        "scripts/run_pipeline.py",
        "--seed", str(job.seed),
        "--condition", job.name,
        "--profile", profile,
        "--out", str(runs_root),
        "--device", device,
    ]
    print(f"[grid] launching {job.name} seed {job.seed}: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=root).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/grid.yaml", help="grid spec YAML")
    p.add_argument("--runs-root", default="runs/grid")
    p.add_argument("--profile", default="smoke")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = project_python(root)
    grid = load_grid(root, args.config)
    runs_root = Path(args.runs_root)

    pending = build_jobs(grid, runs_root)
    dep_timeout_s = float(grid.get("dep_timeout_h", 48)) * 3600
    print(f"[grid] {len(pending)} job(s) from {args.config} (python={py})", flush=True)

    deadline = time.time() + dep_timeout_s
    completed, failed, skipped = 0, 0, 0
    while pending:
        runnable = [j for j in pending if deps_ready(j, runs_root, args.profile)]
        if not runnable:
            if time.time() > deadline:
                blocked = [(j.name, j.seed) for j in pending]
                print(f"[grid] dependency timeout; abandoning {blocked}", flush=True)
                failed += len(pending)
                break
            print(f"[grid] waiting on deps for {len(pending)} job(s)…", flush=True)
            time.sleep(60)
            continue

        job = runnable[0]
        pending.remove(job)
        if job_completed(job.run_dir):
            print(f"[grid] skip {job.name} seed {job.seed} (already COMPLETED)", flush=True)
            skipped += 1
            continue

        code = run_job(root, runs_root, args.profile, py, job, args.device)
        if code == 0:
            completed += 1
        else:
            failed += 1
            print(f"[grid] {job.name} seed {job.seed} exited {code}", flush=True)
        deadline = time.time() + dep_timeout_s  # progress made → reset dep clock

    print(
        f"[grid] done: {completed} completed, {skipped} skipped, {failed} failed",
        flush=True,
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
