#!/usr/bin/env python3
"""Multi-GPU grid launcher (§10.5.3)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.conditions import condition_spec, load_grid, stage_list
from robot_routes.utils.config import ComputeConfig
from robot_routes.utils.device import project_python
from robot_routes.utils.gpu_alloc import GpuAllocator, GpuLease
from robot_routes.utils.gpu_oom import (
    DEFAULT_GRID_OOM_RETRIES,
    cuda_alloc_env,
    read_oom_backoff,
)


def dep_satisfied(root: Path, dep: str, seed: int, runs_root: Path) -> bool:
    # dep format: rac_noreroute.curriculum.eval_val
    parts = dep.split(".")
    cond = parts[0]
    run_dir = runs_root / f"{cond}_seed{seed}"
    if "eval_val" in parts:
        return (run_dir / "eval/val_eval.json").exists() and (
            run_dir / "evaluate_val.stamp"
        ).exists()
    return (run_dir / "COMPLETED").exists()


@dataclass(frozen=True)
class GridJob:
    name: str
    seed: int
    run_dir: Path
    requires: list[str]
    exclusive_gpu: bool = False


@dataclass
class RunningJob:
    job: GridJob
    proc: subprocess.Popen[bytes]
    lease: GpuLease | None


def job_requires_wait(job: GridJob, root: Path, runs_root: Path, profile: str) -> bool:
    if profile == "smoke" or not job.requires:
        return False
    return not all(dep_satisfied(root, dep, job.seed, runs_root) for dep in job.requires)


def mark_completed_if_done(job: GridJob) -> None:
    st_path = job.run_dir / "pipeline_state.json"
    if not st_path.exists():
        return
    st = json.loads(st_path.read_text())
    if st.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED":
        (job.run_dir / "COMPLETED").touch()


def launch_job(
    root: Path,
    runs_root: Path,
    profile: str,
    py: str,
    allocator: GpuAllocator,
    compute: ComputeConfig,
    job: GridJob,
) -> RunningJob:
    lease = allocator.acquire(
        mem_required_gb=compute.mem_required_gb,
        timeout_s=30,
        exclusive=job.exclusive_gpu,
    )
    env = os.environ.copy()
    env.update(cuda_alloc_env())
    env["PYTHONPATH"] = "src"
    env.setdefault("MUJOCO_GL", "egl")
    if lease:
        env.update(lease.env())
    job.run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        "scripts/run_pipeline.py",
        "--seed",
        str(job.seed),
        "--condition",
        job.name,
        "--profile",
        profile,
        "--out",
        str(runs_root),
        "--device",
        "auto",
    ]
    print(
        f"launching {' '.join(cmd)}"
        + (f" gpu={lease.physical_id}" if lease else " cpu")
        + (" exclusive" if job.exclusive_gpu else "")
    )
    proc = subprocess.Popen(cmd, cwd=root, env=env)
    return RunningJob(job=job, proc=proc, lease=lease)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/grid.yaml")
    p.add_argument("--runs-root", default="runs/grid")
    p.add_argument("--profile", default="smoke")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    py = project_python(root)
    grid = load_grid(root)
    compute = ComputeConfig()
    import multiprocessing

    n_cores = multiprocessing.cpu_count()
    allocator = GpuAllocator(jobs_per_gpu=compute.jobs_per_gpu)
    try:
        n_gpu = len(allocator._candidates()) if allocator._nvml else 0
    except Exception:
        n_gpu = 0
    cap = min(compute.jobs_per_gpu * max(n_gpu, 1), max(n_cores // 8, 1))
    print(f"grid launcher: cap={cap} (cores={n_cores}, gpus={n_gpu}, python={py})")
    runs_root = Path(args.runs_root)
    jobs: list[GridJob] = []
    for cond in grid.get("priority", []):
        name = cond if isinstance(cond, str) else cond["name"]
        for seed in grid.get("seeds", [0, 1, 2]):
            run_dir = runs_root / f"{name}_seed{seed}"
            if (run_dir / "pipeline_state.json").exists():
                st = json.loads((run_dir / "pipeline_state.json").read_text())
                if st.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED":
                    continue
            spec = condition_spec(grid, name)
            stages = stage_list(spec)
            requires = list(spec.get("requires") or [])
            if not requires and "ppo" in stages:
                requires = list(grid.get("ppo", {}).get("requires", []) or [])
            jobs.append(
                GridJob(name=name, seed=seed, run_dir=run_dir, requires=requires)
            )

    def oom_retry_job(job: GridJob) -> GridJob | None:
        bo = read_oom_backoff(job.run_dir)
        if not bo or bo.get("consumed"):
            return None
        if int(bo.get("retries", 0)) > DEFAULT_GRID_OOM_RETRIES:
            return None
        bo["consumed"] = True
        (job.run_dir / "oom_backoff.json").write_text(json.dumps(bo, indent=2))
        print(
            f"OOM backoff: re-queue {job.name} seed {job.seed} "
            f"(stage={bo.get('stage')}, exclusive GPU)"
        )
        return GridJob(
            name=job.name,
            seed=job.seed,
            run_dir=job.run_dir,
            requires=job.requires,
            exclusive_gpu=True,
        )

    ready: deque[GridJob] = deque()
    blocked: deque[GridJob] = deque()
    for job in jobs:
        if job_requires_wait(job, root, runs_root, args.profile):
            blocked.append(job)
        else:
            ready.append(job)

    running: list[RunningJob] = []
    dep_deadline = time.time() + grid.get("dep_timeout_h", 48) * 3600

    while ready or blocked or running:
        for rjob in list(running):
            code = rjob.proc.poll()
            if code is None:
                continue
            if rjob.lease:
                rjob.lease.release()
            mark_completed_if_done(rjob.job)
            print(f"finished {rjob.job.name} seed {rjob.job.seed} exit={code}")
            if code != 0:
                retry = oom_retry_job(rjob.job)
                if retry is not None:
                    ready.appendleft(retry)
            running.remove(rjob)

        still_blocked: deque[GridJob] = deque()
        while blocked:
            job = blocked.popleft()
            if job_requires_wait(job, root, runs_root, args.profile):
                still_blocked.append(job)
            else:
                ready.append(job)
        blocked = still_blocked

        while len(running) < cap and ready:
            job = ready.popleft()
            running.append(
                launch_job(root, runs_root, args.profile, py, allocator, compute, job)
            )

        if blocked and not running and not ready:
            if time.time() > dep_deadline:
                print("dependency timeout; remaining jobs:", [(j.name, j.seed) for j in blocked])
                break
            waiting = sorted({(j.name, j.seed, j.requires) for j in blocked})
            print(f"waiting deps for {len(blocked)} job(s), e.g. {waiting[0]}")
            time.sleep(60)
        elif running or blocked:
            time.sleep(5)

    allocator.shutdown()


if __name__ == "__main__":
    main()
