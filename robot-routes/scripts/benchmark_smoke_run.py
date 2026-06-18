#!/usr/bin/env python3
"""Run smoke pipeline under monitoring; emit per-stage timing, RAM, CPU, GPU, parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STAGES = [
    "setup",
    "scene_sets",
    "calibrate_delta",
    "collect_bc",
    "train_bc",
    "dagger_rac",
    "curriculum",
    "evaluate_val",
    "ppo",
    "evaluate_test",
    "verdicts",
    "report_assets",
]

# Expected parallelism from codebase (§10.5.4, scripts/01_collect_bc_demos.py).
STAGE_PARALLEL_DESIGN: dict[str, dict[str, Any]] = {
    "setup": {"design_workers": 1, "design_note": "preflight only"},
    "scene_sets": {"design_workers": 1, "design_note": "single-process scene verify"},
    "calibrate_delta": {"design_workers": 1, "design_note": "cached delta on smoke"},
    "collect_bc": {
        "design_workers": "min(8, cpu_count//2)",
        "design_note": "multiprocessing.Pool — 8 independent MuJoCo+RRT workers",
    },
    "train_bc": {"design_workers": 1, "design_note": "single GPU/CPU trainer"},
    "dagger_rac": {
        "design_workers": 1,
        "design_note": "serial collect_round; CPU policy inference",
    },
    "curriculum": {"design_workers": 1, "design_note": "serial collect like DAgger"},
    "evaluate_val": {"design_workers": 1, "design_note": "serial scene rollouts"},
    "ppo": {"design_workers": "8 envs, sequential step", "design_note": "n_envs=8 rolled out one-by-one"},
    "evaluate_test": {"design_workers": 1, "design_note": "serial scene rollouts"},
    "verdicts": {"design_workers": 1, "design_note": "JSON aggregation"},
    "report_assets": {"design_workers": 1, "design_note": "plot generation"},
}


def _gpu_stats() -> dict[str, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        util_gpu, util_mem, mem_used, mem_total = [float(x.strip()) for x in out.split(",")]
        return {
            "gpu_util_pct": util_gpu,
            "gpu_mem_util_pct": util_mem,
            "gpu_mem_used_mb": mem_used,
            "gpu_mem_total_mb": mem_total,
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return {
            "gpu_util_pct": 0.0,
            "gpu_mem_util_pct": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
        }


def _proc_tree_rss_mb(root_pid: int) -> tuple[float, int]:
    """Sum RSS of root process and all descendants (MB). Return (rss_mb, n_procs)."""
    try:
        import psutil
    except ImportError:
        return 0.0, 1

    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return 0.0, 0

    procs = [root] + root.children(recursive=True)
    rss = sum(p.memory_info().rss for p in procs if p.is_running()) / (1024 * 1024)
    return rss, len(procs)


def _cpu_pct(root_pid: int) -> float:
    try:
        import psutil

        root = psutil.Process(root_pid)
        procs = [root] + root.children(recursive=True)
        return sum(p.cpu_percent(interval=None) for p in procs if p.is_running())
    except Exception:
        return 0.0


def _active_stage(run_dir: Path) -> str | None:
    st_path = run_dir / "pipeline_state.json"
    if not st_path.exists():
        return None
    data = json.loads(st_path.read_text())
    stages = data.get("stages", {})
    for name in STAGES:
        if stages.get(name, {}).get("status") == "RUNNING":
            return name
    # infer last completed + next pending
    last_done = None
    for name in STAGES:
        st = stages.get(name, {}).get("status")
        if st == "COMPLETED":
            last_done = name
        elif st in ("PENDING", "RUNNING") and last_done:
            return name if st == "RUNNING" else name
    live = run_dir / "stage_live.json"
    if live.exists():
        try:
            job = json.loads(live.read_text()).get("job")
            if job:
                return str(job)
        except json.JSONDecodeError:
            pass
    return last_done


def _collect_shard_workers(run_dir: Path) -> int | None:
    collect = run_dir / "collect"
    if not collect.is_dir():
        return None
    shards = list(collect.glob("demos_w*.h5"))
    return len(shards) if shards else None


def monitor_loop(
    root_pid: int,
    run_dir: Path,
    samples: list[dict[str, Any]],
    stop: list[bool],
    poll_s: float = 1.0,
) -> None:
    # warm cpu_percent
    _cpu_pct(root_pid)
    current_stage: str | None = None
    while not stop[0]:
        stage = _active_stage(run_dir)
        if stage is None:
            stage = current_stage or "startup"
        current_stage = stage
        rss_mb, n_procs = _proc_tree_rss_mb(root_pid)
        sample = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "rss_mb": round(rss_mb, 1),
            "n_procs": n_procs,
            "cpu_pct": round(_cpu_pct(root_pid), 1),
            **_gpu_stats(),
        }
        cw = _collect_shard_workers(run_dir)
        if cw is not None and stage == "collect_bc":
            sample["collect_shards_seen"] = cw
        samples.append(sample)
        time.sleep(poll_s)


def _aggregate_stage(samples: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    pts = [s for s in samples if s.get("stage") == stage]
    if not pts:
        return {}
    rss = [p["rss_mb"] for p in pts]
    cpu = [p["cpu_pct"] for p in pts]
    gpu_u = [p["gpu_util_pct"] for p in pts]
    gpu_m = [p["gpu_mem_used_mb"] for p in pts]
    nproc = [p["n_procs"] for p in pts]
    out: dict[str, Any] = {
        "samples": len(pts),
        "rss_mb_avg": round(sum(rss) / len(rss), 1),
        "rss_mb_peak": round(max(rss), 1),
        "cpu_pct_avg": round(sum(cpu) / len(cpu), 1),
        "cpu_pct_peak": round(max(cpu), 1),
        "gpu_util_pct_avg": round(sum(gpu_u) / len(gpu_u), 1),
        "gpu_util_pct_peak": round(max(gpu_u), 1),
        "gpu_mem_used_mb_avg": round(sum(gpu_m) / len(gpu_m), 1),
        "gpu_mem_used_mb_peak": round(max(gpu_m), 1),
        "n_procs_avg": round(sum(nproc) / len(nproc), 1),
        "n_procs_peak": max(nproc),
    }
    shards = [p.get("collect_shards_seen") for p in pts if "collect_shards_seen" in p]
    if shards:
        out["collect_shards_peak"] = max(shards)
    return out


def _stage_timings(run_dir: Path, wall_start: float) -> dict[str, dict[str, Any]]:
    st_path = run_dir / "pipeline_state.json"
    if not st_path.exists():
        return {}
    data = json.loads(st_path.read_text())
    stages = data.get("stages", {})
    ordered: list[tuple[str, float, str, dict[str, Any]]] = []
    for name in STAGES:
        entry = stages.get(name, {})
        ts = entry.get("updated")
        status = entry.get("status", "PENDING")
        if not ts:
            continue
        ordered.append((name, datetime.fromisoformat(ts).timestamp(), status, entry))
    timings: dict[str, dict[str, Any]] = {}
    prev_t = wall_start
    for name, t, status, entry in ordered:
        timings[name] = {
            "status": status,
            "duration_s": round(max(0.0, t - prev_t), 2),
            "reason": entry.get("reason"),
        }
        prev_t = t
    return timings


def _eval_metrics(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, fname in [("val", "eval/val_eval.json"), ("test", "eval/test_eval.json")]:
        p = run_dir / fname
        if p.exists():
            d = json.loads(p.read_text())
            out[f"{key}_success_rate"] = d.get("success_rate")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0"])
    p.add_argument("--label", required=True)
    p.add_argument("--out-root", default="runs/benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poll-s", type=float, default=1.0)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_root = Path(args.out_root)
    run_dir = run_root / f"full_seed{args.seed}"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PIPELINE_SKIP_PREREG"] = "1"
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    py = root / ".venv" / "bin" / "python"
    cmd = [
        str(py),
        "scripts/run_pipeline.py",
        "--seed",
        str(args.seed),
        "--condition",
        "full",
        "--profile",
        "smoke",
        "--out",
        str(run_root),
        "--device",
        args.device,
    ]

    samples: list[dict[str, Any]] = []
    stop = [False]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=root, env=env)
    import threading

    mon = threading.Thread(
        target=monitor_loop,
        args=(proc.pid, run_dir, samples, stop, args.poll_s),
        daemon=True,
    )
    mon.start()
    rc = proc.wait()
    stop[0] = True
    mon.join(timeout=2.0)
    wall_s = round(time.time() - t0, 2)

    timings = _stage_timings(run_dir, t0)
    segments: dict[str, Any] = {}
    for stage in STAGES:
        seg = {
            **STAGE_PARALLEL_DESIGN.get(stage, {}),
            **timings.get(stage, {}),
            **_aggregate_stage(samples, stage),
        }
        if seg:
            segments[stage] = seg

    report = {
        "label": args.label,
        "device_arg": args.device,
        "cuda_visible": env.get("CUDA_VISIBLE_DEVICES", "unset"),
        "wall_s": wall_s,
        "exit_code": rc,
        "run_dir": str(run_dir),
        "eval": _eval_metrics(run_dir),
        "segments": segments,
        "raw_sample_count": len(samples),
    }
    out_path = run_root / f"benchmark_{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
