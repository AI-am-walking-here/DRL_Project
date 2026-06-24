#!/usr/bin/env python3
"""Stage 1: BC demo collection with shard-per-worker merge (§5.1)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from robot_routes.contracts import PathTracker, Transition
from robot_routes.data.schema import merge_shards, write_shard
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.oracle import label as expert_label
from robot_routes.expert.rollout import expert_solves_scene
from robot_routes.pipeline.artifacts import git_hash
from robot_routes.pipeline.stage_progress import write_stage_live
from robot_routes.pipeline.stage_resume import count_demos_h5, detect_collect_resume
from robot_routes.utils.config import BCConfig, ExpertConfig, load_config
from robot_routes.utils.seeding import seed_everything


def collect_episode(env, expert, rng, settle_steps=5):
    obs, info = env.reset(
        seed=int(rng.integers(2**31)), options={"level": 0, "level_bounds": [2, 3]}
    )
    scene = env.scene
    planner_seed = int(rng.integers(2**31))
    if not expert_solves_scene(env, expert, scene, planner_seed=planner_seed, settle_steps=settle_steps):
        return None, None
    obs, info = env.reset(options={"scene": scene})
    path = expert.plan(
        info["q"], scene, planner_seed, time_budget_s=expert.cfg.t_validate_s
    )
    if path is None:
        return None, None
    tracker = PathTracker()
    rows = []
    for _ in range(env.cfg.horizon):
        a = expert_label(info["q"], path, tracker, expert.cfg.lookahead)
        rows.append(
            Transition(
                obs.astype(np.float32),
                a.astype(np.float32),
                info["q"].astype(np.float32),
                info["ee_pos"].astype(np.float32),
                False,
                "full_demo",
                0,
                scene.level,
            )
        )
        obs, _, term, trunc, info = env.step(a)
        if info["success"]:
            for _ in range(settle_steps):
                rows.append(
                    Transition(
                        obs.astype(np.float32),
                        np.zeros(7, np.float32),
                        info["q"].astype(np.float32),
                        info["ee_pos"].astype(np.float32),
                        False,
                        "full_demo",
                        0,
                        scene.level,
                    )
                )
                obs, _, term, trunc, info = env.step(np.zeros(7))
            break
        if term or trunc:
            break
    if not info.get("success"):
        return None, None
    return rows, scene.to_json()


def worker_collect(args_tuple):
    worker_id, n_target, seed, out_dir, root_str = args_tuple
    root = Path(root_str)
    cfg = load_config(root / "configs/train/bc.yaml", BCConfig)
    expert_cfg = load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig)
    rng = seed_everything(seed + worker_id * 10007)
    env = PandaReachEnv()
    expert = ExpertOracle(expert_cfg)
    all_rows = []
    scenes = []
    n = 0
    attempts = 0
    while n < n_target and attempts < n_target * 10:
        attempts += 1
        ep, sj = collect_episode(env, expert, rng, settle_steps=5)
        if ep is None:
            continue
        for r in ep:
            all_rows.append(
                Transition(r.obs, r.action, r.q, r.ee_pos, r.done, r.segment, n, r.level)
            )
        scenes.append(sj)
        n += 1
    shard = Path(out_dir) / f"demos_w{worker_id}.h5"
    write_shard(shard, all_rows, scenes, git_hash=git_hash(root), worker_id=worker_id)
    return str(shard), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/bc.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/bc_collect")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--force-restart", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config, BCConfig)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_workers = args.workers or min(8, max(1, mp.cpu_count() // 2))
    per_worker = max(1, cfg.n_demos // n_workers)
    remainder = cfg.n_demos - per_worker * n_workers
    targets = [per_worker + (1 if i < remainder else 0) for i in range(n_workers)]
    run_dir = Path(args.run_dir) if args.run_dir else out.parent
    resume = detect_collect_resume(
        out, cfg.n_demos, n_workers, force_restart=args.force_restart
    )
    if resume.done:
        print(f"collect_bc: demos.h5 complete ({cfg.n_demos} demos) — skipping")
        return
    if resume.resumed:
        print(
            f"collect_bc: resuming workers {resume.pending_workers or 'merge-only'} "
            f"({len(resume.pending_workers)} pending)"
        )

    if resume.merge_only:
        shards = sorted(out.glob("demos_w*.h5"))
        merge_shards(shards, out / "demos.h5")
        print(f"merged {len(shards)} shards → demos.h5")
        return

    if n_workers == 1 or cfg.n_demos < 4:
        if 0 not in resume.pending_workers:
            shards = sorted(out.glob("demos_w*.h5"))
            merge_shards(shards, out / "demos.h5")
            print("collected (cached shard)")
            return
        shard, n = worker_collect((0, cfg.n_demos, args.seed, str(out), str(root)))
        merge_shards([Path(shard)], out / "demos.h5")
        print(f"collected {n} demos (single worker)")
        return

    ctx = mp.get_context("spawn")
    pending = resume.pending_workers
    with ctx.Pool(len(pending)) as pool:
        results = pool.map(
            worker_collect,
            [(i, targets[i], args.seed, str(out), str(root)) for i in pending],
        )
    shards = sorted(out.glob("demos_w*.h5"))
    total = sum(count for _, count in results)
    for i, shard in enumerate(shards):
        write_stage_live(
            run_dir,
            job="collect_bc",
            phase="merge",
            current=i + 1,
            total=len(shards),
            unit="shards",
            desc=f"merging collect shards {i + 1}/{len(shards)}",
        )
    merge_shards(shards, out / "demos.h5")
    merged_total = count_demos_h5(out / "demos.h5")
    print(
        f"collected {total} new demos from {len(pending)} workers; "
        f"merged {len(shards)} shards → {merged_total} total"
    )


if __name__ == "__main__":
    main()
