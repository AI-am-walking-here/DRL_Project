#!/usr/bin/env python3
"""Stage 3: curriculum + synthetic densification (§7)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from robot_routes.agents.bc_trainer import train_bc
from robot_routes.agents.dagger_rac import collect_round
from robot_routes.agents.policy import load_checkpoint
from robot_routes.data.scene_sets import load_scenes, scene_set_path
from robot_routes.data.schema import merge_shards, write_shard
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.pipeline.notify import notify
from robot_routes.pipeline.stage_progress import write_stage_live
from robot_routes.pipeline.stage_resume import (
    detect_curriculum_resume,
    save_curriculum_state,
)
from robot_routes.utils.config import (
    BCConfig,
    CurriculumConfig,
    DaggerRacConfig,
    DiversityConfig,
    EvalConfig,
    ExpertConfig,
    PolicyConfig,
    load_config,
    load_yaml,
)
from robot_routes.utils.seeding import seed_everything
from robot_routes.utils.device import COLLECT_DEVICE, resolve_device


def eval_level(root: Path, ckpt: Path, level: int, policy_cfg, eval_cfg, div_cfg, delta, n=20):
    name = f"val_L{level}"
    if not scene_set_path(root, name).exists():
        return 0.0
    scenes = load_scenes(root, name)[:n]
    if not scenes:
        return 0.0
    return evaluate_checkpoint(ckpt, scenes, policy_cfg, eval_cfg, div_cfg, delta=delta)[
        "success_rate"
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/curriculum.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/curriculum")
    p.add_argument("--dagger-out", default="runs/dagger_rac")
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--profile", default="full")
    p.add_argument("--rounds-per-level", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--force-restart", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config, CurriculumConfig)
    dagger_cfg = load_config(root / "configs/train/dagger_rac.yaml", DaggerRacConfig)
    if args.profile == "smoke":
        dagger_cfg = dataclasses.replace(dagger_cfg, rounds=1, budget=500)
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    raw_bc = load_yaml(root / "configs/train/bc.yaml")
    policy_cfg = PolicyConfig(**raw_bc.get("policy", {}))
    bc_cfg = load_config(root / "configs/train/bc.yaml", BCConfig)
    eval_cfg = load_config(root / "configs/eval/default.yaml", EvalConfig)
    div_cfg = DiversityConfig(**load_yaml(root / "configs/eval/default.yaml").get("diversity", {}))
    rng = seed_everything(args.seed)
    train_device = resolve_device(args.device)
    collect_device = COLLECT_DEVICE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else out.parent
    dagger_out = Path(args.dagger_out)
    max_level = len(cfg.levels or [[2, 3]]) - 1
    total_steps = max(1, args.rounds_per_level) * (max_level + 1)
    bc_data = dagger_out.parent / "collect" / "demos.h5"
    resume = detect_curriculum_resume(
        out,
        dagger_out,
        dagger_cfg.rounds,
        bc_data,
        total_steps,
        force_restart=args.force_restart,
    )
    if resume.resumed:
        msg = (
            f"resuming curriculum from step {resume.start_step + 1}/{total_steps}"
            if resume.start_step < total_steps
            else "curriculum complete — finalizing"
        )
        print(msg)
        notify(
            run_dir,
            "curriculum_resume",
            start_step=resume.start_step,
            completed_through=resume.completed_through,
        )

    if resume.start_step >= total_steps:
        save_curriculum_state(out, resume.level, cfg.synthetic_obs, resume.history)
        print(f"curriculum complete level={resume.level} synthetic_obs={cfg.synthetic_obs}")
        return

    policy = load_checkpoint(str(resume.policy_ckpt), policy_cfg).to(collect_device)
    policy.eval()
    merged = resume.merged_base
    level = resume.level
    bounds = cfg.levels[level] if cfg.levels else [2, 3]
    history: list[dict] = list(resume.history)
    for step in range(resume.start_step, total_steps):
        sr = eval_level(
            root,
            out / "best.pt" if (out / "best.pt").exists() else resume.policy_ckpt,
            level,
            policy_cfg,
            eval_cfg,
            div_cfg,
            args.delta,
        )
        history.append({"level": level, "success": sr})
        if sr >= cfg.promote and level < max_level:
            level += 1
            bounds = cfg.levels[level]
        elif sr < cfg.demote and level > 0:
            level -= 1
            bounds = cfg.levels[level]
        env = PandaReachEnv()
        dcfg = dataclasses.replace(
            dagger_cfg,
            rounds=1,
            budget=min(dagger_cfg.budget, 2000 if args.profile == "smoke" else dagger_cfg.budget),
        )
        rows = collect_round(
            env,
            policy,
            expert,
            dcfg,
            rng,
            level=level,
            delta_reroute=args.delta,
        )
        write_stage_live(
            run_dir,
            job="curriculum",
            phase=f"step_{step}_collect",
            current=step + 1,
            total=total_steps,
            unit="step",
            desc=f"curriculum step {step + 1}/{total_steps} collect",
        )
        shard = out / f"curriculum_{step}.h5"
        write_shard(shard, rows, [env.scene.to_json()])
        merge_shards([merged, shard], out / f"merged_cur_{step}.h5")
        merged = out / f"merged_cur_{step}.h5"
        policy = train_bc(
            merged,
            dataclasses.replace(bc_cfg, epochs=5 if args.profile == "smoke" else bc_cfg.epochs),
            policy_cfg,
            out / f"ckpt_{step}",
            train_device,
            rng,
        )
        policy.eval()
        policy = policy.to(collect_device)
        shutil.copy(out / f"ckpt_{step}/best.pt", out / "best.pt")
        save_curriculum_state(out, level, cfg.synthetic_obs, history)
        if step >= 2 and args.profile == "smoke":
            break
    save_curriculum_state(out, level, cfg.synthetic_obs, history)
    print(f"curriculum complete level={level} synthetic_obs={cfg.synthetic_obs}")


if __name__ == "__main__":
    main()
