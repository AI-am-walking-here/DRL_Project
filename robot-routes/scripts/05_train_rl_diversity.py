#!/usr/bin/env python3
"""Stage 4: PPO diversity fine-tuning (§8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from robot_routes.agents.policy import load_checkpoint
from robot_routes.agents.ppo_diversity import train_ppo
from robot_routes.contracts import SceneSpec
from robot_routes.pipeline.stage_resume import artifact_complete
from robot_routes.utils.config import PolicyConfig, PPOConfig, load_config, load_yaml
from robot_routes.utils.device import resolve_device
from robot_routes.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/rl_diversity.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--curriculum-ckpt", default="runs/curriculum/best.pt")
    p.add_argument("--pool", default="data/pool_rl.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--force-restart", action="store_true")
    p.add_argument("--run-dir", default="", help="Pipeline run dir for OOM backoff tagging")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config, PPOConfig)
    raw_bc = load_yaml(root / "configs/train/bc.yaml")
    policy_cfg = PolicyConfig(**raw_bc.get("policy", {}))
    seed_everything(args.seed)
    device = resolve_device(args.device)
    policy = load_checkpoint(args.curriculum_ckpt, policy_cfg)
    anchor = load_checkpoint(args.curriculum_ckpt, policy_cfg)
    pool_path = Path(args.pool)
    if pool_path.exists():
        data = json.loads(pool_path.read_text())
        scenes = [SceneSpec.from_json(json.dumps(s)) for s in data["scenes"]]
    else:
        scenes = []
    if not scenes:
        from robot_routes.contracts import Q_HOME

        scenes = [
            SceneSpec((), (0.5, 0.0, 0.4), tuple(float(x) for x in Q_HOME), 3, i)
            for i in range(min(cfg.pool_scenes, 16))
        ]
    policy = policy.to(device)  # type: ignore[assignment]
    anchor = anchor.to(device)  # type: ignore[assignment]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / "ppo.pt"
    if artifact_complete(final) and not args.force_restart:
        print(f"ppo: {final} exists — skipping")
        return
    train_ppo(
        policy,
        anchor,
        cfg,
        scenes,
        out,
        device,
        total_steps=args.steps,
        force_restart=args.force_restart,
        oom_tag_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(f"PPO done → {out / 'ppo.pt'}")


if __name__ == "__main__":
    main()
