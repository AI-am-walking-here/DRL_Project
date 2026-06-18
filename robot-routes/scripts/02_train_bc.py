#!/usr/bin/env python3
"""Stage 1: BC training (§5.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.agents.bc_trainer import train_bc
from robot_routes.pipeline.stage_resume import artifact_complete
from robot_routes.utils.config import BCConfig, PolicyConfig, load_config, load_yaml
from robot_routes.utils.device import resolve_device
from robot_routes.utils.seeding import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/bc.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/bc_train")
    p.add_argument("--data", default="runs/bc_collect/demos.h5")
    p.add_argument("--device", default="auto")
    p.add_argument("--force-restart", action="store_true")
    p.add_argument("--run-dir", default="", help="Pipeline run dir for OOM backoff tagging")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = load_yaml(root / args.config)
    cfg = load_config(root / args.config, BCConfig)
    policy_cfg = PolicyConfig(**raw.get("policy", {}))
    rng = seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.out)
    ckpt = out / "best.pt"
    if artifact_complete(ckpt) and not args.force_restart:
        print(f"train_bc: {ckpt} exists — skipping")
        return
    train_bc(
        Path(args.data),
        cfg,
        policy_cfg,
        out,
        device,
        rng,
        oom_tag_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(f"BC training done → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
