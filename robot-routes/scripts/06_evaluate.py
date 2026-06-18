#!/usr/bin/env python3
"""Evaluation runner (§10)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.eval.evaluate import evaluate_checkpoint, load_scenes
from robot_routes.pipeline.calibration import load_delta
from robot_routes.utils.config import (
    DiversityConfig,
    EvalConfig,
    PolicyConfig,
    load_config,
    load_yaml,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval/default.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--ckpt", default="runs/curriculum/best.pt")
    p.add_argument("--scenes", default="data/scenes/val_L0.json")
    p.add_argument("--test", action="store_true")
    p.add_argument("--routes", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = load_yaml(root / args.config)
    eval_cfg = load_config(root / args.config, EvalConfig)
    div_cfg = DiversityConfig(**raw.get("diversity", {}))
    policy_cfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))
    scenes_path = root / args.scenes if not Path(args.scenes).is_absolute() else Path(args.scenes)
    scenes = load_scenes(scenes_path) if scenes_path.exists() else []
    delta = load_delta(root)
    results = evaluate_checkpoint(
        Path(args.ckpt),
        scenes,
        policy_cfg,
        eval_cfg,
        div_cfg,
        delta=delta,
        include_routes=args.routes,
        root=root,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / ("test_eval.json" if args.test else "val_eval.json")
    out_file.write_text(json.dumps(results, indent=2))
    if args.test:
        (out / "test_touched.json").write_text(
            json.dumps(
                {"seed": args.seed, "ts": __import__("datetime").datetime.utcnow().isoformat()}
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
