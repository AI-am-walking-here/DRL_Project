#!/usr/bin/env python3
"""Render MuJoCo camera rollouts for report videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.agents.policy import load_checkpoint
from robot_routes.data.scene_sets import load_scenes
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.pipeline.mujoco_video import (
    export_rollout,
    rollout_expert_frames,
    rollout_policy_frames,
)
from robot_routes.utils.config import ExpertConfig, PolicyConfig, load_config, load_yaml


def main() -> int:
    p = argparse.ArgumentParser(description="MuJoCo 3D showcase videos")
    p.add_argument("--out", default="runs/showcase_videos/mujoco")
    p.add_argument("--run", default="runs/grid/bc_seed0")
    p.add_argument("--ckpt", default=None, help="policy checkpoint (default: <run>/bc/best.pt)")
    p.add_argument("--scene-set", default="val_L0")
    p.add_argument("--scene-idx", type=int, default=3)
    p.add_argument("--l3-idx", type=int, default=0, help="also render val_L3 expert clutter example")
    p.add_argument("--fps", type=int, default=24)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_yaml(root / "configs/train/bc.yaml")
    policy_cfg = PolicyConfig(**raw.get("policy", {}))
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))

    scenes = load_scenes(root, args.scene_set)
    scene = scenes[args.scene_idx]
    ckpt = Path(args.ckpt) if args.ckpt else Path(args.run) / "bc" / "best.pt"

    manifest: dict[str, object] = {"scene_set": args.scene_set, "scene_idx": args.scene_idx, "exports": []}

    # Expert on val_L0
    frames, ok = rollout_expert_frames(scene, expert)
    paths = export_rollout(frames, out / f"01_expert_{args.scene_set}_scene{args.scene_idx}", fps=args.fps)
    manifest["exports"].append({"label": "expert_l0", "success": ok, **paths})
    print(f"expert {args.scene_set}[{args.scene_idx}] success={ok} -> {paths}")

    # Policy on same scene
    if ckpt.exists():
        policy = load_checkpoint(str(ckpt), policy_cfg)
        frames, ok = rollout_policy_frames(scene, policy)
        paths = export_rollout(
            frames, out / f"02_policy_{Path(args.run).name}_scene{args.scene_idx}", fps=args.fps
        )
        manifest["exports"].append({"label": "policy_same_scene", "success": ok, **paths})
        print(f"policy same scene success={ok} -> {paths}")

        val_path = Path(args.run) / "eval" / "val_eval.json"
        if val_path.exists():
            val = json.loads(val_path.read_text())
            for idx, sr in enumerate(val.get("scene_success", [])):
                if float(sr) < 1.0:
                    continue
                s = scenes[idx]
                frames, ok = rollout_policy_frames(s, policy)
                paths = export_rollout(
                    frames, out / f"03_policy_success_scene{idx}", fps=args.fps
                )
                manifest["exports"].append({"label": f"policy_success_{idx}", "success": ok, **paths})
                print(f"policy success scene {idx} -> {paths}")
                break

    # Expert on cluttered L3
    if (root / "data" / "scenes" / "val_L3.json").exists():
        l3 = load_scenes(root, "val_L3")
        s3 = l3[args.l3_idx]
        frames, ok = rollout_expert_frames(s3, expert)
        paths = export_rollout(
            frames,
            out / f"04_expert_val_L3_scene{args.l3_idx}_{len(s3.obstacles)}obs",
            fps=args.fps,
        )
        manifest["exports"].append(
            {"label": "expert_l3", "success": ok, "obstacles": len(s3.obstacles), **paths}
        )
        print(f"expert val_L3[{args.l3_idx}] ({len(s3.obstacles)} obs) success={ok} -> {paths}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
