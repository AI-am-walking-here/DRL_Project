#!/usr/bin/env python3
"""Is route diversity already latent in the BC-trained MDN?

Samples the policy's mixture stochastically (no PPO) and counts how many distinct
*valid* routes emerge per scene, using the project's own Frechet clustering metric
(count_routes). If the modes already yield >1 route on solvable scenes, we get the
"many routes" story without unstable RL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from robot_routes.agents.policy import load_checkpoint
from robot_routes.contracts import SceneSpec
from robot_routes.diversity.route_metrics import count_routes, resample_path
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.eval.evaluate import rollout_episode
from robot_routes.utils.config import DiversityConfig, PolicyConfig, load_yaml


def load_scenes(path: str, n: int) -> list[SceneSpec]:
    data = json.loads(Path(path).read_text())
    return [SceneSpec.from_json(json.dumps(s)) for s in data["scenes"]][:n]


def probe_set(policy, env, scenes, div, deltas):
    rows = []
    for sc in scenes:
        trajs = []
        for s in range(div.n_rollouts):
            r = rollout_episode(env, policy, sc, s, stochastic=True)
            if r["success"]:
                trajs.append(resample_path(r["ee_traj"], div.resample_pts))
        succ = len(trajs)
        valid = succ >= div.validity_min
        routes = {d: count_routes(trajs, d) for d in deltas} if valid else None
        rows.append({"successes": succ, "valid": valid, "routes": routes})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/grid/full_seed0/bc/best.pt")
    p.add_argument("--n-scenes", type=int, default=15)
    p.add_argument("--out", default="runs/mdn_diversity_probe.json")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]

    pcfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))
    policy = load_checkpoint(root / args.ckpt, pcfg)
    env = PandaReachEnv()
    div = DiversityConfig()
    deltas = [0.10, 0.15, 0.20]

    report = {"ckpt": args.ckpt, "n_rollouts": div.n_rollouts, "validity_min": div.validity_min}
    for name, fn in [("easy(val_L0)", "data/scenes/val_L0.json"),
                     ("hard(val_unseen)", "data/scenes/val_unseen.json")]:
        scenes = load_scenes(str(root / fn), args.n_scenes)
        rows = probe_set(policy, env, scenes, div, deltas)
        valid_rows = [r for r in rows if r["valid"]]
        agg = {
            "mean_successes": float(np.mean([r["successes"] for r in rows])),
            "validity_frac": len(valid_rows) / len(rows),
            "mean_n_routes": {
                str(d): (float(np.mean([r["routes"][d] for r in valid_rows])) if valid_rows else None)
                for d in deltas
            },
        }
        report[name] = agg
        print(f"\n=== {name} (n={len(rows)}, {div.n_rollouts} stochastic rollouts each) ===")
        print(f"  mean successes/{div.n_rollouts}: {agg['mean_successes']:.1f}")
        print(f"  scenes with >={div.validity_min} successes (route-valid): "
              f"{agg['validity_frac']*100:.0f}%")
        for d in deltas:
            mr = agg["mean_n_routes"][str(d)]
            print(f"  mean distinct routes @ delta={d}: {mr if mr is None else round(mr, 2)}")

    (root / args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
