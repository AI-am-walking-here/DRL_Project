#!/usr/bin/env python3
"""Seed-0 PPO apex probe: fine-tune from the BC base under the new reward, evaluating
at 12cm between short training chunks to locate the peak before drift degrades it.

Throwaway diagnostic (not part of the pipeline). Logs one row per eval to stdout and
to <out>/apex_log.jsonl, and checkpoints the policy at every eval so the apex is
recoverable even after the policy collapses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from robot_routes.agents.policy import load_checkpoint
from robot_routes.agents.ppo_diversity import PPOTrainer
from robot_routes.contracts import SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.utils.config import PolicyConfig, PPOConfig, load_config, load_yaml
from robot_routes.utils.device import resolve_device
from robot_routes.utils.seeding import seed_everything


def load_scenes(path: str, n: int | None = None) -> list[SceneSpec]:
    data = json.loads(Path(path).read_text())
    scenes = [SceneSpec.from_json(json.dumps(s)) for s in data["scenes"]]
    return scenes[:n] if n else scenes


def eval_set(policy, env: PandaReachEnv, scenes: list[SceneSpec]) -> dict[str, float]:
    """Deterministic rollout; success/collision use the env's live 12cm criterion."""
    succ = coll = 0
    finals: list[float] = []
    for s in scenes:
        obs, _ = env.reset(options={"scene": s})
        a_prev = None
        reached = hit = False
        last = 0.0
        for _ in range(env.cfg.horizon):
            a = policy.act(obs, stochastic=False, a_prev=a_prev)
            a_prev = a
            obs, _, term, trunc, info = env.step(a)
            last = float(np.linalg.norm(info["ee_pos"] - env._goal))
            if info.get("success"):
                reached = True
            if info.get("collision"):
                hit = True
            if term or trunc:
                break
        succ += reached
        coll += hit
        finals.append(last)
    n = len(scenes)
    return {
        "succ": succ / n,
        "collide": coll / n,
        "mean_final_dist": float(np.mean(finals)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="runs/grid/full_seed0/bc/best.pt")
    p.add_argument("--config", default="configs/train/rl_diversity.yaml")
    p.add_argument("--out", default="runs/ppo_test_seed0")
    p.add_argument("--pool", default="data/pool_rl.json")
    p.add_argument("--beta", type=float, default=0.75)
    p.add_argument("--total-steps", type=int, default=240_000)
    p.add_argument("--eval-every", type=int, default=15_000)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-lr", type=float, default=None)
    p.add_argument("--value-warmup-steps", type=int, default=None)
    p.add_argument("--kl-stop", type=float, default=None)
    p.add_argument("--div-ungate-min-quality", type=float, default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    seed_everything(args.seed)
    device = resolve_device("auto")

    cfg = load_config(root / args.config, PPOConfig)
    cfg.beta = args.beta  # gentler diversity pressure for the probe
    if args.policy_lr is not None:
        cfg.policy_lr = args.policy_lr
    if args.value_warmup_steps is not None:
        cfg.value_warmup_steps = args.value_warmup_steps
    if args.kl_stop is not None:
        cfg.kl_stop = args.kl_stop
    if args.div_ungate_min_quality is not None:
        cfg.div_ungate_min_quality = args.div_ungate_min_quality
    print(
        f"stabilized PPO: policy_lr={cfg.policy_lr} value_lr={cfg.value_lr} "
        f"warmup={cfg.value_warmup_steps} kl_stop={cfg.kl_stop} beta={cfg.beta} "
        f"ungate_min_q={cfg.div_ungate_min_quality}",
        flush=True,
    )
    policy_cfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))

    policy = load_checkpoint(root / args.base, policy_cfg).to(device)
    anchor = load_checkpoint(root / args.base, policy_cfg).to(device)
    pool = load_scenes(str(root / args.pool))
    easy = load_scenes(str(root / "data/scenes/val_L0.json"), args.n_eval)
    hard = load_scenes(str(root / "data/scenes/val_unseen.json"), args.n_eval)

    trainer = PPOTrainer(policy, anchor, cfg, pool, device)
    n_envs = max(1, cfg.n_envs)
    train_envs = [PandaReachEnv() for _ in range(n_envs)]
    eval_env = PandaReachEnv()

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "apex_log.jsonl"
    log_path.write_text("")

    def run_eval(step: int, kl: float) -> dict:
        policy.eval()
        e = eval_set(policy, eval_env, easy)
        h = eval_set(policy, eval_env, hard)
        policy.train()
        row = {
            "step": step,
            "easy_succ": e["succ"],
            "hard_succ": h["succ"],
            "easy_collide": e["collide"],
            "hard_collide": h["collide"],
            "easy_dist": e["mean_final_dist"],
            "hard_dist": h["mean_final_dist"],
            "kl_to_base": kl,
            "beta": float(trainer.beta),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"state_dict": policy.state_dict()}, out / f"ckpt_step{step}.pt")
        phase = "WARMUP" if step <= cfg.value_warmup_steps else "POLICY"
        print(
            f"APEX[{phase}] step={step:>7} | easy={row['easy_succ']*100:3.0f}% "
            f"hard={row['hard_succ']*100:3.0f}% | collide e/h="
            f"{row['easy_collide']*100:3.0f}%/{row['hard_collide']*100:3.0f}% | "
            f"dist e/h={row['easy_dist']:.3f}/{row['hard_dist']:.3f} | "
            f"kl={kl:.4f} beta={trainer.beta:.2f}",
            flush=True,
        )
        return row

    run_eval(0, 0.0)  # baseline (= BC under the new 12cm metric)

    steps = 0
    next_eval = args.eval_every
    kl_acc: list[float] = []
    while steps < args.total_steps:
        buffers = []
        for i in range(n_envs):
            scene_idx = (steps // train_envs[0].cfg.horizon + i) % len(pool)
            buf = trainer.rollout_env(train_envs[i % len(train_envs)], scene_idx)
            buffers.append(buf)
            steps += len(buf.obs)
        stats = trainer.train_step(buffers)
        kl_acc.append(stats["kl"])
        if steps >= next_eval:
            run_eval(steps, float(np.mean(kl_acc)) if kl_acc else 0.0)
            kl_acc = []
            next_eval += args.eval_every

    print("APEX DONE", flush=True)


if __name__ == "__main__":
    main()
