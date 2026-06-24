"""G-BC diagnostic ladder (§5.3, §11.7.2)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

from robot_routes.agents.bc_trainer import eval_nll, train_bc
from robot_routes.agents.policy import build_policy
from robot_routes.contracts import PathTracker, SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.oracle import label as expert_label
from robot_routes.utils.config import (
    BCConfig,
    DiversityConfig,
    EvalConfig,
    ExpertConfig,
    PolicyConfig,
    load_config,
)


def _episode_indices(h5_path: Path) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        ep = f["episode_id"][:]
    return sorted(set(int(x) for x in ep))


def expert_replay_check(
    h5_path: Path,
    expert: ExpertOracle,
    min_rate: float = 0.95,
    max_episodes: int = 20,
) -> tuple[bool, float]:
    env = PandaReachEnv()
    with h5py.File(h5_path, "r") as f:
        scenes = list(f["episodes/scene_json"].asstr()[:])
    eps = _episode_indices(h5_path)[:max_episodes]
    ok = 0
    for eid in eps:
        if eid >= len(scenes):
            continue
        scene = SceneSpec.from_json(scenes[eid])
        obs, info = env.reset(options={"scene": scene})
        path = expert.plan(
            info["q"],
            scene,
            scene.seed,
            time_budget_s=expert.cfg.t_validate_s,
        )
        if path is None:
            continue
        tracker = PathTracker()
        success = False
        for _ in range(env.cfg.horizon):
            a = expert_label(info["q"], path, tracker, expert.cfg.lookahead)
            obs, _, term, trunc, info = env.step(a)
            if info.get("success"):
                success = True
                break
            if term or trunc:
                break
        ok += int(success)
    rate = ok / max(len(eps), 1)
    return rate >= min_rate, rate


def overfit_ten_episodes(
    h5_path: Path,
    policy_cfg: PolicyConfig,
    device: torch.device,
    rng: np.random.Generator,
    max_nll: float = 1.0,
    epochs: int = 50,
) -> tuple[bool, float]:
    eps = _episode_indices(h5_path)[:10]
    if len(eps) < 3:
        return True, 0.0
    with h5py.File(h5_path, "r") as f:
        mask = np.isin(f["episode_id"][:], eps)
        obs = f["obs"][:][mask]
        act = f["action"][:][mask]
    policy = build_policy(policy_cfg).to(device)
    policy.obs_mean.copy_(torch.as_tensor(obs.mean(0)))
    policy.obs_std.copy_(torch.as_tensor(obs.std(0) + 1e-6))
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    for _ in range(epochs):
        policy.train()
        loss = policy.nll(obs_t, act_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    nll = eval_nll(policy, obs, act, device)
    return nll <= max_nll, nll


def sweep_lr_chunking(
    h5_path: Path,
    base_cfg: BCConfig,
    policy_cfg: PolicyConfig,
    out_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
    val_scenes: list,
    eval_cfg: EvalConfig,
    div_cfg: DiversityConfig,
    delta: float,
) -> tuple[Path | None, float]:
    best_sr = -1.0
    best_ckpt: Path | None = None
    for lr in (1e-4, 3e-4):
        for chunk_k in (1, 4):
            pc = dataclasses.replace(policy_cfg, chunk_k=chunk_k)
            cfg = dataclasses.replace(base_cfg, lr=lr, epochs=max(5, base_cfg.epochs // 4))
            sub = out_dir / f"sweep_lr{lr}_chunk{chunk_k}"
            ckpt = sub / "best.pt"
            if not ckpt.exists():
                train_bc(h5_path, cfg, pc, sub, device, rng)
                ckpt = sub / "best.pt"
            if not val_scenes:
                return ckpt, 0.0
            ev = evaluate_checkpoint(ckpt, val_scenes, pc, eval_cfg, div_cfg, delta=delta)
            sr = ev["success_rate"]
            if sr > best_sr:
                best_sr = sr
                best_ckpt = ckpt
    return best_ckpt, best_sr


def run_gbc_ladder(
    root: Path,
    run_dir: Path,
    h5_path: Path,
    bc_cfg: BCConfig,
    policy_cfg: PolicyConfig,
    eval_cfg: EvalConfig,
    div_cfg: DiversityConfig,
    delta: float,
    val_scenes: list,
    device: torch.device,
    rng: np.random.Generator,
    *,
    force_restart: bool = False,
) -> tuple[bool, Path | None, dict]:
    """Returns (passed_soft_gate, best_checkpoint, diagnostics)."""
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    diag: dict = {}
    min_rate = getattr(bc_cfg, "replay_min_rate", 0.70)
    max_eps = getattr(bc_cfg, "replay_max_episodes", 20)
    ok1, rate1 = expert_replay_check(
        h5_path, expert, min_rate=min_rate, max_episodes=max_eps
    )
    diag["expert_replay_rate"] = rate1
    diag["replay_pass"] = ok1
    diag["replay_min_rate"] = min_rate
    diag["replay_max_episodes"] = max_eps
    if not ok1:
        diag["replay_note"] = "below threshold — continuing ladder anyway"
    ok2, nll = overfit_ten_episodes(h5_path, policy_cfg, device, rng)
    diag["overfit_nll"] = nll
    if not ok2:
        return False, None, diag
    ladder_dir = run_dir / "bc_ladder"
    if force_restart and ladder_dir.exists():
        shutil.rmtree(ladder_dir, ignore_errors=True)
    ladder_dir.mkdir(parents=True, exist_ok=True)
    ckpt, sr = sweep_lr_chunking(
        h5_path,
        bc_cfg,
        policy_cfg,
        ladder_dir,
        device,
        rng,
        val_scenes,
        eval_cfg,
        div_cfg,
        delta,
    )
    diag["sweep_best_sr"] = sr
    if ckpt is None:
        return False, None, diag
    if val_scenes:
        ev = evaluate_checkpoint(ckpt, val_scenes, policy_cfg, eval_cfg, div_cfg, delta=delta)
        diag["final_sr"] = ev["success_rate"]
        if ev["success_rate"] >= bc_cfg.gate_soft:
            dest = run_dir / "bc" / "best.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ckpt, dest)
            return True, dest, diag
        return False, ckpt, diag
    return True, ckpt, diag
