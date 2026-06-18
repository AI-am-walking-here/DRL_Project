"""BC trainer (§5.3, §15.7.6)."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from robot_routes.agents.policy import MDNPolicy, build_policy, save_checkpoint
from robot_routes.data.buffer import TransitionDataset
from robot_routes.utils.config import BCConfig, PolicyConfig
from robot_routes.utils.gpu_oom import (
    DEFAULT_MIN_BATCH,
    clear_cuda_cache,
    is_cuda_oom,
    write_oom_backoff,
)


def eval_nll(policy: nn.Module, obs: np.ndarray, act: np.ndarray, device: torch.device) -> float:
    policy.eval()
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
        bs = 4096
        losses = []
        for i in range(0, len(obs), bs):
            losses.append(float(policy.nll(obs_t[i : i + bs], act_t[i : i + bs]).item()))
    return float(np.mean(losses))


def train_bc(
    h5_path: Path,
    cfg: BCConfig,
    policy_cfg: PolicyConfig,
    out_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
    *,
    oom_tag_dir: Path | None = None,
    min_batch: int = DEFAULT_MIN_BATCH,
) -> nn.Module:
    batch = cfg.batch
    last_err: BaseException | None = None
    for attempt in range(4):
        try:
            return _train_bc_once(
                h5_path, cfg, policy_cfg, out_dir, device, rng, batch_size=batch
            )
        except Exception as e:
            if not is_cuda_oom(e) or batch <= min_batch:
                if is_cuda_oom(e) and oom_tag_dir is not None:
                    write_oom_backoff(oom_tag_dir, stage="train_bc", detail=str(e))
                raise
            last_err = e
            clear_cuda_cache()
            batch = max(min_batch, batch // 2)
            print(f"train_bc: CUDA OOM — retrying with batch={batch} (attempt {attempt + 2})")
    if last_err is not None:
        raise last_err
    raise RuntimeError("train_bc: unreachable")


def _train_bc_once(
    h5_path: Path,
    cfg: BCConfig,
    policy_cfg: PolicyConfig,
    out_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
    *,
    batch_size: int,
) -> nn.Module:
    cfg = replace(cfg, batch=batch_size)
    ds = TransitionDataset(h5_path, val_frac=cfg.val_frac, rng=rng)
    policy = build_policy(policy_cfg).to(device)
    policy.obs_mean.copy_(torch.as_tensor(ds.obs[ds.train_mask].mean(0)))
    policy.obs_std.copy_(torch.as_tensor(ds.obs[ds.train_mask].std(0) + 1e-6))
    weights_cfg = {
        "full_demo": 1.0,
        "dagger_label": 1.0,
        "clean_rollout": 0.5,
        "recovery": 1.5,
        "correction": 1.5,
    }
    w = ds.segment_weights(weights_cfg)
    train_idx = np.where(ds.train_mask)[0]
    sampler = WeightedRandomSampler(w[train_idx].tolist(), num_samples=len(train_idx))
    loader = DataLoader(
        ds,
        batch_size=cfg.batch,
        sampler=sampler,
        drop_last=True,
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    val_obs, val_act = ds.val_loader_data()
    best_val = float("inf")
    best_state: dict | None = None
    for epoch in range(cfg.epochs):
        policy.train()
        for obs_b, act_b in loader:
            loss = policy.nll(obs_b.to(device), act_b.to(device))
            if epoch < policy_cfg.entropy_bonus_epochs and isinstance(policy, MDNPolicy):
                logits, _, _ = policy(obs_b.to(device))
                p = torch.log_softmax(logits, -1)
                loss = loss - policy_cfg.entropy_bonus * (-(p.exp() * p).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        v = eval_nll(policy, val_obs, val_act, device)
        if v < best_val:
            best_val = v
            best_state = copy.deepcopy(policy.state_dict())
    if best_state is not None:
        policy.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out_dir / "best.pt"), policy)
    return policy
