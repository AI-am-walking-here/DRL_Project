"""PPO with path-diversity reward (§8)."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from robot_routes.agents.policy import MDNPolicy, mdn_sample
from robot_routes.diversity.route_metrics import frechet, resample_path
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.utils.config import PPOConfig
from robot_routes.utils.gpu_oom import clear_cuda_cache, is_cuda_oom, write_oom_backoff


class ValueNet(nn.Module):
    def __init__(
        self, obs_dim: int = 79, hidden: int = 256, n_scenes: int = 256, embed_dim: int = 16
    ) -> None:
        super().__init__()
        self.scene_embed = nn.Embedding(n_scenes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, scene_idx: torch.Tensor) -> torch.Tensor:
        emb = self.scene_embed(scene_idx)
        return self.net(torch.cat([obs, emb], dim=-1)).squeeze(-1)


@dataclass
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    scene_idx: list[int] = field(default_factory=list)


def compute_gae(
    rewards: list[float], values: list[float], dones: list[bool], gamma: float, lam: float
) -> tuple[list[float], list[float]]:
    adv: list[float] = []
    gae = 0.0
    next_val = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv.insert(0, gae)
        next_val = values[t]
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def policy_log_prob(policy: MDNPolicy, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    logits, mu, log_std = policy(obs)
    log_std = log_std.clamp(-4.0, 0.0)
    z = (act[:, None, :] - mu) / log_std.exp()
    log_comp = -0.5 * (z**2).sum(-1) - log_std.sum(-1) - 3.5 * np.log(2 * np.pi)
    return torch.logsumexp(torch.log_softmax(logits, -1) + log_comp, -1)


class PPOTrainer:
    def __init__(
        self,
        policy: MDNPolicy,
        anchor: MDNPolicy,
        cfg: PPOConfig,
        scenes: list[Any],
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.anchor = anchor
        self.cfg = cfg
        self.scenes = scenes
        self.device = device
        self.value = ValueNet(n_scenes=len(scenes)).to(device)
        # Separate LRs: the policy is a pretrained asset (nudge gently); the value
        # head is from scratch (learn fast). Single optimizer, two param groups.
        self.opt = torch.optim.AdamW(
            [
                {"params": list(policy.parameters()), "lr": cfg.policy_lr},
                {"params": list(self.value.parameters()), "lr": cfg.value_lr},
            ]
        )
        self.archives: dict[int, deque] = {
            i: deque(maxlen=cfg.archive_m) for i in range(len(scenes))
        }
        self.beta = cfg.beta
        self.beta_halvings = 0
        self.env_steps = 0  # drives the value-warmup window

    def rollout_env(self, env: PandaReachEnv, scene_idx: int) -> RolloutBuffer:
        buf = RolloutBuffer()
        scene = self.scenes[scene_idx]
        obs, info = env.reset(options={"scene": scene})
        ee_traj: list[np.ndarray] = []
        for _ in range(env.cfg.horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits, mu, log_std = self.policy(obs_t)
                a = mdn_sample(logits[0], mu[0], log_std[0], a_prev=None)
                lp = policy_log_prob(self.policy, obs_t, a.unsqueeze(0)).item()
            act = a.cpu().numpy()
            buf.obs.append(obs)
            buf.actions.append(act)
            buf.log_probs.append(lp)
            buf.scene_idx.append(scene_idx)
            obs, r, term, trunc, info = env.step(act)
            ee_traj.append(info["ee_pos"].copy())
            buf.rewards.append(r)
            buf.dones.append(term or trunc)
            if term or trunc:
                break
        success = bool(info.get("success"))
        if ee_traj:
            tau = resample_path(np.array(ee_traj))
            min_dist = float(np.linalg.norm(np.array(ee_traj) - env._goal, axis=1).min())
            quality = 1.0 if success else max(0.0, 1.0 - min_dist / self.cfg.div_quality_d0)
            eligible = success or (
                self.cfg.div_ungate and quality >= self.cfg.div_ungate_min_quality
            )
            if eligible:
                archive = self.archives[scene_idx]
                if archive:
                    d_min = min(frechet(tau, t) for t in archive)
                    r_div = self.beta * min(max(d_min, 0.0), self.cfg.rdiv_cap)
                else:
                    r_div = self.beta * self.cfg.rdiv_cap
                buf.rewards[-1] += r_div * quality
                if success or quality >= self.cfg.div_ungate_min_quality:
                    archive.append(tau)
        return buf

    def train_step(self, buffers: list[RolloutBuffer]) -> dict[str, float]:
        obs_all, act_all, lp_old, adv_all, ret_all, scene_all = [], [], [], [], [], []
        for buf in buffers:
            obs_t = torch.as_tensor(np.array(buf.obs), dtype=torch.float32, device=self.device)
            scene_t = torch.as_tensor(buf.scene_idx, dtype=torch.long, device=self.device)
            with torch.no_grad():
                vals = self.value(obs_t, scene_t).cpu().numpy().tolist()
            adv, ret = compute_gae(
                buf.rewards, vals, buf.dones, self.cfg.gamma, self.cfg.gae_lambda
            )
            obs_all.extend(buf.obs)
            act_all.extend(buf.actions)
            lp_old.extend(buf.log_probs)
            adv_all.extend(adv)
            ret_all.extend(ret)
            scene_all.extend(buf.scene_idx)
        self.env_steps += len(obs_all)
        obs_t = torch.as_tensor(np.array(obs_all), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.array(act_all), dtype=torch.float32, device=self.device)
        lp_old_t = torch.as_tensor(lp_old, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv_all, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret_all, dtype=torch.float32, device=self.device)
        scene_t = torch.as_tensor(scene_all, dtype=torch.long, device=self.device)
        lp_new = policy_log_prob(self.policy, obs_t, act_t)
        ratio = torch.exp(lp_new - lp_old_t)
        clip = self.cfg.clip
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((self.value(obs_t, scene_t) - ret_t) ** 2).mean()
        with torch.no_grad():
            obs_sub = obs_t[: min(64, len(obs_t))]
            logits, mu, log_std = self.policy(obs_sub)
            sample_a = mdn_sample(logits, mu, log_std, a_prev=None)
        lp_s = policy_log_prob(self.policy, obs_t[: len(sample_a)], sample_a)
        with torch.no_grad():
            lp_a = policy_log_prob(self.anchor, obs_t[: len(sample_a)], sample_a)
        kl = (lp_s - lp_a).mean()
        ent = -lp_new.mean()
        warmup = self.env_steps <= self.cfg.value_warmup_steps
        if warmup:
            # Freeze the policy until the value head gives trustworthy advantages.
            loss = 0.5 * value_loss
        else:
            loss = (
                policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * ent
                + self.cfg.kl_anchor * kl
            )
        self.opt.zero_grad()
        loss.backward()
        # Trust-region guard: skip the policy nudge whenever it would (or already does)
        # sit past the KL budget — the value head still learns, the policy can't bolt.
        if warmup or abs(float(kl.item())) > self.cfg.kl_stop:
            for param in self.policy.parameters():
                param.grad = None
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "kl": float(kl.item()),
            "entropy": float(ent.item()),
            "warmup": float(warmup),
        }

    def maybe_halve_beta(self, r_div_std: float, success_bonus: float = 10.0) -> None:
        if r_div_std > 2 * success_bonus and self.beta_halvings < self.cfg.beta_halvings_max:
            self.beta /= 2
            self.beta_halvings += 1


def train_ppo(
    policy: MDNPolicy,
    anchor: MDNPolicy,
    cfg: PPOConfig,
    scenes: list[Any],
    out_dir: Path,
    device: torch.device,
    total_steps: int | None = None,
    *,
    force_restart: bool = False,
    oom_tag_dir: Path | None = None,
) -> MDNPolicy:
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "ppo_progress.json"
    partial_path = out_dir / "ppo_partial.pt"
    trainer = PPOTrainer(policy, anchor, cfg, scenes, device)
    n_envs = max(1, cfg.n_envs)
    envs = [PandaReachEnv() for _ in range(n_envs)]
    steps = 0
    target = total_steps or cfg.total_steps
    if not force_restart and progress_path.exists() and partial_path.exists():
        try:
            meta = json.loads(progress_path.read_text())
            steps = int(meta.get("steps", 0))
            blob = torch.load(partial_path, map_location=device, weights_only=True)
            policy.load_state_dict(blob["state_dict"])
            if "trainer_beta" in meta:
                trainer.beta = float(meta["trainer_beta"])
            if "n_envs" in meta:
                n_envs = max(1, int(meta["n_envs"]))
                envs = [PandaReachEnv() for _ in range(n_envs)]
            print(f"ppo: resuming from step {steps}/{target}")
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            steps = 0
    while steps < target:
        try:
            buffers = []
            r_divs: list[float] = []
            for i in range(n_envs):
                scene_idx = (steps // envs[0].cfg.horizon + i) % len(scenes)
                buf = trainer.rollout_env(envs[i % len(envs)], scene_idx)
                buffers.append(buf)
                if buf.rewards:
                    r_divs.append(buf.rewards[-1])
                steps += len(buf.obs)
            trainer.train_step(buffers)
        except Exception as e:
            if not is_cuda_oom(e) or n_envs <= 1:
                if is_cuda_oom(e) and oom_tag_dir is not None:
                    write_oom_backoff(oom_tag_dir, stage="ppo", detail=str(e))
                raise
            clear_cuda_cache()
            n_envs = max(1, n_envs // 2)
            envs = [PandaReachEnv() for _ in range(n_envs)]
            progress_path.write_text(
                json.dumps(
                    {"steps": steps, "target": target, "trainer_beta": trainer.beta, "n_envs": n_envs}
                )
            )
            torch.save({"state_dict": policy.state_dict()}, partial_path)
            print(f"ppo: CUDA OOM — resume with n_envs={n_envs} from step {steps}")
            continue
        if r_divs:
            trainer.maybe_halve_beta(float(np.std(r_divs)))
        progress_path.write_text(
            json.dumps(
                {"steps": steps, "target": target, "trainer_beta": trainer.beta, "n_envs": n_envs}
            )
        )
        torch.save({"state_dict": policy.state_dict()}, partial_path)
    torch.save({"state_dict": policy.state_dict()}, out_dir / "ppo.pt")
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return policy
