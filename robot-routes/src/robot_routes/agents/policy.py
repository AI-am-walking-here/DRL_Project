"""Policy networks: MDN, Gaussian ablation, IMLE fallback (§5.2)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from robot_routes.utils.config import PolicyConfig


def mdn_nll(
    logits: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    log_std = log_std.clamp(-4.0, 0.0)
    z = (a[:, None, :] - mu) / log_std.exp()
    log_comp = -0.5 * (z**2).sum(-1) - log_std.sum(-1) - 3.5 * math.log(2 * math.pi)
    return -(torch.logsumexp(torch.log_softmax(logits, -1) + log_comp, -1)).mean()


def mdn_sample(
    logits: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
    a_prev: torch.Tensor | None = None,
    sigma_c: float = 0.02,
    clip: float = 0.05,
) -> torch.Tensor:
    log_w = torch.log_softmax(logits, -1)
    if a_prev is not None:
        log_w = log_w - ((mu - a_prev) ** 2).sum(-1) / (2 * sigma_c**2)
    k = Categorical(logits=log_w).sample()
    a = mu[k] + log_std[k].clamp(-4.0, 0.0).exp() * torch.randn(7, device=mu.device)
    return a.clamp(-clip, clip)


class MDNPolicy(nn.Module):
    def __init__(
        self, cfg: PolicyConfig | None = None, obs_dim: int = 79, act_dim: int = 7
    ) -> None:
        super().__init__()
        cfg = cfg or PolicyConfig(trunk=[512, 512, 256])
        hidden = cfg.trunk or [512, 512, 256]
        self.k = cfg.k_mix
        self.act_dim = act_dim
        self.sigma_c = cfg.sigma_c
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, self.k * (1 + 2 * act_dim))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = (obs - self.obs_mean) / self.obs_std
        out = self.head(self.trunk(x))
        logits = out[..., : self.k]
        rest = out[..., self.k :].reshape(*out.shape[:-1], self.k, 2, self.act_dim)
        mu, log_std = rest.unbind(-2)
        return logits, mu, log_std

    def nll(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor:
        logits, mu, log_std = self(obs_b)
        return mdn_nll(logits, mu, log_std, act_b)

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        *,
        stochastic: bool,
        a_prev: np.ndarray | None = None,
        use_kernel: bool = True,
    ) -> np.ndarray:
        obs = torch.as_tensor(
            obs_np, dtype=torch.float32, device=next(self.parameters()).device
        ).unsqueeze(0)
        logits, mu, log_std = self(obs)
        if stochastic:
            ap = None
            if use_kernel and a_prev is not None:
                ap = torch.as_tensor(a_prev, dtype=torch.float32, device=obs.device)
            a = mdn_sample(logits[0], mu[0], log_std[0], a_prev=ap, sigma_c=self.sigma_c)
        else:
            a = mu[0, int(torch.argmax(logits[0]))].clamp(-0.05, 0.05)
        return a.cpu().numpy()


class GaussianPolicy(nn.Module):
    def __init__(
        self, obs_dim: int = 79, act_dim: int = 7, hidden: list[int] | None = None
    ) -> None:
        super().__init__()
        hidden = hidden or [512, 512, 256]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(d, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def nll(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor:
        x = (obs_b - self.obs_mean) / self.obs_std
        mu = self.mean(self.trunk(x))
        std = self.log_std.clamp(-4, 0).exp()
        z = (act_b - mu) / std
        return (0.5 * z**2 + self.log_std).sum(-1).mean()

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        *,
        stochastic: bool,
        a_prev: np.ndarray | None = None,
        use_kernel: bool = True,
    ) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mu = self.mean(self.trunk((obs - self.obs_mean) / self.obs_std))
        if stochastic:
            std = self.log_std.clamp(-4, 0).exp()
            a = mu + std * torch.randn_like(mu)
        else:
            a = mu
        return a.squeeze(0).clamp(-0.05, 0.05).cpu().numpy()


class IMLEPolicy(nn.Module):
    """Implicit mode — stores K candidate actions per forward pass."""

    def __init__(
        self, obs_dim: int = 79, act_dim: int = 7, k: int = 5, hidden: list[int] | None = None
    ) -> None:
        super().__init__()
        hidden = hidden or [512, 512, 256]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, k * act_dim)
        self.k = k
        self.act_dim = act_dim
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def forward_candidates(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) / self.obs_std
        return self.head(self.trunk(x)).reshape(-1, self.k, self.act_dim)

    def imle_loss(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor:
        cands = self.forward_candidates(obs_b)
        d = ((cands - act_b[:, None, :]) ** 2).sum(-1)
        return d.min(dim=1).values.mean()

    def nll(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor:
        return self.imle_loss(obs_b, act_b)

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        *,
        stochastic: bool,
        a_prev: np.ndarray | None = None,
        use_kernel: bool = True,
    ) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        cands = self.forward_candidates(obs)[0]
        if stochastic:
            idx = int(torch.randint(0, self.k, (1,)).item())
        else:
            idx = 0
        return cands[idx].clamp(-0.05, 0.05).cpu().numpy()


def build_policy(cfg: PolicyConfig, obs_dim: int = 79) -> nn.Module:
    head = cfg.head
    if head == "gaussian":
        return GaussianPolicy(obs_dim, hidden=cfg.trunk)
    if head == "imle":
        return IMLEPolicy(obs_dim, k=cfg.k_mix, hidden=cfg.trunk)
    return MDNPolicy(cfg, obs_dim)


def save_checkpoint(path: str, policy: nn.Module, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "state_dict": policy.state_dict(),
        "class": policy.__class__.__name__,
    }
    if isinstance(policy, MDNPolicy):
        payload["head"] = "mdn"
    elif isinstance(policy, IMLEPolicy):
        payload["head"] = "imle"
    elif isinstance(policy, GaussianPolicy):
        payload["head"] = "gaussian"
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, cfg: PolicyConfig, obs_dim: int = 79) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = payload.get("head", cfg.head)
    policy_cfg = dataclasses.replace(cfg, head=head)
    policy = build_policy(policy_cfg, obs_dim)
    policy.load_state_dict(payload["state_dict"])
    return policy
