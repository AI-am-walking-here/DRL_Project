"""Lightweight PPO smoke test."""

from __future__ import annotations

import pytest
import torch

from robot_routes.agents.policy import MDNPolicy
from robot_routes.agents.ppo_diversity import compute_gae, policy_log_prob

pytestmark = pytest.mark.wp8


def test_gae():
    adv, ret = compute_gae([1.0, 1.0], [0.5, 0.5], [False, True], 0.99, 0.95)
    assert len(adv) == 2
    assert len(ret) == 2


def test_policy_log_prob():
    p = MDNPolicy()
    obs = torch.randn(4, 79)
    act = torch.randn(4, 7) * 0.01
    lp = policy_log_prob(p, obs, act)
    assert lp.shape == (4,)
