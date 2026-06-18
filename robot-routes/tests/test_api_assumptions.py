"""Tier 0 API assumption tests (§15.4)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch


def test_mujoco_surface():
    import mujoco

    for fn in (
        "mj_step",
        "mj_forward",
        "mj_geomDistance",
        "mj_stateSize",
        "mj_getState",
        "mj_setState",
        "mj_name2id",
        "mj_jacSite",
    ):
        assert hasattr(mujoco, fn), fn
    assert hasattr(mujoco.mjtState, "mjSTATE_INTEGRATION")
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    assert n >= 1


def test_gymnasium_is_five_tuple():
    class E(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-1, 1, (1,))
            self.action_space = gym.spaces.Box(-1, 1, (1,))

        def reset(self, *, seed=None, options=None):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0.0, False, False, {}

    env = E()
    out = env.step(np.zeros(1))
    assert len(out) == 5


def test_torch_surface():
    x = torch.randn(2, 3)
    torch.logsumexp(x, -1)
    torch.distributions.Categorical(logits=x)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)


def test_pynvml_optional():
    try:
        import pynvml

        assert hasattr(pynvml, "nvmlInit")
    except ImportError:
        pytest.skip("pynvml not installed")
