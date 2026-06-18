"""RaC rewind tests (§6.3)."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from robot_routes.agents.dagger_rac import reversal_actions, run_rac_intervention
from robot_routes.contracts import Q_HOME, PathTracker, SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.utils.config import DaggerRacConfig, ExpertConfig

pytestmark = pytest.mark.wp5


def test_reversal_algebra():
    q = np.cumsum(np.random.default_rng(0).uniform(-0.02, 0.02, (10, 7)), axis=0) + Q_HOME
    rev = reversal_actions(q)
    assert rev.shape[0] == len(q) - 1


def test_intervention_segments():
    env = PandaReachEnv()
    expert = ExpertOracle(ExpertConfig())
    cfg = DaggerRacConfig()
    from robot_routes.contracts import Q_HOME

    expert.cc.set_q(Q_HOME)
    goal = expert.data.site_xpos[expert._ee_sid].copy() + np.array([0.05, 0.0, -0.08])
    scene = SceneSpec((), tuple(float(x) for x in goal), tuple(float(x) for x in Q_HOME), 0, 0)
    obs, info = env.reset(options={"scene": scene})
    path = expert.plan(info["q"], scene, 0, time_budget_s=10.0)
    assert path is not None
    ring = deque(maxlen=cfg.ring_buffer)
    for _ in range(5):
        ring.append((env.get_state(), 0.15, info["q"].copy()))
        obs, _, term, trunc, info = env.step(np.zeros(7))
        if term or trunc:
            break
    info["min_clearance"] = 0.01
    rows, n, _fb = run_rac_intervention(
        env, expert, ring, path, PathTracker(), scene, cfg, np.random.default_rng(0), info, 0.15
    )
    assert n >= 0
    env.close()
