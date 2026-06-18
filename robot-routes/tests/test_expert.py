"""Expert / RRT tests (§4)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_routes.contracts import Q_HOME, SceneSpec
from robot_routes.expert.oracle import ExpertOracle, goal_roots
from robot_routes.utils.config import ExpertConfig

pytestmark = pytest.mark.wp3


@pytest.fixture
def expert():
    return ExpertOracle(ExpertConfig(max_iters=5000))


def test_margin_gt_danger():
    cfg = ExpertConfig()
    from robot_routes.utils.config import DaggerRacConfig

    d = DaggerRacConfig()
    assert cfg.margin_plan_m > d.eps_danger_m


def test_collision_free_config(expert):
    cc = expert.cc
    q = Q_HOME.copy()
    assert cc.free(q)


def _reachable_goal(expert: ExpertOracle, delta: np.ndarray | None = None) -> np.ndarray:
    from robot_routes.contracts import Q_HOME

    expert.cc.set_q(Q_HOME)
    ee = expert.data.site_xpos[expert._ee_sid].copy()
    return ee + (delta if delta is not None else np.array([0.05, 0.0, -0.08]))


def test_plan_simple_scene(expert):
    goal = _reachable_goal(expert)
    scene = SceneSpec(
        (),
        tuple(float(x) for x in goal),
        tuple(float(x) for x in Q_HOME),
        0,
        7,
    )
    path = expert.plan(Q_HOME, scene, 0, time_budget_s=10.0)
    assert path is not None
    assert len(path.waypoints) >= 2
    assert len(path.waypoints) >= 2


def test_label_terminal_hold(expert):
    from robot_routes.contracts import JointPath, PathTracker

    wp = np.tile(Q_HOME, (5, 1))
    path = JointPath(wp)
    tracker = PathTracker(idx=4)
    a = expert.label(Q_HOME, path, tracker)
    np.testing.assert_array_equal(a, np.zeros(7))


def test_ik_elbow_families(expert):
    goal = _reachable_goal(expert)
    roots = goal_roots(
        expert.model,
        expert.data,
        goal,
        expert._ee_sid,
        expert._jlo,
        expert._jhi,
        np.random.default_rng(0),
        expert.cc,
    )
    assert len(roots) >= 1
