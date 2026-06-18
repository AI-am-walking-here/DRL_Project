"""Route diversity metric tests (§9)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_routes.diversity.route_metrics import (
    count_routes,
    frechet,
    paired_bootstrap_ci,
)

pytestmark = pytest.mark.wp7


def test_frechet_identical():
    p = np.linspace(0, 1, 64)[:, None] * np.array([1, 0, 0])
    assert frechet(p, p) == pytest.approx(0.0, abs=1e-9)


def test_frechet_parallel_offset():
    p = np.zeros((64, 3))
    q = np.zeros((64, 3))
    q[:, 0] = 0.2
    assert frechet(p, q) == pytest.approx(0.2, abs=1e-6)


def test_cluster_two_routes():
    left = np.zeros((64, 3))
    left[:, 1] = np.linspace(0, 0.3, 64)
    right = np.zeros((64, 3))
    right[:, 1] = np.linspace(0, -0.3, 64)
    trajs = [left, right] + [
        left + np.random.default_rng(i).normal(0, 0.01, (64, 3)) for i in range(8)
    ]
    n = count_routes(trajs, delta=0.15)
    assert n == 2


def test_bootstrap_planted_effect():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 0.7, 50)
    b = a - 0.15
    lo, hi = paired_bootstrap_ci(a, b, rng, n=1000)
    assert lo > 0
