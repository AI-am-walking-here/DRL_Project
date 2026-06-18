#!/usr/bin/env python3
"""Smoke test: reset → 20 steps → state roundtrip → clearance (§WP1)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    seed_everything(0)
    t0 = time.time()
    render_mode = "rgb_array" if args.render else None
    env = PandaReachEnv(render_mode=render_mode)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (79,)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs, _, term, trunc, info = env.step(rng.uniform(-0.05, 0.05, 7))
        assert "min_clearance" in info
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(2**31)))
    s = env.get_state()
    obs2 = env.set_state(s)
    obs3, _, _, _, _ = env.step(np.zeros(7))
    c = env.min_clearance()
    assert np.isfinite(c)
    if args.render:
        env.render()
    env.close()
    elapsed = time.time() - t0
    print(f"smoke ok in {elapsed:.1f}s, clearance={c:.3f}")
    assert elapsed < 60.0, f"smoke too slow: {elapsed}s"


if __name__ == "__main__":
    main()
