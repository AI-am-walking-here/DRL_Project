"""Closed-loop expert rollout checks (planner + execution)."""

from __future__ import annotations

import numpy as np

from robot_routes.contracts import PathTracker, SceneSpec
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.oracle import label as expert_label


def expert_solves_scene(
    env: PandaReachEnv,
    expert: ExpertOracle,
    scene: SceneSpec,
    *,
    planner_seed: int | None = None,
    settle_steps: int = 5,
) -> bool:
    """True only if the expert plans and reaches the goal without truncating early."""
    obs, info = env.reset(options={"scene": scene})
    seed = int(scene.seed if planner_seed is None else planner_seed)
    path = expert.plan(
        info["q"],
        scene,
        seed,
        time_budget_s=expert.cfg.t_validate_s,
    )
    if path is None:
        return False
    tracker = PathTracker()
    for _ in range(env.cfg.horizon):
        a = expert_label(info["q"], path, tracker, expert.cfg.lookahead)
        obs, _, term, trunc, info = env.step(a)
        if info.get("success"):
            for _ in range(settle_steps):
                obs, _, term, trunc, info = env.step(np.zeros(7, dtype=np.float64))
            return True
        if term or trunc:
            return False
    return False
