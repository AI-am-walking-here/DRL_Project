"""Batch evaluation (§10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from robot_routes.agents.policy import load_checkpoint
from robot_routes.contracts import SceneSpec
from robot_routes.diversity.route_metrics import count_routes, resample_path, route_entropy
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.utils.config import (
    DiversityConfig,
    EvalConfig,
    ExpertConfig,
    PolicyConfig,
    load_config,
)


def rollout_episode(
    env: PandaReachEnv, policy: Any, scene: SceneSpec, seed: int, stochastic: bool
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed, options={"scene": scene})
    ee_traj = [info["ee_pos"].copy()]
    recoveries = 0
    in_recovery = False
    a_prev = None
    for _ in range(env.cfg.horizon):
        if info["min_clearance"] < 0.02 and not in_recovery:
            in_recovery = True
        if in_recovery and info["min_clearance"] > 0.10:
            recoveries += 1
            in_recovery = False
        a = policy.act(obs, stochastic=stochastic, a_prev=a_prev)
        a_prev = a
        obs, _, term, trunc, info = env.step(a)
        ee_traj.append(info["ee_pos"].copy())
        if term or trunc:
            break
    return {
        "success": bool(info.get("success")),
        "collision": bool(info.get("collision")),
        "timeout": bool(trunc and not info.get("success")),
        "length": len(ee_traj),
        "recoveries": recoveries,
        "ee_traj": np.array(ee_traj),
    }


def evaluate_routes(
    env: PandaReachEnv,
    policy: Any,
    scene: SceneSpec,
    div_cfg: DiversityConfig,
    delta: float,
) -> dict[str, Any]:
    seeds = list(range(div_cfg.n_rollouts))
    trajs: list[np.ndarray] = []
    successes = 0
    for s in seeds:
        r = rollout_episode(env, policy, scene, s, stochastic=True)
        if r["success"]:
            successes += 1
            trajs.append(resample_path(r["ee_traj"], div_cfg.resample_pts))
    if successes < div_cfg.validity_min:
        return {"n_routes": None, "valid": False, "successes": successes}
    n_clusters = count_routes(trajs, delta)
    occ = [1] * n_clusters
    return {
        "n_routes": n_clusters,
        "routes_per_success": n_clusters / successes,
        "route_entropy": route_entropy(occ),
        "valid": True,
        "successes": successes,
    }


def planner_route_ceiling(
    scene: SceneSpec,
    expert: ExpertOracle,
    div_cfg: DiversityConfig,
    delta: float,
    root: Path | None = None,
) -> dict[str, Any]:
    trajs: list[np.ndarray] = []
    for s in range(div_cfg.n_rollouts):
        path = expert.plan(
            np.array(scene.q_start),
            scene,
            scene.seed + s,
            time_budget_s=expert.cfg.t_validate_s,
        )
        if path is not None:
            trajs.append(resample_path(expert.ee_path(path), div_cfg.resample_pts))
    if len(trajs) < div_cfg.validity_min:
        return {"n_routes": None, "valid": False}
    n = count_routes(trajs, delta)
    return {"n_routes": n, "valid": True}


def evaluate_checkpoint(
    ckpt_path: Path,
    scenes: list[SceneSpec],
    policy_cfg: PolicyConfig,
    eval_cfg: EvalConfig,
    div_cfg: DiversityConfig,
    delta: float = 0.15,
    stochastic: bool = False,
    include_routes: bool = False,
    routes_limit: int | None = None,
    root: Path | None = None,
    include_planner_ceiling: bool = True,
) -> dict[str, Any]:
    if not scenes:
        return {
            "success_rate": float("nan"),
            "collision_rate": float("nan"),
            "timeout_rate": float("nan"),
            "mean_length": float("nan"),
            "mean_recoveries": float("nan"),
            "n_scenes": 0,
        }
    env = PandaReachEnv()
    policy = load_checkpoint(str(ckpt_path), policy_cfg)
    policy.eval()
    results: list[dict[str, Any]] = []
    for scene in scenes:
        r = rollout_episode(env, policy, scene, scene.seed, stochastic=stochastic)
        results.append(r)
    sr = np.mean([r["success"] for r in results])
    scene_success = [float(r["success"]) for r in results]
    per_scene_rec = [r["recoveries"] for r in results]
    per_scene_sr = scene_success
    h3_r = float("nan")
    if len(per_scene_rec) > 2:
        h3_r = float(spearmanr(per_scene_rec, per_scene_sr).statistic or 0.0)
    out: dict[str, Any] = {
        "success_rate": float(sr),
        "collision_rate": float(np.mean([r["collision"] for r in results])),
        "timeout_rate": float(np.mean([r["timeout"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "mean_recoveries": float(np.mean(per_scene_rec)),
        "n_scenes": len(scenes),
        "scene_success": scene_success,
        "h3_spearman": h3_r,
    }
    if include_routes:
        route_scenes = scenes[: routes_limit or eval_cfg.routes_scenes]
        if include_planner_ceiling:
            expert = ExpertOracle(
                load_config((root or Path(".")) / "configs/expert/rrt_connect.yaml", ExpertConfig)
            )
        else:
            expert = None
        route_stats = []
        planner_stats = []
        valid = 0
        for scene in route_scenes:
            rs = evaluate_routes(env, policy, scene, div_cfg, delta)
            route_stats.append(rs)
            if expert is not None:
                ps = planner_route_ceiling(scene, expert, div_cfg, delta)
                planner_stats.append(ps)
            if rs.get("valid"):
                valid += 1
        n_routes = [r["n_routes"] for r in route_stats if r.get("valid")]
        out["mean_n_routes"] = float(np.mean(n_routes)) if n_routes else float("nan")
        out["validity_frac"] = valid / max(len(route_scenes), 1)
        pceil = [
            p["n_routes"] for p in planner_stats if p.get("valid") and p["n_routes"] is not None
        ]
        out["planner_ceiling_mean"] = float(np.mean(pceil)) if pceil else float("nan")
    return out


def load_scenes(path: Path) -> list[SceneSpec]:
    data = json.loads(path.read_text())
    return [SceneSpec.from_json(json.dumps(s)) for s in data["scenes"]]
