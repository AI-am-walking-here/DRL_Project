"""DAgger + RaC outer loop (§6, §15.7.7)."""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Any

import numpy as np

from robot_routes.contracts import JointPath, PathTracker, Transition
from robot_routes.diversity.route_metrics import frechet, resample_path
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.oracle import label as expert_label
from robot_routes.utils.config import DaggerRacConfig


def reversal_actions(q_hist: np.ndarray, clip: float = 0.05) -> np.ndarray:
    return np.clip(np.diff(q_hist[::-1], axis=0), -clip, clip)


def run_rac_intervention(
    env: PandaReachEnv,
    expert: ExpertOracle,
    ring: deque,
    path: JointPath,
    tracker: PathTracker,
    scene: Any,
    cfg: DaggerRacConfig,
    rng: np.random.Generator,
    info: dict,
    delta_reroute: float,
    episode_id: int = 0,
) -> tuple[list[Transition], int, bool]:
    rows: list[Transition] = []
    n_steps = 0
    snaps = list(ring)
    if info.get("collision"):
        pre = next((s for s in reversed(snaps) if s[1] > 0.0), snaps[0])
        env.set_state(pre[0])
        for _ in range(cfg.settle_steps):
            env.step(np.zeros(7))
            n_steps += 1
    ti = next((i for i in range(len(snaps) - 1, -1, -1) if snaps[i][1] >= cfg.eps_safe_m), 0)
    q_hist = np.stack([s[2] for s in snaps[ti:]])
    q_hist = q_hist[: max(2, len(q_hist) - 3)]
    obs = env._encode()
    q_now = env.data.qpos[:7].copy()
    bridge = np.clip(q_hist[-1] - q_now, -0.05, 0.05)[None]
    rec, ok = [], True
    for a in np.vstack([bridge, reversal_actions(q_hist)]):
        if env._contact_violation():
            ok = False
            break
        rec.append(
            Transition(
                obs.astype(np.float32),
                a.astype(np.float32),
                env.data.qpos[:7].astype(np.float32),
                env.data.site_xpos[env._ee_sid].astype(np.float32),
                False,
                "recovery",
                episode_id,
                scene.level,
            )
        )
        obs, _, term, _, info = env.step(a)
        n_steps += 1
        if info["collision"]:
            ok = False
            break
    if ok:
        rows += rec
    suffix_ee = resample_path(expert.ee_path(JointPath(path.waypoints[tracker.idx :])))
    best, best_d = None, -np.inf
    chosen = None
    for attempt in range(cfg.reroute_attempts):
        cand = expert.plan(
            info["q"],
            scene,
            int(rng.integers(2**31)),
            time_budget_s=expert.cfg.t_label_s,
            forbid_similar_to=suffix_ee if cfg.reroute_enabled else None,
        )
        if cand is None:
            continue
        d = frechet(suffix_ee, resample_path(expert.ee_path(cand)))
        if d > best_d:
            best, best_d = cand, d
        if not cfg.reroute_enabled or d >= delta_reroute:
            chosen = cand
            break
    fallback_used = False
    if chosen is None:
        chosen = best
        fallback_used = best is not None
    if chosen is None:
        return rows, n_steps, fallback_used
    tr2 = PathTracker()
    for _ in range(env.cfg.horizon):
        a_c = expert.label(info["q"], chosen, tr2)
        rows.append(
            Transition(
                obs.astype(np.float32),
                a_c.astype(np.float32),
                info["q"].astype(np.float32),
                info["ee_pos"].astype(np.float32),
                False,
                "correction",
                episode_id,
                scene.level,
            )
        )
        obs, _, term, trunc, info = env.step(a_c)
        n_steps += 1
        if info["success"]:
            for _ in range(cfg.settle_steps):
                rows.append(
                    dataclasses.replace(rows[-1], obs=obs, action=np.zeros(7, dtype=np.float32))
                )
                obs, _, _, _, info = env.step(np.zeros(7))
                n_steps += 1
            break
        if info["collision"] or trunc or term:
            break
    return rows, n_steps, fallback_used


def expert_finish_from_state(
    env: PandaReachEnv,
    expert: ExpertOracle,
    scene: Any,
    cfg: DaggerRacConfig,
    rng: np.random.Generator,
    obs: np.ndarray,
    info: dict,
    episode_id: int,
) -> tuple[list[Transition], int]:
    """Expert labels from the current state — no reversal recovery (almost-solve finish)."""
    rows: list[Transition] = []
    n_steps = 0
    path = expert.plan(
        info["q"],
        scene,
        int(rng.integers(2**31)),
        time_budget_s=expert.cfg.t_label_s,
    )
    if path is None:
        return rows, n_steps
    tracker = PathTracker()
    for _ in range(env.cfg.horizon):
        a_c = expert.label(info["q"], path, tracker)
        rows.append(
            Transition(
                obs.astype(np.float32),
                a_c.astype(np.float32),
                info["q"].astype(np.float32),
                info["ee_pos"].astype(np.float32),
                False,
                "dagger_label",
                episode_id,
                scene.level,
            )
        )
        obs, _, term, trunc, info = env.step(a_c)
        n_steps += 1
        if info["success"]:
            for _ in range(cfg.settle_steps):
                rows.append(
                    dataclasses.replace(rows[-1], obs=obs, action=np.zeros(7, dtype=np.float32))
                )
                obs, _, _, _, info = env.step(np.zeros(7))
                n_steps += 1
            break
        if info["collision"] or trunc or term:
            break
    return rows, n_steps


def _episode_min_dist(ee_hist: list[np.ndarray], goal: np.ndarray) -> float:
    if not ee_hist:
        return float("inf")
    return float(min(np.linalg.norm(ee - goal) for ee in ee_hist))


def collect_round(
    env: PandaReachEnv,
    policy: Any,
    expert: ExpertOracle,
    cfg: DaggerRacConfig,
    rng: np.random.Generator,
    level: int = 0,
    delta_reroute: float = 0.15,
    sample_scene_fn: Any = None,
    meta_out: dict[str, Any] | None = None,
    progress_cb: Any = None,
    initial_rows: list[Transition] | None = None,
    episode_scenes_init: dict[int, str] | None = None,
    checkpoint_cb: Any = None,
    checkpoint_every: int = 500,
) -> list[Transition]:
    out: list[Transition] = list(initial_rows or [])
    episode_scenes: dict[int, str] = dict(episode_scenes_init or {})
    b = len(out)
    episode_id = max(episode_scenes.keys(), default=-1) + 1
    last_ckpt = b
    reroute_attempts = 0
    fallback_accepts = 0
    episodes_kept = 0
    episodes_skipped = 0
    bounds = cfg.level_bounds if cfg.level_bounds else [2, 3]
    while b < cfg.budget:
        if sample_scene_fn:
            scene = sample_scene_fn(env, expert, rng, level)
            if scene is None:
                continue
            obs, info = env.reset(options={"scene": scene})
        else:
            obs, info = env.reset(
                seed=int(rng.integers(2**31)), options={"level": level, "level_bounds": bounds}
            )
            scene = env.scene
        episode_scenes[episode_id] = scene.to_json()
        path = expert.plan(
            info["q"],
            scene,
            int(rng.integers(2**31)),
            time_budget_s=expert.cfg.t_label_s,
        )
        if path is None:
            continue
        tracker = PathTracker()
        ring: deque = deque(maxlen=cfg.ring_buffer)
        a_prev = None
        ee_hist: list[np.ndarray] = []
        last_replan = -99
        rows: list[Transition] = []
        intervened = False
        steps = 0
        for t in range(env.cfg.horizon):
            q = info["q"]
            ring.append((env.get_state(), info.get("min_clearance", np.inf), q.copy()))
            if (
                np.linalg.norm(q - path.waypoints[tracker.idx]) > cfg.replan_drift_rad
                and t - last_replan >= 20
            ):
                new = expert.plan(
                    q,
                    scene,
                    scene.seed,
                    time_budget_s=expert.cfg.t_label_s,
                    warm_start=path.waypoints[tracker.idx :],
                )
                last_replan = t
                if new is not None:
                    path, tracker = new, PathTracker()
            a_star = expert_label(q, path, tracker, expert.cfg.lookahead)
            rows.append(
                Transition(
                    obs.astype(np.float32),
                    a_star.astype(np.float32),
                    q.astype(np.float32),
                    info.get("ee_pos", np.zeros(3)).astype(np.float32),
                    False,
                    "dagger_label",
                    episode_id,
                    scene.level,
                )
            )
            a = policy.act(obs, stochastic=True, a_prev=a_prev)
            a_prev = a
            obs, _, term, trunc, info = env.step(a)
            steps += 1
            ee_hist.append(info["ee_pos"])
            stuck = (
                len(ee_hist) > cfg.stuck_steps
                and np.linalg.norm(ee_hist[-1] - ee_hist[-cfg.stuck_steps]) < cfg.stuck_eps_m
            )
            trigger = info["collision"] or info["min_clearance"] < cfg.eps_danger_m or stuck
            if trigger and cfg.rac_enabled:
                min_dist = _episode_min_dist(ee_hist, env._goal)
                almost = min_dist <= cfg.almost_solve_tol_m
                if cfg.rac_only_if_almost and almost and not info["collision"]:
                    extra, n = expert_finish_from_state(
                        env, expert, scene, cfg, rng, obs, info, episode_id
                    )
                    rows += extra
                    steps += n
                    intervened = True
                elif cfg.rac_only_if_almost and not almost:
                    intervened = True
                else:
                    extra, n, fb = run_rac_intervention(
                        env,
                        expert,
                        ring,
                        path,
                        tracker,
                        scene,
                        cfg,
                        rng,
                        info,
                        delta_reroute,
                        episode_id=episode_id,
                    )
                    reroute_attempts += cfg.reroute_attempts
                    if fb:
                        fallback_accepts += 1
                    rows += extra
                    steps += n
                    intervened = True
                break
            if term or trunc:
                break
        min_dist = _episode_min_dist(ee_hist, env._goal)
        success = bool(info.get("success"))
        almost = min_dist <= cfg.almost_solve_tol_m
        if cfg.skip_far_failures and min_dist > cfg.max_far_dist_m and not success:
            episodes_skipped += 1
            episode_id += 1
            continue
        if cfg.skip_far_failures and intervened and not almost and not success:
            episodes_skipped += 1
            episode_id += 1
            continue
        if not intervened:
            rows = [dataclasses.replace(r, segment="clean_rollout") for r in rows]
        out += rows
        b += steps
        episodes_kept += 1
        episode_id += 1
        if progress_cb is not None:
            progress_cb(b, cfg.budget, episode_id)
        if checkpoint_cb is not None and b - last_ckpt >= checkpoint_every:
            checkpoint_cb(out, episode_scenes, b)
            last_ckpt = b
    if meta_out is not None:
        meta_out["reroute_attempts"] = reroute_attempts
        meta_out["fallback_accepts"] = fallback_accepts
        meta_out["episode_scenes"] = episode_scenes
        meta_out["episodes_kept"] = episodes_kept
        meta_out["episodes_skipped"] = episodes_skipped
    return out
