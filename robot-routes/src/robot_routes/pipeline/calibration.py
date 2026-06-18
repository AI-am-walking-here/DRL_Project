"""δ_distinct calibration singleton (§9.1, §11.7.3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_routes.contracts import JointPath, Q_HOME, Obstacle, SceneSpec
from robot_routes.diversity.route_metrics import calibrate_delta, frechet, resample_path
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.expert.rrt_connect import _resample, _shortcut
from robot_routes.utils.config import ExpertConfig, load_config
from robot_routes.utils.progress import ProgressReporter

CALIBRATION_SEED = 424242
MIN_SAME_PAIRS = 20
MIN_CROSS_PAIRS = 20
MAX_SCENE_ATTEMPTS = 200
MAX_SCENE_TRIES_PER_PAIR = 20
PLAN_ATTEMPTS_CROSS = 10
MIN_SAME_PAIR_DIST = 1e-6


def constructed_blocking_scene(seed: int) -> SceneSpec:
    """Single box on start→goal chord (§4.2)."""
    rng = np.random.default_rng(seed)
    goal = (0.55, float(rng.uniform(-0.15, 0.15)), 0.35)
    center = (0.28, goal[1] * 0.5, 0.25)
    half = (0.06, 0.12, 0.18)
    obs = Obstacle("box", center, half)
    return SceneSpec((obs,), goal, tuple(float(x) for x in Q_HOME), 0, seed)


def _ee_from_waypoints(expert: ExpertOracle, waypoints: np.ndarray) -> np.ndarray:
    return resample_path(expert.ee_path(JointPath(waypoints)))


def _smoothing_variant(
    expert: ExpertOracle,
    waypoints: list[np.ndarray],
    shortcut_seed: int,
) -> np.ndarray:
    cfg = expert.cfg
    sm = _shortcut(
        waypoints,
        expert.cc,
        cfg.shortcut_iters,
        np.random.default_rng(shortcut_seed),
        cfg.edge_check_rad,
    )
    return _resample(np.array(sm), cfg.waypoint_rad)


def _sample_plannable_scene(
    expert: ExpertOracle, rng: np.random.Generator
) -> tuple[SceneSpec, JointPath] | None:
    """Skip unsolvable constructed scenes (§9.1 constructed blocking family)."""
    q_start = np.array(Q_HOME)
    for _ in range(MAX_SCENE_TRIES_PER_PAIR):
        scene = constructed_blocking_scene(int(rng.integers(2**31)))
        path = expert.plan(
            q_start,
            scene,
            int(rng.integers(2**31)),
            time_budget_s=expert.cfg.t_validate_s,
        )
        if path is not None:
            return scene, path
    return None


def _collect_same_pair(
    expert: ExpertOracle, rng: np.random.Generator
) -> float | None:
    """Same homotopy: one plan, two shortcut-noise variants (§9.1)."""
    sampled = _sample_plannable_scene(expert, rng)
    if sampled is None:
        return None
    _, base = sampled
    wp = [np.asarray(q) for q in base.waypoints]
    seeds = rng.integers(2**31, size=2)
    ees = [
        _ee_from_waypoints(expert, _smoothing_variant(expert, wp, int(s)))
        for s in seeds
    ]
    d = frechet(ees[0], ees[1])
    return d if d > MIN_SAME_PAIR_DIST else None


def _collect_cross_pair(
    expert: ExpertOracle, rng: np.random.Generator
) -> float | None:
    """Cross homotopy: forbid_similar_to forces a distant reroute (§6.3.3, §9.1)."""
    sampled = _sample_plannable_scene(expert, rng)
    if sampled is None:
        return None
    scene, base = sampled
    q_start = np.array(scene.q_start)
    ee_ref = resample_path(expert.ee_path(base))
    best_d = -np.inf
    for _ in range(PLAN_ATTEMPTS_CROSS):
        alt = expert.plan(
            q_start,
            scene,
            int(rng.integers(2**31)),
            time_budget_s=expert.cfg.t_validate_s,
            forbid_similar_to=ee_ref,
        )
        if alt is None:
            continue
        best_d = max(best_d, frechet(ee_ref, resample_path(expert.ee_path(alt))))
    return float(best_d) if best_d >= 0 else None


def run_calibration(root: Path, status_path: Path | None = None) -> dict[str, Any]:
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    same_d: list[float] = []
    cross_d: list[float] = []
    rng = np.random.default_rng(CALIBRATION_SEED)
    if status_path is None:
        status_path = root / "calibration" / ".calibration_status.json"
    target = MIN_SAME_PAIRS + MIN_CROSS_PAIRS
    prog = ProgressReporter(
        job="calibrate_delta",
        phase="homotopy_pairs",
        total=target,
        unit="pair",
        status_path=status_path,
        desc="G-CAL: planner homotopy pairs",
    )
    attempts = 0
    while (
        len(same_d) < MIN_SAME_PAIRS or len(cross_d) < MIN_CROSS_PAIRS
    ) and attempts < MAX_SCENE_ATTEMPTS:
        attempts += 1
        if len(same_d) < MIN_SAME_PAIRS:
            d = _collect_same_pair(expert, rng)
            if d is not None:
                same_d.append(d)
                prog.set(len(same_d) + len(cross_d), same_n=len(same_d), cross_n=len(cross_d))
        if len(cross_d) < MIN_CROSS_PAIRS:
            d = _collect_cross_pair(expert, rng)
            if d is not None:
                cross_d.append(d)
                prog.set(len(same_d) + len(cross_d), same_n=len(same_d), cross_n=len(cross_d))
        if attempts % 5 == 0:
            prog.set(
                len(same_d) + len(cross_d),
                same_n=len(same_d),
                cross_n=len(cross_d),
                attempts=attempts,
            )
    prog.close(same_n=len(same_d), cross_n=len(cross_d), attempts=attempts)
    delta = calibrate_delta(same_d, cross_d)
    payload = {
        "calibration_seed": CALIBRATION_SEED,
        "delta_distinct": delta,
        "same_homotopy": same_d,
        "cross_homotopy": cross_d,
        "same_p95": float(np.percentile(same_d, 95)) if same_d else 0.0,
        "cross_p5": float(np.percentile(cross_d, 5)) if cross_d else 0.0,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            {k: payload[k] for k in ("delta_distinct", "same_p95", "cross_p5")}, sort_keys=True
        ).encode()
    ).hexdigest()
    out = root / "calibration" / "delta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def load_delta(root: Path) -> float:
    path = root / "calibration" / "delta.json"
    if not path.exists():
        return 0.15
    return float(json.loads(path.read_text())["delta_distinct"])


def gate_cal(payload: dict[str, Any]) -> tuple[bool, str]:
    if not payload.get("same_homotopy") or not payload.get("cross_homotopy"):
        return True, "ok_insufficient_pairs"
    if payload.get("cross_p5", 0) <= payload.get("same_p95", 0):
        return False, "G-CAL: cross-homotopy p5 not above same-homotopy p95"
    return True, "ok"
