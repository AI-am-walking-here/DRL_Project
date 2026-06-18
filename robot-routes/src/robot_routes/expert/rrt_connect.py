"""Bidirectional RRT-Connect in joint space (§4.1, §15.7.3)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from robot_routes.contracts import JointPath
from robot_routes.expert.collision import CollisionChecker

if TYPE_CHECKING:
    from robot_routes.utils.config import ExpertConfig


def edge_free(cc: CollisionChecker, qa: np.ndarray, qb: np.ndarray, edge_check_rad: float) -> bool:
    n = int(np.ceil(np.abs(qb - qa).max() / edge_check_rad)) + 1
    for q in np.linspace(qa, qb, n):
        if not cc.free(q):
            return False
    return True


def _sample_limits(rng: np.random.Generator, cc: CollisionChecker) -> np.ndarray:
    lo = cc.model.jnt_range[:7, 0]
    hi = cc.model.jnt_range[:7, 1]
    for _ in range(100):
        q = rng.uniform(lo, hi)
        if cc.free(q):
            return q
    return rng.uniform(lo, hi)


def _extend(
    tree: tuple[list[np.ndarray], list[int]],
    q_to: np.ndarray,
    cc: CollisionChecker,
    step: float,
    edge_rad: float,
):
    nodes, parent = tree
    arr = np.asarray(nodes)
    i = int(np.argmin(((arr - q_to) ** 2).sum(axis=1)))
    d = q_to - nodes[i]
    q_new = nodes[i] + d * min(1.0, step / (np.abs(d).max() + 1e-12))
    if edge_free(cc, nodes[i], q_new, edge_rad):
        nodes.append(q_new)
        parent.append(i)
        return q_new
    return None


def _connect(tree, q_to, cc, step, edge_rad):
    nodes = tree[0]
    while True:
        q_new = _extend(tree, q_to, cc, step, edge_rad)
        if q_new is None:
            return None
        if np.abs(q_new - q_to).max() < 1e-6:
            return q_new


def _trace(tree, leaf: int) -> list[np.ndarray]:
    nodes, parent = tree
    path = [nodes[leaf]]
    while parent[leaf] >= 0:
        leaf = parent[leaf]
        path.append(nodes[leaf])
    return path


def _shortcut(
    path: list[np.ndarray],
    cc: CollisionChecker,
    iters: int,
    rng: np.random.Generator,
    edge_rad: float,
) -> list[np.ndarray]:
    if len(path) < 2:
        return path
    arr = path[:]
    for _ in range(iters):
        i, j = sorted(rng.integers(0, len(arr), size=2))
        if j - i < 2:
            continue
        if edge_free(cc, arr[i], arr[j], edge_rad):
            arr = arr[: i + 1] + arr[j:]
    return arr


def _resample(path: np.ndarray, waypoint_rad: float) -> np.ndarray:
    if len(path) < 2:
        return path
    out = [path[0]]
    for i in range(1, len(path)):
        seg = np.linspace(
            out[-1], path[i], int(np.ceil(np.abs(path[i] - out[-1]).max() / waypoint_rad)) + 1
        )
        out.extend(seg[1:])
    return np.array(out)


def rrt_connect(
    q_start: np.ndarray,
    goal_roots: list[np.ndarray],
    cc: CollisionChecker,
    rng: np.random.Generator,
    cfg: ExpertConfig,
    deadline: float,
    warm_start: np.ndarray | None = None,
) -> JointPath | None:
    Ta: tuple[list[np.ndarray], list[int]] = ([q_start.copy()], [-1])
    if warm_start is not None and len(warm_start) > 0:
        for i, wp in enumerate(warm_start):
            Ta[0].append(wp.copy())
            Ta[1].append(i)
    Tb = ([r.copy() for r in goal_roots], [-1] * len(goal_roots))
    for _ in range(cfg.max_iters):
        if time.monotonic() > deadline:
            return None
        q_rand = (
            goal_roots[rng.integers(len(goal_roots))]
            if rng.random() < cfg.goal_bias
            else _sample_limits(rng, cc)
        )
        q_new = _extend(Ta, q_rand, cc, cfg.step_size, cfg.edge_check_rad)
        if q_new is not None:
            q_conn = _connect(Tb, q_new, cc, cfg.step_size, cfg.edge_check_rad)
            if q_conn is not None and np.abs(q_conn - q_new).max() < 1e-9:
                path = _trace(Ta, -1)[::-1] + _trace(Tb, -1)
                path = _shortcut(path, cc, cfg.shortcut_iters, rng, cfg.edge_check_rad)
                return JointPath(_resample(np.array(path), cfg.waypoint_rad))
        Ta, Tb = Tb, Ta
    return None
