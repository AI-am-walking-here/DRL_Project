"""Expert oracle: RRT-Connect planner + pure-pursuit labels (§4.2)."""

from __future__ import annotations

import time

import mujoco
import numpy as np

from robot_routes.contracts import Q_HOME, JointPath, PathTracker, SceneSpec
from robot_routes.expert.collision import CollisionChecker
from robot_routes.expert.rrt_connect import rrt_connect
from robot_routes.utils.config import ExpertConfig

SCENE_XML = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "envs"
    / "assets"
    / "reach_scene.xml"
)


def ik_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    sid: int,
    q0: np.ndarray,
    jlo: np.ndarray,
    jhi: np.ndarray,
    iters: int = 200,
    damp: float = 1e-2,
    tol: float = 0.04,
    margin: float = 0.02,
) -> np.ndarray | None:
    q = q0.copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            inside = np.all(q >= jlo + margin) and np.all(q <= jhi - margin)
            return q if inside else None
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), err)
        q = np.clip(q + dq, jlo + margin, jhi - margin)
    return None


def goal_roots(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    sid: int,
    jlo: np.ndarray,
    jhi: np.ndarray,
    rng: np.random.Generator,
    cc: CollisionChecker,
    n: int = 16,
    dedup: float = 0.2,
) -> list[np.ndarray]:
    sols: list[np.ndarray] = []
    for fam_q4 in (-0.6, -2.4):
        for _ in range(n // 2):
            seed = Q_HOME.copy() + rng.uniform(-0.4, 0.4, 7)
            seed[3] = fam_q4 + rng.uniform(-0.2, 0.2)
            q = ik_dls(model, data, target, sid, seed, jlo, jhi)
            if q is None or not cc.free(q):
                continue
            if all(np.linalg.norm(q - s) >= dedup for s in sols):
                sols.append(q)
    return sols


def label(
    q: np.ndarray,
    path: JointPath,
    tracker: PathTracker,
    lookahead: int = 3,
    window: int = 10,
    clip: float = 0.05,
) -> np.ndarray:
    wp = path.waypoints
    seg = wp[tracker.idx : tracker.idx + window + 1]
    tracker.idx += int(np.argmin(np.linalg.norm(seg - q, axis=1)))
    if tracker.idx >= len(wp) - 1:
        return np.zeros(7)
    tgt = wp[min(tracker.idx + lookahead, len(wp) - 1)]
    return np.clip(tgt - q, -clip, clip)


class ExpertOracle:
    def __init__(self, cfg: ExpertConfig | None = None) -> None:
        self.cfg = cfg or ExpertConfig()
        self.cc = CollisionChecker(self.cfg.margin_plan_m)
        self.model = self.cc.model
        self.data = self.cc.data
        self._ee_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._jlo = self.model.jnt_range[:7, 0]
        self._jhi = self.model.jnt_range[:7, 1]

    def plan(
        self,
        q_start: np.ndarray,
        scene: SceneSpec,
        rng_seed: int,
        *,
        time_budget_s: float,
        forbid_similar_to: np.ndarray | None = None,
        warm_start: np.ndarray | None = None,
    ) -> JointPath | None:
        rng = np.random.default_rng(rng_seed)
        self.cc.set_scene(scene)
        target = np.asarray(scene.goal, dtype=np.float64)
        roots = goal_roots(
            self.model,
            self.data,
            target,
            self._ee_sid,
            self._jlo,
            self._jhi,
            rng,
            self.cc,
            self.cfg.ik_restarts,
            self.cfg.ik_dedup_rad,
        )
        if not roots:
            return None
        deadline = time.monotonic() + time_budget_s
        best_path: JointPath | None = None
        best_d = -np.inf
        n_attempts = 5 if forbid_similar_to is not None else 1
        for attempt in range(n_attempts):
            path = rrt_connect(
                np.asarray(q_start, dtype=np.float64),
                roots,
                self.cc,
                np.random.default_rng(rng_seed + attempt),
                self.cfg,
                deadline,
                warm_start=warm_start,
            )
            if path is None:
                continue
            if forbid_similar_to is not None:
                from robot_routes.diversity.route_metrics import frechet, resample_path

                ee_new = resample_path(self.ee_path(path))
                d = frechet(forbid_similar_to, ee_new)
                if d > best_d:
                    best_path, best_d = path, d
                if d >= getattr(self, "delta_reroute", 0.15):
                    return path
            else:
                return path
        return best_path

    def label(self, q: np.ndarray, path: JointPath, tracker: PathTracker) -> np.ndarray:
        return label(q, path, tracker, self.cfg.lookahead)

    def ee_path(self, path: JointPath) -> np.ndarray:
        pts = []
        for wp in path.waypoints:
            self.cc.set_q(wp)
            pts.append(self.data.site_xpos[self._ee_sid].copy())
        return np.array(pts)
