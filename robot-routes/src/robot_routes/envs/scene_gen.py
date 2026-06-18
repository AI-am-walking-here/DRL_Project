"""Randomized obstacle scenes + curriculum (§3.1, §15.7.1)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from robot_routes.contracts import Q_HOME, Obstacle, SceneSpec
from robot_routes.utils.config import SceneConfig


def point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p: np.ndarray, o: Obstacle) -> bool:
    c = np.asarray(o.center)
    if o.kind == "sphere":
        return float(np.linalg.norm(p - c)) <= o.size[0]
    half = np.asarray(o.size)
    return bool(np.all(np.abs(p - c) <= half))


def _noisy_home(rng: np.random.Generator, cfg: SceneConfig) -> tuple[float, ...]:
    q = Q_HOME + rng.uniform(-cfg.start_noise_rad, cfg.start_noise_rad, 7)
    return tuple(float(x) for x in q)


def sample_scene(
    level: int,
    rng: np.random.Generator,
    cfg: SceneConfig,
    level_bounds: list[int],
    start_ee: np.ndarray,
    penetrates_start: Callable[[SceneSpec], bool],
    unseen: bool = False,
) -> SceneSpec | None:
    lo, hi = level_bounds
    n_obs = int(rng.integers(lo, hi + 1))
    box_min = cfg.unseen_half_min if unseen else cfg.box_half_min
    box_max = cfg.unseen_half_max if unseen else cfg.box_half_max
    for _ in range(200):
        obstacles: list[Obstacle] = []
        while len(obstacles) < n_obs:
            r = rng.uniform(cfg.r_min, cfg.r_max)
            th = rng.uniform(-np.pi, np.pi)
            c = np.array(
                [r * np.cos(th), r * np.sin(th), rng.uniform(cfg.z_min, cfg.z_max)],
                dtype=np.float64,
            )
            if c[0] <= 0.1:
                continue
            if rng.random() < 0.5:
                size = tuple(float(x) for x in rng.uniform(box_min, box_max, 3))
                kind: str = "box"
            else:
                size = (
                    float(rng.uniform(cfg.sph_r_min, cfg.sph_r_max)),
                    0.0,
                    0.0,
                )
                kind = "sphere"
            br = max(size) if kind == "box" else size[0]
            ok = all(
                np.linalg.norm(c - np.array(o.center))
                > 0.8 * (br + (max(o.size) if o.kind == "box" else o.size[0]))
                for o in obstacles
            )
            if ok:
                obstacles.append(Obstacle(kind=kind, center=tuple(c), size=size))  # type: ignore[arg-type]
        r_g = rng.uniform(cfg.r_min, cfg.r_max)
        th_g = rng.uniform(-np.pi, np.pi)
        goal = np.array(
            [
                abs(r_g) * np.cos(th_g),
                abs(r_g) * np.sin(th_g),
                rng.uniform(cfg.z_min, cfg.z_max),
            ],
            dtype=np.float64,
        )
        if goal[0] <= 0.1 or np.linalg.norm(goal - start_ee) < cfg.goal_min_dist:
            continue
        if any(_inside(goal, o) for o in obstacles):
            continue
        spec = SceneSpec(
            tuple(obstacles),
            tuple(float(x) for x in goal),
            _noisy_home(rng, cfg),
            level,
            int(rng.integers(2**31)),
        )
        if penetrates_start(spec):
            continue
        return spec
    return None


def make_slots(scene: SceneSpec, start_ee: np.ndarray) -> np.ndarray:
    goal = np.asarray(scene.goal, dtype=np.float64)
    rows: list[tuple[float, np.ndarray]] = []
    for o in scene.obstacles:
        c = np.asarray(o.center, dtype=np.float64)
        row = np.array(
            [0.0 if o.kind == "box" else 1.0, *c, *o.size],
            dtype=np.float64,
        )
        rows.append((point_seg_dist(c, start_ee, goal), row))
    rows.sort(key=lambda t: t[0])
    slots = np.zeros((8, 7), dtype=np.float64)
    for i, (_, row) in enumerate(rows[:8]):
        slots[i] = row
    return slots


def encode_obs(
    q: np.ndarray,
    qvel: np.ndarray,
    ee: np.ndarray,
    goal: np.ndarray,
    slots: np.ndarray,
) -> np.ndarray:
    return np.concatenate([q, qvel, ee, goal, goal - ee, slots.ravel()]).astype(np.float32)
