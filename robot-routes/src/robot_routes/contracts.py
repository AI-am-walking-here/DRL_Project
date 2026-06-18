"""Single source of truth for all cross-module types.
Spec §15.2. Changing any signature here requires an ISSUES.md entry (§14.4 rule 1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    import torch

Vec3 = np.ndarray  # shape (3,), float64
Vec7 = np.ndarray  # shape (7,), float64

SegmentType = Literal["full_demo", "dagger_label", "clean_rollout", "recovery", "correction"]
SEGMENT_CODE: dict[SegmentType, int] = {
    "full_demo": 0,
    "dagger_label": 1,
    "clean_rollout": 2,
    "recovery": 3,
    "correction": 4,
}
SEGMENT_NAME: dict[int, SegmentType] = {v: k for k, v in SEGMENT_CODE.items()}

Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)


@dataclass(frozen=True)
class Obstacle:
    kind: Literal["box", "sphere"]
    center: tuple[float, float, float]
    size: tuple[float, float, float]  # box: half-extents; sphere: (r, 0.0, 0.0)


@dataclass(frozen=True)
class SceneSpec:
    obstacles: tuple[Obstacle, ...]
    goal: tuple[float, float, float]
    q_start: tuple[float, ...]
    level: int
    seed: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> SceneSpec:
        d = json.loads(text)
        obstacles = tuple(
            Obstacle(o["kind"], tuple(o["center"]), tuple(o["size"])) for o in d["obstacles"]
        )
        return SceneSpec(
            obstacles=obstacles,
            goal=tuple(d["goal"]),
            q_start=tuple(d["q_start"]),
            level=int(d["level"]),
            seed=int(d["seed"]),
        )


@dataclass(frozen=True)
class MjSimState:
    state: np.ndarray  # mj_getState(..., mjSTATE_INTEGRATION), float64
    step_idx: int


@dataclass
class JointPath:
    waypoints: np.ndarray  # (M, 7) float64


@dataclass
class PathTracker:
    idx: int = 0


@dataclass(frozen=True)
class Transition:
    obs: np.ndarray
    action: np.ndarray
    q: np.ndarray
    ee_pos: np.ndarray
    done: bool
    segment: SegmentType
    episode_id: int
    level: int


class Expert(Protocol):
    def plan(
        self,
        q_start: Vec7,
        scene: SceneSpec,
        rng_seed: int,
        *,
        time_budget_s: float,
        forbid_similar_to: np.ndarray | None = None,
        warm_start: np.ndarray | None = None,
    ) -> JointPath | None: ...

    def label(self, q: Vec7, path: JointPath, tracker: PathTracker) -> Vec7: ...

    def ee_path(self, path: JointPath) -> np.ndarray: ...


class Env(Protocol):
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]: ...

    def step(self, action: Vec7) -> tuple[np.ndarray, float, bool, bool, dict]: ...

    def get_state(self) -> MjSimState: ...

    def set_state(self, s: MjSimState) -> np.ndarray: ...

    def min_clearance(self) -> float: ...

    @property
    def scene(self) -> SceneSpec: ...


class Policy(Protocol):
    def act(self, obs: np.ndarray, *, stochastic: bool, a_prev: Vec7 | None = None) -> Vec7: ...

    def nll(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> torch.Tensor: ...
