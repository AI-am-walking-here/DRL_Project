"""Contract tests (§15.2)."""

from __future__ import annotations

import json

import numpy as np

from robot_routes.contracts import (
    SEGMENT_CODE,
    MjSimState,
    Obstacle,
    SceneSpec,
    Transition,
)


def test_scene_json_roundtrip():
    spec = SceneSpec(
        (Obstacle("box", (0.5, 0.0, 0.3), (0.05, 0.05, 0.05)),),
        (0.6, 0.1, 0.4),
        (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
        0,
        42,
    )
    restored = SceneSpec.from_json(spec.to_json())
    assert restored == spec
    assert json.loads(spec.to_json()) == json.loads(restored.to_json())


def test_segment_codes():
    assert len(SEGMENT_CODE) == 5


def test_transition_shapes():
    t = Transition(
        obs=np.zeros(79, np.float32),
        action=np.zeros(7, np.float32),
        q=np.zeros(7, np.float32),
        ee_pos=np.zeros(3, np.float32),
        done=False,
        segment="full_demo",
        episode_id=0,
        level=0,
    )
    assert t.obs.shape == (79,)


def test_mj_sim_state():
    s = MjSimState(state=np.zeros(100), step_idx=5)
    assert s.step_idx == 5
