"""Mid-round DAgger checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_routes.contracts import Transition
from robot_routes.pipeline.dagger_checkpoint import (
    CHECKPOINT_EVERY,
    clear_dagger_partial,
    load_dagger_partial,
    partial_budget,
    save_dagger_partial,
)


def _row(ep: int) -> Transition:
    return Transition(
        obs=np.zeros(79, np.float32),
        action=np.zeros(7, np.float32),
        q=np.zeros(7, np.float32),
        ee_pos=np.zeros(3, np.float32),
        done=False,
        segment="dagger_label",
        episode_id=ep,
        level=0,
    )


def test_save_and_load_partial(tmp_path: Path) -> None:
    out = tmp_path / "dagger"
    out.mkdir()
    rows = [_row(0), _row(0), _row(1)]
    scenes = {0: '{"seed": 1}', 1: '{"seed": 2}'}
    save_dagger_partial(out, 0, rows, scenes, budget=3)
    loaded_rows, loaded_scenes, budget = load_dagger_partial(out, 0)
    assert len(loaded_rows) == 3
    assert budget == 3
    assert loaded_scenes[0] == scenes[0]
    assert partial_budget(out, 0) == 3
    clear_dagger_partial(out, 0)
    assert partial_budget(out, 0) == 0


def test_checkpoint_every_default() -> None:
    assert CHECKPOINT_EVERY == 500
