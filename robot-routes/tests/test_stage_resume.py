"""Stage resume tests (collect + curriculum)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from robot_routes.data.schema import write_shard
from robot_routes.pipeline.stage_resume import (
    artifact_complete,
    detect_collect_resume,
    detect_curriculum_resume,
    eval_artifact_valid,
    eval_resume_meta,
)


def _write_min_shard(path: Path, n_eps: int = 5) -> None:
    from robot_routes.contracts import Transition

    rows = [
        Transition(
            np.zeros(79, np.float32),
            np.zeros(7, np.float32),
            np.zeros(7, np.float32),
            np.zeros(3, np.float32),
            False,
            "full_demo",
            i,
            0,
        )
        for i in range(n_eps)
    ]
    write_shard(path, rows, ['{"seed": 0}'] * n_eps)


def test_collect_merge_only(tmp_path: Path) -> None:
    out = tmp_path / "collect"
    out.mkdir()
    _write_min_shard(out / "demos_w0.h5", 10)
    _write_min_shard(out / "demos_w1.h5", 10)
    st = detect_collect_resume(out, n_demos=20, n_workers=2)
    assert st.merge_only
    assert st.pending_workers == []
    assert st.resumed


def test_collect_pending_workers(tmp_path: Path) -> None:
    out = tmp_path / "collect"
    out.mkdir()
    _write_min_shard(out / "demos_w0.h5", 10)
    st = detect_collect_resume(out, n_demos=20, n_workers=2)
    assert st.pending_workers == [1]
    assert not st.done


def test_collect_done_merged(tmp_path: Path) -> None:
    out = tmp_path / "collect"
    out.mkdir()
    _write_min_shard(out / "demos.h5", 20)
    st = detect_collect_resume(out, n_demos=20, n_workers=2)
    assert st.done


def test_curriculum_resume_step_2(tmp_path: Path) -> None:
    out = tmp_path / "curriculum"
    dagger = tmp_path / "dagger"
    out.mkdir()
    dagger.mkdir()
    for step in range(2):
        (out / f"ckpt_{step}").mkdir()
        (out / f"ckpt_{step}" / "best.pt").write_bytes(b"x")
        _write_min_shard(out / f"merged_cur_{step}.h5", 1)
    (out / "curriculum_state.json").write_text(
        '{"level": 1, "history": [{"level": 0, "success": 0.5}, {"level": 1, "success": 0.6}]}'
    )
    bc = tmp_path / "demos.h5"
    _write_min_shard(bc, 1)
    st = detect_curriculum_resume(out, dagger, 6, bc, total_steps=4)
    assert st.start_step == 2
    assert st.resumed
    assert st.policy_ckpt == out / "ckpt_1" / "best.pt"


def test_eval_artifact_valid(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    path = tmp_path / "val_eval.json"
    path.write_text(
        json.dumps(
            {
                "success_rate": 0.5,
                "_resume_meta": eval_resume_meta(ckpt, profile="full", post_ppo=False),
            }
        )
    )
    assert eval_artifact_valid(path, ckpt, meta={"profile": "full", "post_ppo": False})
    assert not eval_artifact_valid(path, ckpt, meta={"profile": "full", "post_ppo": True})
    other = tmp_path / "other.pt"
    other.write_bytes(b"y")
    assert not eval_artifact_valid(path, other)


def test_eval_artifact_rejects_legacy(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"x")
    path = tmp_path / "val_eval.json"
    path.write_text('{"success_rate": 0.5}')
    assert not eval_artifact_valid(path, ckpt)


def test_artifact_complete_nonempty(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.pt"
    assert not artifact_complete(p)
    p.write_bytes(b"x")
    assert artifact_complete(p)
    p.write_bytes(b"")
    assert not artifact_complete(p)
