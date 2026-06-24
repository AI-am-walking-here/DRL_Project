"""G-DATA and related gate tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_routes.contracts import Transition
from robot_routes.data.schema import write_shard
from robot_routes.expert.collision import CollisionChecker
from robot_routes.pipeline.gates import _verify_recovery_segments, gate_data
from robot_routes.utils.config import DaggerRacConfig, load_config


def _row(segment: str, episode_id: int = 0) -> Transition:
    return Transition(
        np.zeros(79, np.float32),
        np.zeros(7, np.float32),
        np.zeros(7, np.float32),
        np.zeros(3, np.float32),
        False,
        segment,  # type: ignore[arg-type]
        episode_id,
        0,
    )


def test_gate_data_accepts_low_global_recovery_ratio(tmp_path: Path) -> None:
    """RaC shards normally have many more correction than recovery transitions."""
    rows = [_row("recovery")] * 2 + [_row("correction")] * 20
    shard = tmp_path / "shard.h5"
    write_shard(shard, rows, ['{"seed": 0}'])
    cfg = load_config(Path("configs/train/dagger_rac.yaml"), DaggerRacConfig)
    ok, msg = gate_data(shard, cfg, cc=None)
    assert ok, msg


def test_gate_data_rejects_corrections_without_recovery(tmp_path: Path) -> None:
    rows = [_row("correction")] * 5
    shard = tmp_path / "shard.h5"
    write_shard(shard, rows, ['{"seed": 0}'])
    cfg = load_config(Path("configs/train/dagger_rac.yaml"), DaggerRacConfig)
    ok, msg = gate_data(shard, cfg, cc=None)
    assert not ok
    assert "without recovery" in msg


def test_round0_shard_passes_g_data() -> None:
    shard = Path("runs/grid/rac_noreroute_seed0/dagger/round_0.h5")
    if not shard.exists():
        return
    import json

    meta = json.loads(shard.with_name("round_0_meta.json").read_text())
    cfg = load_config(Path("configs/train/dagger_rac.yaml"), DaggerRacConfig)
    cc = CollisionChecker()
    ok, msg = gate_data(shard, cfg, meta=meta, cc=cc)
    assert ok, msg
