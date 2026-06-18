"""GPU OOM backoff helpers."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.utils.gpu_oom import (
    DEFAULT_GRID_OOM_RETRIES,
    consume_oom_backoff,
    grid_oom_retries_left,
    write_oom_backoff,
)


def test_write_and_consume_backoff(tmp_path: Path) -> None:
    write_oom_backoff(tmp_path, stage="train_bc", detail="test")
    assert grid_oom_retries_left(tmp_path, DEFAULT_GRID_OOM_RETRIES)
    assert consume_oom_backoff(tmp_path)
    data = json.loads((tmp_path / "oom_backoff.json").read_text())
    assert data["consumed"] is True


def test_no_retry_after_max(tmp_path: Path) -> None:
    path = tmp_path / "oom_backoff.json"
    path.write_text(
        json.dumps(
            {
                "stage": "ppo",
                "retries": DEFAULT_GRID_OOM_RETRIES + 1,
                "consumed": False,
                "request_exclusive": True,
            }
        )
    )
    assert not grid_oom_retries_left(tmp_path, DEFAULT_GRID_OOM_RETRIES)
