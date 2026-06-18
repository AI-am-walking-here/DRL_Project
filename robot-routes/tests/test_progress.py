"""Tests for progress reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path

from robot_routes.utils.progress import ProgressReporter


def test_progress_writes_status(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    prog = ProgressReporter(
        job="test",
        phase="phase_a",
        total=4,
        unit="item",
        status_path=status,
        desc="test job",
    )
    prog.update(2, note="mid")
    data = json.loads(status.read_text())
    assert data["current"] == 2
    assert data["total"] == 4
    assert data["pct"] == 50.0
    assert data["done"] is False
    assert data["note"] == "mid"
    prog.close(final=True, ok=True)
    done = json.loads(status.read_text())
    assert done["done"] is True
    assert done["current"] == 4
    assert done["ok"] is True
