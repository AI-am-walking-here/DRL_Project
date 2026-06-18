"""Tests for frozen scene-set manifest (§10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.wp1
def test_scene_manifest_exists():
    manifest = ROOT / "data/scenes/manifest.json"
    if not manifest.exists():
        pytest.skip("run make scene-sets PROFILE=smoke first")
    from robot_routes.data.scene_sets import verify_scene_sets

    verify_scene_sets(ROOT, profile="smoke")
