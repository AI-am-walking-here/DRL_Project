"""Experiment condition wiring (§10.1)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from robot_routes.utils.config import CurriculumConfig, DaggerRacConfig, load_config


def load_grid(root: Path, config: str | Path = "configs/grid.yaml") -> dict[str, Any]:
    """Load a grid spec. `config` may be absolute or relative to `root`."""
    path = Path(config)
    if not path.is_absolute():
        path = root / path
    return yaml.safe_load(path.read_text())


def condition_spec(grid: dict[str, Any], name: str) -> dict[str, Any]:
    for c in grid.get("conditions", []):
        if c["name"] == name:
            return c
    raise KeyError(f"unknown condition: {name}")


def dagger_overrides(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "rac_enabled" in spec:
        out["rac_enabled"] = bool(spec["rac_enabled"])
    if "reroute_enabled" in spec:
        out["reroute_enabled"] = bool(spec["reroute_enabled"])
    return out


def curriculum_overrides(spec: dict[str, Any]) -> dict[str, Any]:
    if "synthetic_obs" in spec:
        return {"synthetic_obs": bool(spec["synthetic_obs"])}
    return {}


def apply_dagger_config(
    root: Path, spec: dict[str, Any], rel: str = "configs/train/dagger_rac.yaml"
):
    cfg = load_config(root / rel, DaggerRacConfig)
    overrides = dagger_overrides(spec)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def apply_curriculum_config(
    root: Path, spec: dict[str, Any], rel: str = "configs/train/curriculum.yaml"
):
    cfg = load_config(root / rel, CurriculumConfig)
    overrides = curriculum_overrides(spec)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def stage_list(spec: dict[str, Any]) -> list[str]:
    return list(spec.get("stages", []))
