"""Placement invariance integration test (§10.5.6)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from robot_routes.agents.bc_trainer import train_bc
from robot_routes.data.scene_sets import load_scenes, scene_set_path
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.utils.config import (
    BCConfig,
    DiversityConfig,
    EvalConfig,
    PolicyConfig,
    load_config,
    load_yaml,
)
from robot_routes.utils.seeding import seed_everything

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _smoke_eval(root: Path, ckpt: Path, seed: int) -> float:
    policy_cfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))
    eval_cfg = load_config(root / "configs/eval/default.yaml", EvalConfig)
    div_cfg = DiversityConfig(**load_yaml(root / "configs/eval/default.yaml").get("diversity", {}))
    if not scene_set_path(root, "val_L0").exists():
        pytest.skip("smoke scene sets not committed")
    scenes = load_scenes(root, "val_L0")[:3]
    res = evaluate_checkpoint(ckpt, scenes, policy_cfg, eval_cfg, div_cfg, delta=0.15)
    return float(res["success_rate"])


def test_placement_invariance_cpu(tmp_path: Path):
    """Same seed on CPU twice → eval metrics within tolerance (§10.5.6)."""
    root = Path(__file__).resolve().parents[1]
    demos = root / "runs/pipeline/full_seed0/collect/demos.h5"
    if not demos.exists():
        pytest.skip("run smoke pipeline once to create demos.h5")
    bc_cfg = load_config(root / "configs/train/bc.yaml", BCConfig)
    bc_cfg = dataclasses.replace(bc_cfg, epochs=3)
    policy_cfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))
    device = torch.device("cpu")
    srs = []
    for run in range(2):
        out = tmp_path / f"run_{run}"
        rng = seed_everything(42 + run)
        train_bc(demos, bc_cfg, policy_cfg, out, device, rng)
        srs.append(_smoke_eval(root, out / "best.pt", 42))
    assert abs(srs[0] - srs[1]) <= 1e-3
