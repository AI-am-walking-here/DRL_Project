"""G-PPO go/no-go predicate (§11.7.2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robot_routes.diversity.route_metrics import paired_bootstrap_ci
from robot_routes.pipeline.gates import gate_ppo


def parse_deadline(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def sibling_run_dir(runs_root: Path, condition: str, seed: int) -> Path:
    return runs_root / f"{condition}_seed{seed}"


def load_scene_success(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([])
    data = json.loads(path.read_text())
    return np.array(data.get("scene_success", []), dtype=float)


def evaluate_g_ppo(
    *,
    run_dir: Path,
    runs_root: Path,
    seed: int,
    ref_condition: str,
    ppo_deadline: str,
    head_compatible: bool,
    rng: np.random.Generator | None = None,
) -> tuple[bool, str, dict]:
    rng = rng or np.random.default_rng(seed)
    val_path = run_dir / "eval/val_eval.json"
    ref_path = sibling_run_dir(runs_root, ref_condition, seed) / "eval/val_eval.json"
    before = datetime.now(timezone.utc) < parse_deadline(ppo_deadline)
    validity_frac = 0.0
    ci_excludes_zero = False
    meta: dict = {"before_deadline": before, "ref_exists": ref_path.exists()}
    if val_path.exists():
        val = json.loads(val_path.read_text())
        validity_frac = float(val.get("validity_frac", 0.0))
        meta["validity_frac"] = validity_frac
    a = load_scene_success(val_path)
    b = load_scene_success(ref_path)
    if len(a) and len(a) == len(b):
        lo, hi = paired_bootstrap_ci(a, b, rng)
        ci_excludes_zero = lo > 0
        meta["paired_ci"] = [lo, hi]
    elif ref_path.exists() and val_path.exists():
        meta["paired_ci"] = "length_mismatch"
    go, reason = gate_ppo(
        ci_excludes_zero=ci_excludes_zero,
        validity_frac=validity_frac,
        before_deadline=before,
        head_compatible=head_compatible,
    )
    return go, reason, meta


def resolve_compare_evals(
    runs_root: Path,
    condition: str,
    seed: int,
) -> dict[str, Path]:
    """Map hypothesis keys to sibling condition eval JSON paths."""
    mapping = {
        "H1": sibling_run_dir(runs_root, "bc_dagger", seed) / "eval/test_eval.json",
        "H2": sibling_run_dir(runs_root, "rac_noreroute", seed) / "eval/test_eval.json",
    }
    if condition != "full":
        mapping = {}
    return {k: v for k, v in mapping.items() if v.exists()}
