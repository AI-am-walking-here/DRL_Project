"""Executable gate predicates (§11.7.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_routes.contracts import SEGMENT_NAME, SceneSpec
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.expert.collision import CollisionChecker
from robot_routes.utils.config import DiversityConfig, EvalConfig, PolicyConfig


# Recovery rows are validated with env-equivalent contact detection (not margin_plan).
def gate_bc(success_rate: float, soft: float = 0.25, target: float = 0.40) -> tuple[bool, str, str]:
    if success_rate >= soft:
        return True, "pass", "ok"
    if success_rate >= target:
        return True, "pass", "above_target"
    return False, "fail", f"val_L0 success {success_rate:.3f} < {soft}"


def _verify_recovery_segments(shard: Path, cc: CollisionChecker) -> tuple[bool, str]:
    with h5py.File(shard, "r") as f:
        segs = f["segment"][:]
        q = f["q"][:]
        ep = f["episode_id"][:]
        scenes_raw = list(f["episodes/scene_json"].asstr()[:])
    rec_idx = np.where(segs == 3)[0]
    if len(rec_idx) == 0:
        return True, "ok"
    for i in rec_idx:
        eid = int(ep[i])
        if eid >= len(scenes_raw):
            continue
        scene = SceneSpec.from_json(scenes_raw[eid])
        cc.set_scene(scene)
        cc.set_q(q[i])
        if cc.in_contact():
            return False, f"recovery segment {i} in contact"
    return True, "ok"


def gate_data(
    shard: Path,
    cfg: Any,
    meta: dict[str, Any] | None = None,
    cc: CollisionChecker | None = None,
) -> tuple[bool, str]:
    if not shard.exists():
        return False, "missing shard"
    with h5py.File(shard, "r") as f:
        if np.isnan(f["obs"][:]).any() or np.isnan(f["action"][:]).any():
            return False, "NaN in shard"
        segs = f["segment"][:]
    rec = int((segs == 3).sum())
    cor = int((segs == 4).sum())
    if cc is not None:
        ok_rec, msg = _verify_recovery_segments(shard, cc)
        if not ok_rec:
            return False, msg
    if rec + cor > 0:
        if cor > 0 and rec == 0:
            return False, "corrections without recovery segments"
        ratio = rec / max(cor, 1)
        # RaC correction rollouts are much longer than short reversal recovery bridges;
        # only flag pathological over-recovery (not a low global ratio).
        if ratio > 2.0:
            return False, f"recovery:correction ratio {ratio:.2f} > 2.0"
    if meta is not None:
        attempts = max(int(meta.get("reroute_attempts", 0)), 1)
        fb = int(meta.get("fallback_accepts", 0))
        rate = fb / attempts
        if rate >= 0.8:
            return False, f"fallback acceptance rate {rate:.2f} >= 0.8"
    return True, "ok"


def gate_regress(curr: float, prev: float, tol: float = 0.10) -> tuple[bool, str]:
    if curr >= prev - tol:
        return True, "ok"
    return False, f"regression {prev:.3f} -> {curr:.3f}"


def gate_regress_abort(drops: list[float], tol: float = 0.20) -> tuple[bool, str]:
    if len(drops) < 2:
        return True, "ok"
    if drops[-1] > tol and drops[-2] > tol:
        return False, "two consecutive regressions > 20 pts"
    return True, "ok"


def gate_dither(median_switches: float, limit: float = 10.0) -> tuple[bool, str]:
    if median_switches <= limit:
        return True, "ok"
    return False, f"median mode switches {median_switches:.1f} > {limit}"


def gate_ppo(
    *,
    ci_excludes_zero: bool,
    validity_frac: float,
    before_deadline: bool,
    head_compatible: bool = True,
) -> tuple[bool, str]:
    if not head_compatible:
        return False, "head_incompatible"
    if not before_deadline:
        return False, "deadline_passed"
    if validity_frac < 0.60:
        return False, f"validity gate {validity_frac:.2f} < 0.60"
    if not ci_excludes_zero:
        return False, "val_unseen CI includes zero"
    return True, "go"


def gate_beta(r_div_std: float, success_bonus: float = 10.0) -> bool:
    return r_div_std <= 2 * success_bonus


def eval_success_on_scenes(
    ckpt: Path,
    scenes: list[SceneSpec],
    policy_cfg: PolicyConfig,
    eval_cfg: EvalConfig,
    div_cfg: DiversityConfig,
    delta: float,
    routes: bool = False,
) -> dict[str, Any]:
    return evaluate_checkpoint(
        ckpt,
        scenes,
        policy_cfg,
        eval_cfg,
        div_cfg,
        delta=delta,
        include_routes=routes,
    )


def load_eval_success(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text()).get("success_rate", 0.0))
