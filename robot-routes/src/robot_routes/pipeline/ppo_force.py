"""Force-run PPO (class deliverable) even when G-PPO would skip."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PPO_DOWNSTREAM = ("evaluate_test", "verdicts", "report_assets")


def ppo_checkpoint(run_dir: Path) -> Path:
    return run_dir / "ppo" / "ppo.pt"


def ppo_trained(run_dir: Path) -> bool:
    ckpt = ppo_checkpoint(run_dir)
    return ckpt.is_file() and ckpt.stat().st_size > 0


def load_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pipeline_state.json"
    if not path.exists():
        return {"stages": {}, "events": [], "dagger_rounds": []}
    return json.loads(path.read_text())


def ppo_stage_done(run_dir: Path, state: dict[str, Any] | None = None) -> bool:
    state = state if state is not None else load_run_state(run_dir)
    entry = state.get("stages", {}).get("ppo", {})
    return entry.get("status") == "COMPLETED" and ppo_trained(run_dir)


def needs_ppo_force_rerun(
    run_dir: Path,
    *,
    ppo_force: bool,
    has_ppo_stage: bool,
) -> bool:
    """True when profile forces PPO but this run never produced a PPO checkpoint."""
    if not ppo_force or not has_ppo_stage:
        return False
    return not ppo_stage_done(run_dir)


def grid_job_complete(
    run_dir: Path,
    *,
    ppo_force: bool,
    has_ppo_stage: bool,
) -> bool:
    """Grid skip predicate: COMPLETED runs with missing forced PPO are not done."""
    if needs_ppo_force_rerun(run_dir, ppo_force=ppo_force, has_ppo_stage=has_ppo_stage):
        return False
    if (run_dir / "COMPLETED").exists():
        return True
    state = load_run_state(run_dir)
    return state.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED"


def arm_ppo_force(run_dir: Path) -> bool:
    """Clear a prior G-PPO skip so Stage 4 can run and downstream evals refresh."""
    path = run_dir / "pipeline_state.json"
    if not path.exists():
        return False
    data = load_run_state(run_dir)
    if ppo_stage_done(run_dir, data):
        return False

    prev = data.get("stages", {}).get("ppo", {}).get("status", "PENDING")
    ts = datetime.now(timezone.utc).isoformat()
    data.setdefault("stages", {}).setdefault("ppo", {})["status"] = "PENDING"
    data["stages"]["ppo"]["updated"] = ts
    data["stages"]["ppo"].pop("reason", None)

    for stage in _PPO_DOWNSTREAM:
        if stage in data.get("stages", {}):
            data["stages"][stage]["status"] = "PENDING"
            data["stages"][stage]["updated"] = ts
    pl = data.setdefault("stages", {}).setdefault("pipeline", {})
    pl["status"] = "PENDING"
    pl["updated"] = ts

    data.setdefault("events", []).append(
        {
            "ts": ts,
            "event": "ppo_force_armed",
            "previous_ppo_status": prev,
        }
    )
    path.write_text(json.dumps(data, indent=2))

    (run_dir / "ppo.stamp").unlink(missing_ok=True)
    for stage in _PPO_DOWNSTREAM:
        (run_dir / f"{stage}.stamp").unlink(missing_ok=True)
    (run_dir / "pipeline.stamp").unlink(missing_ok=True)
    (run_dir / "COMPLETED").unlink(missing_ok=True)
    return True
