"""Safe merge for pipeline_state.json (§11.7.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}, "events": [], "dagger_rounds": []}
    return json.loads(path.read_text())


def merge_pipeline_state(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Read-modify-write without dropping unrelated keys."""
    data = load_state(path)
    for key, val in patch.items():
        if key == "events" and isinstance(val, list):
            seen = {e.get("ts", "") + e.get("event", "") for e in data.get("events", [])}
            for e in val:
                sig = e.get("ts", "") + e.get("event", "")
                if sig not in seen:
                    data.setdefault("events", []).append(e)
                    seen.add(sig)
        elif key == "stages" and isinstance(val, dict):
            data.setdefault("stages", {}).update(val)
        else:
            data[key] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def reconcile_from_stamps(run_dir: Path, stages: list[str]) -> None:
    """Repair stale WAITING_DEP / missing terminal stages when stamp files exist."""
    path = run_dir / "pipeline_state.json"
    data = load_state(path)
    changed = False
    if "ppo" in stages:
        ppo_i = stages.index("ppo")
        downstream = stages[ppo_i + 1 :]
        if any((run_dir / f"{s}.stamp").exists() for s in downstream):
            entry = data.setdefault("stages", {}).setdefault("ppo", {})
            if entry.get("status") in ("WAITING_DEP", "PENDING", "RUNNING", None):
                entry["status"] = "SKIPPED"
                entry.setdefault("reason", "reconciled_from_downstream_stamps")
                changed = True
    for stage in stages:
        stamp = run_dir / f"{stage}.stamp"
        if not stamp.exists():
            continue
        entry = data.setdefault("stages", {}).setdefault(stage, {})
        if entry.get("status") in ("RUNNING", "PENDING", "WAITING_DEP", None) and stage != "ppo":
            entry["status"] = "COMPLETED"
            changed = True
    if (run_dir / "verdicts.stamp").exists():
        pl = data.setdefault("stages", {}).setdefault("pipeline", {})
        if pl.get("status") != "COMPLETED":
            pl["status"] = "COMPLETED"
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2))
