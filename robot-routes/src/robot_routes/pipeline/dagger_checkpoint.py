"""Mid-round DAgger checkpointing (crash-safe collection resume)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robot_routes.contracts import Transition
from robot_routes.data.schema import read_shard, write_shard

CHECKPOINT_EVERY = 500


def partial_paths(out: Path, round_k: int) -> tuple[Path, Path]:
    return out / f"round_{round_k}_partial.h5", out / f"round_{round_k}_progress.json"


def save_dagger_partial(
    out: Path,
    round_k: int,
    rows: list[Transition],
    episode_scenes: dict[int, str],
    budget: int,
    *,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> None:
    """Flush in-memory collection progress to disk."""
    partial, progress = partial_paths(out, round_k)
    max_ep = max(episode_scenes.keys()) if episode_scenes else -1
    scenes = [episode_scenes.get(i, "") for i in range(max_ep + 1)]
    write_shard(partial, rows, scenes)
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "round": round_k,
        "budget": budget,
        "transitions": len(rows),
        "episodes": len(scenes),
        "checkpoint_every": checkpoint_every,
    }
    progress.write_text(json.dumps(payload, indent=2))


def load_dagger_partial(
    out: Path, round_k: int
) -> tuple[list[Transition], dict[int, str], int]:
    """Restore partial round collection, or empty if none."""
    partial, progress = partial_paths(out, round_k)
    if not partial.exists():
        return [], {}, 0
    try:
        rows, scenes = read_shard(partial)
    except (OSError, KeyError, ValueError):
        return [], {}, 0
    episode_scenes = {i: s for i, s in enumerate(scenes)}
    budget = len(rows)
    if progress.exists():
        try:
            meta = json.loads(progress.read_text())
            budget = int(meta.get("budget", budget))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return rows, episode_scenes, budget


def clear_dagger_partial(out: Path, round_k: int) -> None:
    partial, progress = partial_paths(out, round_k)
    partial.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)


def partial_budget(out: Path, round_k: int) -> int:
    _, _, budget = load_dagger_partial(out, round_k)
    return budget
