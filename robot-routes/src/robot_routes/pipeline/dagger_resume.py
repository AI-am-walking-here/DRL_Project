"""DAgger round-level resume detection (§6)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DaggerResumeState:
    start_round: int
    policy_ckpt: Path
    merged_base: Path
    round_stats: list[float]
    prev_sr: float
    regress_drops: list[float]
    resumed: bool
    completed_through: int | None  # last fully completed round index, else None
    reuse_shard: bool = False  # round_{k}.h5 exists; validate G-DATA before skipping collect


def _ckpt_path(out: Path, k: int) -> Path:
    return out / f"ckpt_{k}" / "best.pt"


def _merged_path(out: Path, k: int) -> Path:
    return out / f"merged_{k}.h5"


def _clear_round_artifacts(out: Path, k: int) -> None:
    """Remove partial round outputs so collection/retrain restarts cleanly."""
    for p in [
        out / f"round_{k}.h5",
        out / f"round_{k}_meta.json",
        out / f"delta_d_{k}.h5",
        _merged_path(out, k),
        out / f"ckpt_{k}",
    ]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def _load_round_stats(out: Path) -> list[float]:
    path = out / "round_stats.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [float(x) for x in data]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def _regress_drops_from_stats(stats: list[float]) -> list[float]:
    drops: list[float] = []
    for i in range(1, len(stats)):
        drops.append(stats[i - 1] - stats[i] if stats[i] < stats[i - 1] else 0.0)
    return drops


def detect_dagger_resume(
    out: Path,
    bc_data: Path,
    bc_ckpt: Path,
    total_rounds: int,
    *,
    force_restart: bool = False,
) -> DaggerResumeState:
    """Pick start round and checkpoint/data paths from on-disk artifacts."""
    fresh = DaggerResumeState(
        start_round=0,
        policy_ckpt=bc_ckpt,
        merged_base=bc_data,
        round_stats=[],
        prev_sr=0.0,
        regress_drops=[],
        resumed=False,
        completed_through=None,
        reuse_shard=False,
    )
    if force_restart or total_rounds <= 0:
        return fresh

    last_complete = -1
    for k in range(total_rounds):
        if _ckpt_path(out, k).exists():
            last_complete = k

    if last_complete >= total_rounds - 1:
        stats = _load_round_stats(out)
        return DaggerResumeState(
            start_round=total_rounds,
            policy_ckpt=_ckpt_path(out, total_rounds - 1),
            merged_base=_merged_path(out, total_rounds - 1),
            round_stats=stats[:total_rounds],
            prev_sr=stats[-1] if stats else 0.0,
            regress_drops=_regress_drops_from_stats(stats[:total_rounds]),
            resumed=True,
            completed_through=total_rounds - 1,
            reuse_shard=False,
        )

    start = last_complete + 1
    reuse_shard = False
    for k in range(last_complete + 1, total_rounds):
        has_shard = (out / f"round_{k}.h5").exists()
        has_ckpt = _ckpt_path(out, k).exists()
        if has_ckpt:
            last_complete = k
            start = k + 1
            continue
        if has_shard:
            reuse_shard = True
        start = k
        break

    if start <= 0 and not reuse_shard:
        return fresh

    if start == 0 and reuse_shard:
        return DaggerResumeState(
            start_round=0,
            policy_ckpt=bc_ckpt,
            merged_base=bc_data,
            round_stats=_load_round_stats(out),
            prev_sr=0.0,
            regress_drops=[],
            resumed=True,
            completed_through=None,
            reuse_shard=True,
        )

    if not _ckpt_path(out, start - 1).exists():
        return fresh

    stats = _load_round_stats(out)[:start]
    prev_sr = stats[-1] if stats else 0.0
    return DaggerResumeState(
        start_round=start,
        policy_ckpt=_ckpt_path(out, start - 1),
        merged_base=_merged_path(out, start - 1),
        round_stats=list(stats),
        prev_sr=prev_sr,
        regress_drops=_regress_drops_from_stats(stats),
        resumed=True,
        completed_through=start - 1,
        reuse_shard=reuse_shard,
    )


def save_round_stats(out: Path, round_stats: list[float]) -> None:
    """Persist partial progress after each round (crash-safe)."""
    (out / "round_stats.json").write_text(json.dumps(round_stats))
