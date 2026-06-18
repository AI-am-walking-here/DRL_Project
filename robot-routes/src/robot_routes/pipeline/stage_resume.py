"""Stage-level resume detection for long-running pipeline scripts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py


def count_demos_h5(path: Path) -> int:
    """Episode count from merged or shard HDF5."""
    if not path.exists():
        return 0
    with h5py.File(path, "r") as f:
        if "episodes" in f and "scene_json" in f["episodes"]:
            return int(len(f["episodes"]["scene_json"]))
        if "episode_id" in f and len(f["episode_id"]) > 0:
            return int(f["episode_id"][:].max()) + 1
    return 0


def artifact_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


@dataclass(frozen=True)
class CollectResumeState:
    done: bool
    pending_workers: list[int]
    merge_only: bool
    resumed: bool


def detect_collect_resume(
    out: Path,
    n_demos: int,
    n_workers: int,
    *,
    force_restart: bool = False,
) -> CollectResumeState:
    merged = out / "demos.h5"
    if force_restart:
        for p in out.glob("demos_w*.h5"):
            p.unlink(missing_ok=True)
        merged.unlink(missing_ok=True)
        return CollectResumeState(
            done=False,
            pending_workers=list(range(n_workers)),
            merge_only=False,
            resumed=False,
        )

    if merged.exists() and count_demos_h5(merged) >= n_demos:
        return CollectResumeState(
            done=True,
            pending_workers=[],
            merge_only=False,
            resumed=True,
        )

    pending = [i for i in range(n_workers) if not (out / f"demos_w{i}.h5").exists()]
    if not pending:
        return CollectResumeState(
            done=False,
            pending_workers=[],
            merge_only=True,
            resumed=any((out / f"demos_w{i}.h5").exists() for i in range(n_workers)),
        )
    return CollectResumeState(
        done=False,
        pending_workers=pending,
        merge_only=False,
        resumed=any((out / f"demos_w{i}.h5").exists() for i in range(n_workers)),
    )


@dataclass(frozen=True)
class CurriculumResumeState:
    start_step: int
    total_steps: int
    policy_ckpt: Path
    merged_base: Path
    level: int
    history: list[dict]
    resumed: bool
    completed_through: int | None


def _curriculum_ckpt(out: Path, step: int) -> Path:
    return out / f"ckpt_{step}" / "best.pt"


def _curriculum_merged(out: Path, step: int) -> Path:
    return out / f"merged_cur_{step}.h5"


def _clear_curriculum_step(out: Path, step: int) -> None:
    for p in [
        out / f"curriculum_{step}.h5",
        _curriculum_merged(out, step),
        out / f"ckpt_{step}",
    ]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def detect_curriculum_resume(
    out: Path,
    dagger_out: Path,
    dagger_rounds: int,
    bc_data: Path,
    total_steps: int,
    *,
    force_restart: bool = False,
) -> CurriculumResumeState:
    dagger_ckpt = dagger_out / "best.pt"
    if not dagger_ckpt.exists():
        dagger_ckpt = dagger_out / "ckpt_0" / "best.pt"
    dagger_merged = dagger_out / f"merged_{dagger_rounds - 1}.h5"
    if not dagger_merged.exists():
        dagger_merged = bc_data

    fresh = CurriculumResumeState(
        start_step=0,
        total_steps=total_steps,
        policy_ckpt=dagger_ckpt,
        merged_base=dagger_merged,
        level=0,
        history=[],
        resumed=False,
        completed_through=None,
    )
    if force_restart or total_steps <= 0:
        return fresh

    state_path = out / "curriculum_state.json"
    saved_state: dict = {}
    if state_path.exists():
        try:
            saved_state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            saved_state = {}

    last_complete = -1
    for step in range(total_steps):
        if _curriculum_ckpt(out, step).exists() and _curriculum_merged(out, step).exists():
            last_complete = step

    if last_complete >= total_steps - 1:
        return CurriculumResumeState(
            start_step=total_steps,
            total_steps=total_steps,
            policy_ckpt=_curriculum_ckpt(out, total_steps - 1),
            merged_base=_curriculum_merged(out, total_steps - 1),
            level=int(saved_state.get("level", 0)),
            history=list(saved_state.get("history", [])),
            resumed=True,
            completed_through=total_steps - 1,
        )

    start = last_complete + 1
    for step in range(last_complete + 1, total_steps):
        has_shard = (out / f"curriculum_{step}.h5").exists()
        has_ckpt = _curriculum_ckpt(out, step).exists()
        if has_ckpt:
            last_complete = step
            start = step + 1
            continue
        if has_shard:
            _clear_curriculum_step(out, step)
        start = step
        break

    if start <= 0:
        return fresh

    if not _curriculum_ckpt(out, start - 1).exists():
        return fresh

    # Truncate history to completed steps
    history = list(saved_state.get("history", []))[:start]
    level = int(saved_state.get("level", 0))
    if history:
        level = int(history[-1].get("level", level))

    return CurriculumResumeState(
        start_step=start,
        total_steps=total_steps,
        policy_ckpt=_curriculum_ckpt(out, start - 1),
        merged_base=_curriculum_merged(out, start - 1),
        level=level,
        history=history,
        resumed=True,
        completed_through=start - 1,
    )


def save_curriculum_state(out: Path, level: int, synthetic_obs: bool, history: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "curriculum_state.json").write_text(
        json.dumps(
            {"level": level, "synthetic_obs": synthetic_obs, "history": history},
            indent=2,
        )
    )


def eval_artifact_valid(
    path: Path,
    ckpt: Path,
    *,
    meta: dict | None = None,
) -> bool:
    """True when an eval JSON exists and matches the checkpoint (and optional meta)."""
    if not artifact_complete(path):
        return False
    try:
        data = json.loads(path.read_text())
        saved = data.get("_resume_meta")
        if not isinstance(saved, dict):
            return False
        if Path(str(saved.get("ckpt", ""))).resolve() != ckpt.resolve():
            return False
        if meta:
            for key, value in meta.items():
                if saved.get(key) != value:
                    return False
        return True
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def eval_resume_meta(ckpt: Path, **extra: object) -> dict:
    return {"ckpt": str(ckpt.resolve()), **extra}
