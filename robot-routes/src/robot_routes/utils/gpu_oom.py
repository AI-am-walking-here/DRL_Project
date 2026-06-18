"""CUDA OOM detection, training backoff, and grid relaunch hints (§10.5.2)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OOM_BACKOFF_FILE = "oom_backoff.json"
DEFAULT_MIN_BATCH = 256
DEFAULT_GRID_OOM_RETRIES = 1


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def cuda_alloc_env() -> dict[str, str]:
    """Reduce fragmentation on long runs (safe default for grid children)."""
    cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in cur:
        return {}
    extra = "expandable_segments:True"
    merged = f"{cur},{extra}" if cur else extra
    return {"PYTORCH_CUDA_ALLOC_CONF": merged}


def write_oom_backoff(
    run_dir: Path,
    *,
    stage: str,
    detail: str = "",
    request_exclusive: bool = True,
) -> Path:
    """Signal grid launcher to retry this run once with an exclusive GPU lease."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / OOM_BACKOFF_FILE
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "detail": detail[:500],
        "request_exclusive": request_exclusive,
        "consumed": False,
        "retries": int(json.loads(path.read_text()).get("retries", 0)) + 1
        if path.exists()
        else 1,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_oom_backoff(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / OOM_BACKOFF_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def consume_oom_backoff(run_dir: Path) -> bool:
    """True if an unconsumed backoff request exists (grid should retry exclusive)."""
    data = read_oom_backoff(run_dir)
    if not data or data.get("consumed"):
        return False
    data["consumed"] = True
    (run_dir / OOM_BACKOFF_FILE).write_text(json.dumps(data, indent=2))
    return bool(data.get("request_exclusive", True))


def grid_oom_retries_left(run_dir: Path, max_retries: int = DEFAULT_GRID_OOM_RETRIES) -> bool:
    data = read_oom_backoff(run_dir)
    if not data:
        return False
    return int(data.get("retries", 0)) <= max_retries and not data.get("consumed")
