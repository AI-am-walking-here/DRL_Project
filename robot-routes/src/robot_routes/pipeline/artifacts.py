"""Run artifact helpers: config dumps, git hash (§11)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def git_hash(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def dump_resolved_config(run_dir: Path, name: str, cfg: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"resolved_{name}.yaml"
    path.write_text(yaml.dump(cfg, default_flow_style=False))
    return path


def write_run_meta(run_dir: Path, root: Path, extra: dict[str, Any] | None = None) -> None:
    meta = {"git_hash": git_hash(root), **(extra or {})}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
