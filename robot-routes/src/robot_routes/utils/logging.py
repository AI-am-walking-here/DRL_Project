"""Optional wandb/tensorboard logging wrapper."""

from __future__ import annotations

import json
from pathlib import Path


def log_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(metrics) + "\n")
