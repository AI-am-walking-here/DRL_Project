"""Project Python + PyTorch device resolution (§10.5.4)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Collection / MuJoCo rollouts keep policy inference on CPU (§10.5.4).
COLLECT_DEVICE = torch.device("cpu")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_python(root: Path | None = None) -> str:
    """Prefer repo .venv when present; otherwise the active interpreter."""
    root = root or project_root()
    venv_py = root / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def resolve_device(name: str = "auto") -> torch.device:
    """CUDA when available; CPU fallback. Honors CUDA_VISIBLE_DEVICES masking."""
    if name == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(name)
    return torch.device("cpu")
