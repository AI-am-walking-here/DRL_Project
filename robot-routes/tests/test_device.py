"""Device resolution tests (§10.5.4, §10.5.6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from robot_routes.utils.device import COLLECT_DEVICE, project_python, resolve_device


def test_collect_device_is_cpu():
    assert COLLECT_DEVICE.type == "cpu"


def test_resolve_device_auto_cpu_when_cuda_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto").type == "cpu"


def test_resolve_device_auto_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto").type == "cuda"


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_project_python_prefers_venv(tmp_path: Path):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    assert project_python(tmp_path) == str(py)
