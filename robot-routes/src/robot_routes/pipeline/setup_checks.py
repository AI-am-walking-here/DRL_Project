"""Setup preflight checks (§11.7.3–4)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

STATFS_NFS = 0x6969


def is_nfs(path: Path) -> bool:
    try:
        st = os.statvfs(path)
        f_type = getattr(st, "f_type", None)
        if f_type is None:
            return False
        return f_type == STATFS_NFS
    except OSError:
        return False


def check_local_filesystem(path: Path) -> None:
    if is_nfs(path.resolve()):
        raise RuntimeError(f"run directory {path} is on NFS — HDF5/flock unreliable (§11.7.4)")


def check_nvidia_driver(min_major: int = 525) -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ver = out.strip().split("\n")[0].split(".")[0]
        if int(ver) < min_major:
            raise RuntimeError(f"NVIDIA driver {out.strip()} < minimum {min_major} (§11.7.4)")
    except FileNotFoundError:
        pass
    except ValueError as e:
        raise RuntimeError(f"could not parse nvidia-smi driver version: {e}") from e


def require_prereg_tag(root: Path, tag: str = "prereg-v1") -> str:
    try:
        out = subprocess.check_output(
            ["git", "tag", "-l", tag],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""
    if not out:
        raise RuntimeError(
            f"git tag {tag!r} required before full-scale runs (§11.7.4); "
            "create tag or set PIPELINE_SKIP_PREREG=1 for smoke/dev"
        )
    return out


def assert_delta_invariant(root: Path, delta: float, delta_sha: str) -> None:
    dagger = root / "configs/train/dagger_rac.yaml"
    if not dagger.exists():
        return
    import yaml

    data = yaml.safe_load(dagger.read_text())
    reroute = float(data.get("delta_reroute_m", delta))
    if abs(reroute - delta) > 1e-6:
        raise RuntimeError(f"δ_reroute ({reroute}) != calibrated δ_distinct ({delta}) (§11.7.3)")
    cal = root / "calibration/delta.json"
    if cal.exists() and delta_sha:
        payload = json.loads(cal.read_text())
        file_sha = payload.get("sha256", "")
        if file_sha and file_sha != delta_sha:
            raise RuntimeError("calibration hash mismatch across grid (§11.7.3)")


def run_setup_checks(
    root: Path,
    run_dir: Path,
    *,
    profile: str,
    skip_prereg: bool = False,
) -> dict[str, Any]:
    check_local_filesystem(run_dir)
    check_nvidia_driver()
    meta: dict[str, Any] = {}
    if profile != "smoke" and not skip_prereg and not os.environ.get("PIPELINE_SKIP_PREREG"):
        meta["prereg_tag"] = require_prereg_tag(root)
    elif os.environ.get("PIPELINE_SKIP_PREREG"):
        meta["prereg_tag"] = "skipped"
    return meta
