"""NVML-based GPU discovery, scoring, leasing (§10.5.2, §15.7.9)."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pynvml as _pynvml

    pynvml = _pynvml
    _HAS_NVML = True
except ImportError:
    pynvml = None  # type: ignore[assignment,misc]
    _HAS_NVML = False


@dataclass(frozen=True)
class GpuLease:
    physical_id: int
    lock_path: Path

    def env(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": str(self.physical_id)}

    def release(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()


class GpuAllocator:
    def __init__(
        self,
        lock_dir: Path | None = None,
        jobs_per_gpu: int = 2,
        util_w: float = 0.5,
        mem_w: float = 0.5,
    ) -> None:
        self.lock_dir = Path(lock_dir or Path.home() / ".robot_routes" / "gpu_locks")
        self.jobs_per_gpu = jobs_per_gpu
        self.util_w = util_w
        self.mem_w = mem_w
        self._nvml = False
        if _HAS_NVML:
            try:
                assert pynvml is not None
                pynvml.nvmlInit()
                self._nvml = True
            except Exception:
                self._nvml = False
        self._check_lock_dir()

    def _check_lock_dir(self) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        st = os.statvfs(self.lock_dir)
        # NFS often has type 0x6969 or f_type check; simple heuristic: remote if not local
        # Refuse NFS for flock reliability (§11.7.4)
        if hasattr(st, "f_type") and st.f_type == 0x6969:
            raise RuntimeError(f"GPU lock dir must not be on NFS: {self.lock_dir}")

    def _candidates(self) -> list[int]:
        if not self._nvml:
            return []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip():
            return [int(x) for x in visible.split(",") if x.strip()]
        assert pynvml is not None
        return list(range(pynvml.nvmlDeviceGetCount()))

    def _reap_stale(self) -> None:
        for gpu_dir in self.lock_dir.glob("gpu*"):
            for lease in gpu_dir.glob("*.json"):
                try:
                    meta = json.loads(lease.read_text())
                    os.kill(int(meta["pid"]), 0)
                except (ProcessLookupError, OSError, KeyError, ValueError):
                    lease.unlink(missing_ok=True)

    def acquire(
        self,
        mem_required_gb: float = 2.0,
        exclusive: bool = False,
        timeout_s: float | None = None,
    ) -> GpuLease | None:
        if not self._nvml:
            return None
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        dir_lock = self.lock_dir / "dir.lock"
        while True:
            with open(dir_lock, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                self._reap_stale()
                scores: list[tuple[float, int]] = []
                for gid in self._candidates():
                    assert pynvml is not None
                    h = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    utils = []
                    for _ in range(3):
                        utils.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                        time.sleep(0.2)
                    util = float(np.mean(utils))
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpu_dir = self.lock_dir / f"gpu{gid}"
                    leases = len(list(gpu_dir.glob("*.json"))) if gpu_dir.exists() else 0
                    free_gb = mem.free / 2**30
                    if free_gb < mem_required_gb + 0.5:
                        continue
                    if exclusive and leases > 0:
                        continue
                    if not exclusive and leases >= self.jobs_per_gpu:
                        continue
                    score = self.util_w * util / 100 + self.mem_w * (1 - mem.free / mem.total)
                    scores.append((score, gid))
                if scores:
                    gid = min(scores)[1]
                    gpu_dir = self.lock_dir / f"gpu{gid}"
                    gpu_dir.mkdir(exist_ok=True)
                    lease_path = gpu_dir / f"{os.getpid()}.json"
                    lease_path.write_text(
                        json.dumps(
                            {"pid": os.getpid(), "mem_gb": mem_required_gb, "ts": time.time()}
                        )
                    )
                    return GpuLease(physical_id=gid, lock_path=lease_path)
            if deadline and time.monotonic() > deadline:
                return None
            time.sleep(10)

    def shutdown(self) -> None:
        if self._nvml and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
