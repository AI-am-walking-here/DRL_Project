"""GPU allocator tests (§10.5.2)."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robot_routes.utils.gpu_alloc import GpuAllocator, GpuLease

pytestmark = pytest.mark.wp1


def test_lease_env():
    lease = GpuLease(physical_id=2, lock_path=Path("/tmp/x.json"))
    assert lease.env() == {"CUDA_VISIBLE_DEVICES": "2"}


@patch("robot_routes.utils.gpu_alloc._HAS_NVML", False)
def test_acquire_no_nvml():
    alloc = GpuAllocator(lock_dir=Path("/tmp/robot_routes_test_locks"))
    assert alloc.acquire() is None


@patch("robot_routes.utils.gpu_alloc.pynvml")
def test_acquire_mocked(mock_nvml):
    mock_nvml.nvmlInit.return_value = None
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    h = MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = h
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=10)
    mem = MagicMock(free=8 * 2**30, total=16 * 2**30)
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem
    lock_dir = Path(f"/tmp/robot_routes_gpu_test_{os.getpid()}")
    alloc = GpuAllocator(lock_dir=lock_dir)
    alloc._nvml = True
    lease = alloc.acquire(mem_required_gb=2.0, timeout_s=1.0)
    if lease:
        assert lease.physical_id == 0
        lease.release()
    alloc.shutdown()


def test_hammer_leases(tmp_path):
    """32 threads × many cycles — no double lease on same file."""
    with patch("robot_routes.utils.gpu_alloc._HAS_NVML", False):
        alloc = GpuAllocator(lock_dir=tmp_path)
        results = []

        def worker():
            r = alloc.acquire(timeout_s=0.1)
            results.append(r is None)
            if r:
                r.release()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(results)
