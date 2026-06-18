"""Route diversity metrics (§9, §15.5.1, §15.7.8)."""

from __future__ import annotations

import numpy as np


def frechet(p: np.ndarray, q: np.ndarray) -> float:
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    ca = np.full(d.shape, np.inf)
    ca[0, 0] = d[0, 0]
    for i in range(1, len(p)):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, len(q)):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, len(p)):
        for j in range(1, len(q)):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[-1, -1])


def resample_path(path: np.ndarray, n: int = 64) -> np.ndarray:
    if len(path) < 2:
        return np.tile(path[0], (n, 1)) if len(path) else np.zeros((n, 3))
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(path[0], (n, 1))
    targets = np.linspace(0, total, n)
    out = np.zeros((n, 3))
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(j, len(path) - 2)
        alpha = (t - cum[j]) / (cum[j + 1] - cum[j] + 1e-12)
        out[i] = (1 - alpha) * path[j] + alpha * path[j + 1]
    return out


def count_routes(trajs: list[np.ndarray], delta: float) -> int:
    n = len(trajs)
    if n == 0:
        return 0
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = frechet(trajs[i], trajs[j])
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, bi, bj = np.inf, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = max(D[a, b] for a in clusters[i] for b in clusters[j])
                if d < best:
                    best, bi, bj = d, i, j
        if best >= delta:
            break
        clusters[bi] += clusters.pop(bj)
    return len(clusters)


def route_entropy(cluster_ids: list[int]) -> float:
    if not cluster_ids:
        return 0.0
    _, counts = np.unique(cluster_ids, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    diffs = a - b
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    samples = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def calibrate_delta(same: list[float], cross: list[float]) -> float:
    return float((np.percentile(same, 95) + np.percentile(cross, 5)) / 2)
