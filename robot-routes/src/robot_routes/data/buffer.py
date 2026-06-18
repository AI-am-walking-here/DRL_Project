"""Aggregated dataset buffer with weighted sampling (§6.2)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from robot_routes.contracts import SEGMENT_NAME


class TransitionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, h5_path: Path, val_frac: float = 0.05, rng: np.random.Generator | None = None
    ) -> None:
        self.path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.obs = f["obs"][:]
            self.action = f["action"][:]
            self.segment = f["segment"][:]
            self.episode_id = f["episode_id"][:]
        rng = rng or np.random.default_rng(0)
        episodes = np.unique(self.episode_id)
        rng.shuffle(episodes)
        n_val = max(1, int(len(episodes) * val_frac))
        val_eps = set(episodes[:n_val])
        self.val_mask = np.array([e in val_eps for e in self.episode_id])
        self.train_mask = ~self.val_mask

    def __len__(self) -> int:
        return int(self.train_mask.sum())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        train_idx = np.where(self.train_mask)[0][idx]
        return (
            torch.as_tensor(self.obs[train_idx], dtype=torch.float32),
            torch.as_tensor(self.action[train_idx], dtype=torch.float32),
        )

    def val_loader_data(self) -> tuple[np.ndarray, np.ndarray]:
        vi = np.where(self.val_mask)[0]
        return self.obs[vi], self.action[vi]

    def segment_weights(self, weights: dict[str, float]) -> np.ndarray:
        w = np.ones(len(self.obs))
        for i, seg in enumerate(self.segment):
            if self.train_mask[i]:
                w[i] = weights.get(SEGMENT_NAME[int(seg)], 1.0)
            else:
                w[i] = 0.0
        return w
