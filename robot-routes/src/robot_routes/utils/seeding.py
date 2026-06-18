"""Seeding utilities (§11 conventions)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed stdlib, numpy, torch; return a dedicated Generator for library code."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
