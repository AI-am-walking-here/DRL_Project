"""YAML config loading into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(f.type, val)
        elif getattr(f.type, "__origin__", None) is list and val:
            inner = f.type.__args__[0]
            if is_dataclass(inner):
                kwargs[f.name] = [_from_dict(inner, v) for v in val]
            else:
                kwargs[f.name] = val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: Path | str, cls: type[T]) -> T:
    return _from_dict(cls, load_yaml(path))


@dataclass
class EnvConfig:
    ctrl_hz: float = 20.0
    n_substeps: int = 25
    horizon: int = 300
    action_clip: float = 0.05
    success_tol_m: float = 0.05
    success_hold_steps: int = 5
    jlimit_margin_rad: float = 0.02
    max_obstacles: int = 8
    obs_dim: int = 79
    dt: float = 0.002
    contact_exclusions: list[tuple[str, str]] | None = None
    seed_ranges: dict[str, list[int]] | None = None


@dataclass
class SceneConfig:
    r_min: float = 0.25
    r_max: float = 0.75
    z_min: float = 0.05
    z_max: float = 0.7
    box_half_min: float = 0.03
    box_half_max: float = 0.10
    sph_r_min: float = 0.04
    sph_r_max: float = 0.10
    goal_min_dist: float = 0.35
    start_noise_rad: float = 0.05
    unseen_half_min: float = 0.08
    unseen_half_max: float = 0.14


@dataclass
class ExpertConfig:
    step_size: float = 0.15
    goal_bias: float = 0.10
    max_iters: int = 20000
    t_validate_s: float = 10.0
    t_label_s: float = 3.0
    margin_plan_m: float = 0.03
    edge_check_rad: float = 0.03
    ik_restarts: int = 16
    ik_dedup_rad: float = 0.2
    shortcut_iters: int = 100
    waypoint_rad: float = 0.04
    lookahead: int = 3


@dataclass
class PolicyConfig:
    trunk: list[int] | None = None
    k_mix: int = 5
    logstd_clamp: list[float] | None = None
    entropy_bonus: float = 1e-3
    entropy_bonus_epochs: int = 20
    sigma_c: float = 0.02
    dither_escalate_median: int = 10
    chunk_k: int = 4
    head: str = "mdn"


@dataclass
class BCConfig:
    n_demos: int = 2000
    lr: float = 3e-4
    batch: int = 1024
    epochs: int = 200
    grad_clip: float = 1.0
    val_frac: float = 0.05
    gate_soft: float = 0.25
    gate_target: float = 0.40
    replay_min_rate: float = 0.70
    replay_max_episodes: int = 20


@dataclass
class DaggerRacConfig:
    rounds: int = 6
    budget: int = 40000
    eps_danger_m: float = 0.02
    eps_safe_m: float = 0.10
    ring_buffer: int = 40
    stuck_steps: int = 60
    stuck_eps_m: float = 0.01
    replan_drift_rad: float = 0.25
    settle_steps: int = 5
    retrain_epochs: int = 100
    reroute_attempts: int = 5
    rac_enabled: bool = True
    reroute_enabled: bool = True
    weights: dict[str, float] | None = None


@dataclass
class CurriculumConfig:
    levels: list[list[int]] | None = None
    promote: float = 0.70
    demote: float = 0.30
    offlevel_cap: float = 0.5
    synth_frac: float = 0.2
    synthetic_obs: bool = True


@dataclass
class PPOConfig:
    clip: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    lr: float = 1e-4
    n_envs: int = 8
    kl_anchor: float = 0.5
    entropy_coef: float = 0.005
    pool_scenes: int = 256
    archive_m: int = 16
    beta: float = 2.0
    rdiv_cap: float = 0.5
    beta_halvings_max: int = 3
    total_steps: int = 1_000_000


@dataclass
class DiversityConfig:
    resample_pts: int = 64
    n_rollouts: int = 20
    validity_min: int = 8
    rollout_seeds: str = "0..19"
    recovery_merge_steps: int = 10


@dataclass
class EvalConfig:
    val_per_level: int = 100
    test_per_level: int = 200
    routes_scenes: int = 50
    bootstrap_n: int = 10000
    seeds: list[int] | None = None
    mde_pts: float = 10.0
    unseen_half_min: float = 0.08
    unseen_half_max: float = 0.14
    unseen_reject_max: float = 0.5


@dataclass
class ComputeConfig:
    jobs_per_gpu: int = 2
    mem_required_gb: float = 2.0
    util_w: float = 0.5
    mem_w: float = 0.5
    nvml_samples: int = 3
    nvml_interval_ms: int = 200
    watchdog_min: int = 30
    dep_timeout_h: int = 48
    disk_min_gb: int = 50
    device: str = "auto"
