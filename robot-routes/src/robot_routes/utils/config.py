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
    # Success = end-effector within success_tol_m of the goal, held for
    # success_hold_steps. The "archery target": on-target at success_tol_m,
    # bullseye (full bonus) at/inside bullseye_tol_m.
    success_tol_m: float = 0.12
    success_hold_steps: int = 5
    bullseye_tol_m: float = 0.05
    # Reward weights (PPO only; BC/DAgger are supervised and ignore reward).
    w_progress: float = 10.0  # potential-based: reward per metre closed toward goal
    w_time: float = 0.02  # living penalty per step → reward fast reaches
    w_collision: float = 15.0  # a hit is bad ...
    w_action: float = 1e-3
    r_success: float = 10.0  # reaching the 12cm target at all = the goal
    r_bullseye: float = 10.0  # extra, climbs steeply toward the 5cm center
    r_timeout: float = 25.0  # ... but stalling out is much worse than a hit
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
    # Collection: focus on scenes the policy nearly solves (expert can finish from there).
    level_bounds: list[int] | None = None  # default [2, 3] in collect_round
    almost_solve_tol_m: float = 0.18  # keep episodes that get within ~18cm of goal
    max_far_dist_m: float = 0.45  # discard episodes that never get closer than this
    skip_far_failures: bool = True  # drop hopeless episodes (no recovery training)
    rac_only_if_almost: bool = True  # RaC recovery only when almost at goal; else expert-finish or skip
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
    lr: float = 1e-4  # legacy / fallback
    # Stabilized fine-tuning: warm up the value head before touching the policy,
    # nudge the policy with a low LR, and hard-stop any update that drifts too far.
    policy_lr: float = 1e-5
    value_lr: float = 1e-3
    value_warmup_steps: int = 15_000
    kl_stop: float = 0.05
    n_envs: int = 8
    kl_anchor: float = 0.5
    entropy_coef: float = 0.005
    pool_scenes: int = 256
    archive_m: int = 16
    beta: float = 2.0
    rdiv_cap: float = 0.5
    beta_halvings_max: int = 3
    total_steps: int = 1_000_000
    # Diversity-reward gating. Default: success-only (near-miss ungating consistently
    # degraded the policy in BC-base probes — see initial_testings/ppo_collapse_experiments).
    div_ungate: bool = False
    div_quality_d0: float = 0.30
    div_ungate_min_quality: float = 1.0  # 1.0 = success-only even if div_ungate is enabled


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
