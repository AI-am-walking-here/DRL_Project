"""G-DITHER mode-switch diagnostic + IMLE branch (§11.7.2)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from robot_routes.agents.bc_trainer import train_bc
from robot_routes.agents.policy import MDNPolicy, build_policy, load_checkpoint
from robot_routes.contracts import SceneSpec
from robot_routes.data.scene_sets import load_scenes, scene_set_path
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.pipeline.gates import gate_dither
from robot_routes.utils.config import BCConfig, DiversityConfig, EvalConfig, PolicyConfig


def _selected_component(policy: MDNPolicy, obs_np: np.ndarray, a_prev: np.ndarray | None) -> int:
    obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
    logits, mu, _ = policy(obs)
    log_w = torch.log_softmax(logits[0], -1)
    if a_prev is not None:
        ap = torch.as_tensor(a_prev, dtype=torch.float32)
        log_w = log_w - ((mu[0] - ap) ** 2).sum(-1) / (2 * policy.sigma_c**2)
    return int(torch.argmax(log_w).item())


def median_mode_switches(
    policy: MDNPolicy,
    scenes: list[SceneSpec],
    *,
    n_episodes: int = 10,
) -> float:
    env = PandaReachEnv()
    counts: list[int] = []
    for scene in scenes[:n_episodes]:
        obs, info = env.reset(seed=scene.seed, options={"scene": scene})
        prev_k: int | None = None
        switches = 0
        a_prev = None
        for _ in range(env.cfg.horizon):
            k = _selected_component(policy, obs, a_prev)
            if prev_k is not None and k != prev_k:
                switches += 1
            prev_k = k
            a = policy.act(obs, stochastic=True, a_prev=a_prev)
            a_prev = a
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        counts.append(switches)
    return float(np.median(counts)) if counts else 0.0


def run_dither_gate(
    root: Path,
    run_dir: Path,
    ckpt: Path,
    merged_h5: Path,
    policy_cfg: PolicyConfig,
    bc_cfg: BCConfig,
    eval_cfg: EvalConfig,
    div_cfg: DiversityConfig,
    delta: float,
    device: torch.device,
    rng: np.random.Generator,
    profile: str,
    *,
    force_restart: bool = False,
) -> tuple[Path, str, bool]:
    """Returns (winning_ckpt, head_name, head_compatible_for_ppo)."""
    state_path = run_dir / "dither_state.json"
    out_ckpt = run_dir / "curriculum" / "best.pt"
    imle_dir = run_dir / "dither_imle"
    if force_restart:
        if state_path.exists():
            state_path.unlink()
        if imle_dir.is_dir():
            shutil.rmtree(imle_dir, ignore_errors=True)
    elif state_path.exists():
        try:
            saved = json.loads(state_path.read_text())
            if saved.get("passed") and "winner" not in saved:
                return ckpt, "mdn", True
            if "winner" in saved and out_ckpt.exists():
                head = str(saved["winner"])
                compatible = bool(saved.get("head_compatible", head != "imle"))
                return out_ckpt, head, compatible
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    policy = load_checkpoint(str(ckpt), policy_cfg)
    if not isinstance(policy, MDNPolicy):
        return ckpt, policy_cfg.head, True
    scenes = load_scenes(root, "val_L0")[:10] if scene_set_path(root, "val_L0").exists() else []
    med = median_mode_switches(policy, scenes)
    ok, msg = gate_dither(med, float(policy_cfg.dither_escalate_median))
    state_path.write_text(json.dumps({"median_switches": med, "passed": ok, "msg": msg}, indent=2))
    if ok or not merged_h5.exists():
        return ckpt, "mdn", True
    imle_dir = run_dir / "dither_imle"
    imle_cfg = dataclasses.replace(policy_cfg, head="imle")
    epochs = 5 if profile == "smoke" else bc_cfg.epochs
    imle_policy = train_bc(
        merged_h5,
        dataclasses.replace(bc_cfg, epochs=epochs),
        imle_cfg,
        imle_dir,
        device,
        rng,
    )
    imle_ckpt = imle_dir / "best.pt"
    mdn_sr = (
        evaluate_checkpoint(ckpt, scenes, policy_cfg, eval_cfg, div_cfg, delta=delta)[
            "success_rate"
        ]
        if scenes
        else 0.0
    )
    imle_sr = (
        evaluate_checkpoint(imle_ckpt, scenes, imle_cfg, eval_cfg, div_cfg, delta=delta)[
            "success_rate"
        ]
        if scenes
        else 0.0
    )
    winner = imle_ckpt if imle_sr >= mdn_sr else ckpt
    head = "imle" if imle_sr >= mdn_sr else "mdn"
    head_compatible = head != "imle"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(winner, out_ckpt)
    branch = {
        "median_switches": med,
        "mdn_sr": mdn_sr,
        "imle_sr": imle_sr,
        "winner": head,
        "head_compatible": head_compatible,
    }
    state_path.write_text(json.dumps(branch, indent=2))
    return out_ckpt, head, head_compatible
