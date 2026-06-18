#!/usr/bin/env python3
"""Stage 2: DAgger + RaC (§6) with G-DATA / G-REGRESS enforcement."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typing import Any

import torch

from robot_routes.agents.bc_trainer import train_bc
from robot_routes.agents.dagger_rac import collect_round
from robot_routes.agents.policy import load_checkpoint
from robot_routes.data.scene_sets import load_scenes, scene_set_path
from robot_routes.data.schema import merge_shards, write_shard
from robot_routes.envs.panda_reach_env import PandaReachEnv
from robot_routes.eval.evaluate import evaluate_checkpoint
from robot_routes.expert.collision import CollisionChecker
from robot_routes.expert.oracle import ExpertOracle
from robot_routes.pipeline.dagger_checkpoint import (
    CHECKPOINT_EVERY,
    clear_dagger_partial,
    load_dagger_partial,
    partial_budget,
    save_dagger_partial,
)
from robot_routes.pipeline.dagger_resume import (
    DaggerResumeState,
    _clear_round_artifacts,
    detect_dagger_resume,
    save_round_stats,
)
from robot_routes.pipeline.gates import gate_data, gate_regress, gate_regress_abort
from robot_routes.pipeline.notify import notify
from robot_routes.pipeline.stage_progress import write_stage_live
from robot_routes.pipeline.videos import render_round_videos
from robot_routes.utils.config import (
    BCConfig,
    DaggerRacConfig,
    DiversityConfig,
    EvalConfig,
    ExpertConfig,
    PolicyConfig,
    load_config,
    load_yaml,
)
from robot_routes.utils.seeding import seed_everything
from robot_routes.utils.device import COLLECT_DEVICE, resolve_device


def collect_round_with_retry(
    env: PandaReachEnv,
    policy,
    expert: ExpertOracle,
    cfg: DaggerRacConfig,
    rng,
    delta: float,
    k: int,
    base_seed: int,
    cc: CollisionChecker,
    out: Path,
    profile: str = "full",
    run_dir: Path | None = None,
    progress_cb: Any = None,
) -> tuple[Path, dict]:
    initial_rows, ep_map, start_b = load_dagger_partial(out, k)
    if start_b > 0:
        print(
            f"resuming dagger round {k} collection from {start_b} transitions "
            f"({len(ep_map)} episodes)"
        )

    def _checkpoint_cb(rows, episode_scenes: dict, budget: int) -> None:
        save_dagger_partial(out, k, rows, episode_scenes, budget)

    for attempt in range(2):
        round_rng = seed_everything(base_seed + k * 1000 + attempt * 7777)
        meta: dict = {}
        rows = collect_round(
            env,
            policy,
            expert,
            cfg,
            round_rng,
            delta_reroute=delta,
            meta_out=meta,
            progress_cb=progress_cb,
            initial_rows=list(initial_rows) if attempt == 0 else None,
            episode_scenes_init=dict(ep_map) if attempt == 0 else None,
            checkpoint_cb=_checkpoint_cb if attempt == 0 else None,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        shard = out / f"round_{k}.h5"
        ep_map = meta.get("episode_scenes", ep_map)
        max_ep = max((r.episode_id for r in rows), default=0)
        scenes = [ep_map.get(i, env.scene.to_json()) for i in range(max_ep + 1)]
        write_shard(shard, rows, scenes)
        clear_dagger_partial(out, k)
        ok, msg = gate_data(shard, cfg, meta=meta, cc=cc)
        meta_path = out / f"round_{k}_meta.json"
        meta_path.write_text(json.dumps({"attempt": attempt, "gate_data": msg, **meta}, indent=2))
        if ok:
            return shard, meta
        if attempt == 0:
            notify(
                out.parent if (out.parent / "pipeline_state.json").exists() else None,
                "G-DATA_retry",
                round=k,
                msg=msg,
            )
            shard.unlink(missing_ok=True)
            clear_dagger_partial(out, k)
        else:
            if profile == "smoke":
                notify(run_dir, "G-DATA_smoke_warn", round=k, msg=msg)
                return shard, meta
            raise RuntimeError(f"G-DATA failed round {k} after retry: {msg}")
    raise RuntimeError("unreachable")


def resolve_round_shard(
    out: Path,
    k: int,
    cfg: DaggerRacConfig,
    cc: CollisionChecker,
    *,
    reuse: bool,
    collect_fn: Any,
) -> tuple[Path, dict]:
    """Reuse round_k.h5 when G-DATA passes; otherwise collect (or clear stale shard)."""
    shard_path = out / f"round_{k}.h5"
    meta_path = out / f"round_{k}_meta.json"
    if reuse and shard_path.exists():
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        ok, msg = gate_data(shard_path, cfg, meta=meta, cc=cc)
        if ok:
            print(f"dagger round {k}: reusing {shard_path} (G-DATA ok)")
            return shard_path, meta
        print(f"dagger round {k}: stale shard failed G-DATA ({msg}), re-collecting")
        _clear_round_artifacts(out, k)
    return collect_fn()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train/dagger_rac.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/dagger_rac")
    p.add_argument("--bc-data", default="runs/bc_collect/demos.h5")
    p.add_argument("--bc-ckpt", default="runs/bc_train/best.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--delta", type=float, default=0.15)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--profile", default="full")
    p.add_argument(
        "--force-restart",
        action="store_true",
        help="ignore on-disk round checkpoints and restart from round 0",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / args.config, DaggerRacConfig)
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    cc = CollisionChecker(margin_plan=expert.cfg.margin_plan_m)
    raw_bc = load_yaml(root / "configs/train/bc.yaml")
    policy_cfg = PolicyConfig(**raw_bc.get("policy", {}))
    bc_cfg = load_config(root / "configs/train/bc.yaml", BCConfig)
    eval_cfg = load_config(root / "configs/eval/default.yaml", EvalConfig)
    div_cfg = DiversityConfig(**load_yaml(root / "configs/eval/default.yaml").get("diversity", {}))
    rng = seed_everything(args.seed)
    train_device = resolve_device(args.device)
    collect_device = COLLECT_DEVICE
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir) if args.run_dir else out.parent
    resume: DaggerResumeState = detect_dagger_resume(
        out,
        Path(args.bc_data),
        Path(args.bc_ckpt),
        cfg.rounds,
        force_restart=args.force_restart,
    )
    if resume.resumed:
        pb = partial_budget(out, resume.start_round)
        msg = (
            f"resuming dagger from round {resume.start_round + 1}/{cfg.rounds}"
            + (f" ({pb} trans checkpointed)" if pb else "")
            if resume.start_round < cfg.rounds
            else f"dagger all {cfg.rounds} rounds complete — finalizing"
        )
        print(msg)
        notify(
            run_dir,
            "dagger_resume",
            start_round=resume.start_round,
            completed_through=resume.completed_through,
            policy_ckpt=str(resume.policy_ckpt),
            merged_base=str(resume.merged_base),
            partial_transitions=pb,
        )

    if resume.start_round >= cfg.rounds:
        best = out / "best.pt"
        final_ckpt = out / f"ckpt_{cfg.rounds - 1}" / "best.pt"
        if final_ckpt.exists():
            shutil.copy(final_ckpt, best)
        save_round_stats(out, resume.round_stats)
        if args.run_dir:
            from robot_routes.pipeline.state_merge import merge_pipeline_state

            merge_pipeline_state(
                Path(args.run_dir) / "pipeline_state.json",
                {"dagger_rounds": resume.round_stats},
            )
        return

    policy = load_checkpoint(str(resume.policy_ckpt), policy_cfg).to(collect_device)
    policy.eval()
    val_scenes = load_scenes(root, "val_L0")[:20] if scene_set_path(root, "val_L0").exists() else []
    prev_sr = resume.prev_sr
    round_stats: list[float] = list(resume.round_stats)
    regress_drops: list[float] = list(resume.regress_drops)
    merged_base = resume.merged_base
    for k in range(resume.start_round, cfg.rounds):
        env = PandaReachEnv()

        def _progress_cb(budget_used: int, budget_total: int, episode: int, *, round_k: int = k) -> None:
            write_stage_live(
                run_dir,
                job="dagger_rac",
                phase=f"round_{round_k}_collect",
                round=round_k,
                current=budget_used,
                total=budget_total,
                unit="trans",
                episode=episode,
                desc=f"dagger round {round_k + 1}/{cfg.rounds} collect",
            )

        shard, _ = resolve_round_shard(
            out,
            k,
            cfg,
            cc,
            reuse=resume.reuse_shard and k == resume.start_round,
            collect_fn=lambda: collect_round_with_retry(
                env,
                policy,
                expert,
                cfg,
                rng,
                args.delta,
                k,
                args.seed,
                cc,
                out,
                args.profile,
                run_dir,
                progress_cb=_progress_cb,
            ),
        )
        delta_shard = out / f"delta_d_{k}.h5"
        shutil.copy(shard, delta_shard)
        merged_path = out / f"merged_{k}.h5"
        merge_shards([merged_base, shard], merged_path)
        retrain_cfg = dataclasses.replace(bc_cfg, epochs=cfg.retrain_epochs)
        ckpt_dir = out / f"ckpt_{k}"
        write_stage_live(
            run_dir,
            job="dagger_rac",
            phase=f"round_{k}_retrain",
            round=k,
            current=0,
            total=retrain_cfg.epochs,
            unit="epoch",
            desc=f"dagger round {k + 1}/{cfg.rounds} retrain",
        )
        regressed = False
        for reg_attempt in range(2):
            policy = train_bc(
                merged_path,
                retrain_cfg,
                policy_cfg,
                ckpt_dir,
                train_device,
                rng,
                oom_tag_dir=run_dir,
            )
            policy.eval()
            policy = policy.to(collect_device)
            write_stage_live(
                run_dir,
                job="dagger_rac",
                phase=f"round_{k}_retrain",
                round=k,
                current=retrain_cfg.epochs,
                total=retrain_cfg.epochs,
                unit="epoch",
                desc=f"dagger round {k + 1}/{cfg.rounds} retrain done",
            )
            sr = 0.0
            if val_scenes:
                ev = evaluate_checkpoint(
                    ckpt_dir / "best.pt",
                    val_scenes,
                    policy_cfg,
                    eval_cfg,
                    div_cfg,
                    delta=args.delta,
                )
                sr = ev["success_rate"]
            ok_reg, rmsg = gate_regress(sr, prev_sr)
            if ok_reg:
                break
            if reg_attempt == 0:
                notify(run_dir, "G-REGRESS_retry", round=k, msg=rmsg)
                (out / f"delta_d_{k}.h5").unlink(missing_ok=True)
                shard.unlink(missing_ok=True)
                merged_path.unlink(missing_ok=True)
                env = PandaReachEnv()
                shard, _ = collect_round_with_retry(
                    env,
                    policy,
                    expert,
                    cfg,
                    rng,
                    args.delta,
                    k,
                    args.seed + 999,
                    cc,
                    out,
                    args.profile,
                    run_dir,
                    progress_cb=_progress_cb,
                )
                shutil.copy(shard, out / f"delta_d_{k}.h5")
                merge_shards([merged_base, shard], merged_path)
                regressed = True
            else:
                notify(run_dir, "G-REGRESS_flag", round=k, msg=rmsg)
        round_stats.append(sr)
        regress_drops.append(prev_sr - sr if sr < prev_sr else 0.0)
        ok_abort, abort_msg = gate_regress_abort(regress_drops)
        if not ok_abort:
            notify(run_dir, "G-REGRESS_abort", msg=abort_msg)
            raise RuntimeError(abort_msg)
        prev_sr = sr
        if val_scenes:
            try:
                render_round_videos(policy, val_scenes[:4], out / f"videos_r{k}")
            except Exception:
                pass
        print(f"round {k+1}/{cfg.rounds} done, val_sr={sr:.3f}, regressed={regressed}")
        merged_base = merged_path
        save_round_stats(out, round_stats)
        if args.run_dir:
            from robot_routes.pipeline.state_merge import merge_pipeline_state

            merge_pipeline_state(
                Path(args.run_dir) / "pipeline_state.json", {"dagger_rounds": round_stats}
            )
    best = out / "best.pt"
    if (out / f"ckpt_{cfg.rounds - 1}/best.pt").exists():
        shutil.copy(out / f"ckpt_{cfg.rounds - 1}/best.pt", best)
    (out / "round_stats.json").write_text(json.dumps(round_stats))
    if args.run_dir:
        from robot_routes.pipeline.state_merge import merge_pipeline_state

        merge_pipeline_state(
            Path(args.run_dir) / "pipeline_state.json", {"dagger_rounds": round_stats}
        )


if __name__ == "__main__":
    main()
