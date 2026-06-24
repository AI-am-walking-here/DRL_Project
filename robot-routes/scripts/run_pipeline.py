"""Pipeline orchestrator DAG (§11.7)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_routes.data.scene_sets import load_scenes, scene_set_path, verify_scene_sets
from robot_routes.pipeline.artifacts import dump_resolved_config, git_hash, write_run_meta
from robot_routes.pipeline.bc_ladder import run_gbc_ladder
from robot_routes.pipeline.calibration import gate_cal, load_delta, run_calibration
from robot_routes.pipeline.conditions import condition_spec, load_grid, stage_list
from robot_routes.pipeline.dither import run_dither_gate
from robot_routes.pipeline.g_ppo import evaluate_g_ppo, resolve_compare_evals
from robot_routes.pipeline.gates import eval_success_on_scenes, gate_bc
from robot_routes.pipeline.notify import notify
from robot_routes.pipeline.progress import PipelineProgress, ordered_stages
from robot_routes.pipeline.setup_checks import assert_delta_invariant, run_setup_checks
from robot_routes.pipeline.stage_resume import eval_artifact_valid, eval_resume_meta
from robot_routes.pipeline.state_merge import merge_pipeline_state, reconcile_from_stamps
from robot_routes.pipeline.verdicts import write_verdicts
from robot_routes.pipeline.watchdog import StageWatchdog, run_with_heartbeat
from robot_routes.utils.config import (
    BCConfig,
    DaggerRacConfig,
    DiversityConfig,
    EvalConfig,
    PolicyConfig,
    load_config,
    load_yaml,
)
from robot_routes.utils.device import project_python, resolve_device
from robot_routes.utils.gpu_oom import cuda_alloc_env, is_cuda_oom, write_oom_backoff
from robot_routes.utils.seeding import seed_everything

STAGES = [
    "setup",
    "scene_sets",
    "calibrate_delta",
    "collect_bc",
    "train_bc",
    "dagger_rac",
    "curriculum",
    "evaluate_val",
    "ppo",
    "evaluate_test",
    "verdicts",
    "report_assets",
]


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


class PipelineState:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "pipeline_state.json"
        self.data: dict[str, Any] = {"stages": {}, "events": [], "dagger_rounds": []}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "PENDING")

    def stage_hash(self, stage: str) -> str | None:
        return self.data["stages"].get(stage, {}).get("config_hash")

    def set_status(self, stage: str, status: str, **extra: Any) -> None:
        entry = self.data["stages"].setdefault(stage, {})
        entry["status"] = status
        entry.update(extra)
        entry["updated"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def invalidate_downstream(self, stage: str) -> None:
        if stage not in STAGES:
            return
        idx = STAGES.index(stage)
        for s in STAGES[idx + 1 :]:
            if s in self.data["stages"]:
                self.data["stages"][s]["status"] = "PENDING"
        self.save()

    def heartbeat(self) -> None:
        (self.run_dir / "heartbeat").write_text(str(time.time()))

    def log_event(self, event: str, **kwargs: Any) -> None:
        self.data["events"].append(
            {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
        )
        self.save()


def disk_ok(min_gb: float) -> bool:
    st = os.statvfs(".")
    free_gb = (st.f_bavail * st.f_frsize) / 2**30
    return free_gb >= min_gb


def validate_invariants(
    env_cfg: dict[str, Any], expert_cfg: dict[str, Any], dagger_cfg: dict[str, Any]
) -> None:
    assert expert_cfg["margin_plan_m"] > dagger_cfg["eps_danger_m"]
    assert dagger_cfg["eps_safe_m"] > expert_cfg["margin_plan_m"]
    assert env_cfg["success_hold_steps"] == dagger_cfg["settle_steps"] == 5


def profile_config_path(
    root: Path,
    run_dir: Path,
    rel_path: str,
    profile: str,
    overrides: dict[str, Any],
) -> str:
    base = root / rel_path
    if profile == "full" and not overrides:
        return str(base)
    data = yaml.safe_load(base.read_text())
    data.update(overrides)
    out = run_dir / "configs" / Path(rel_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(data, default_flow_style=False))
    return str(out)


def use_frozen_run_configs(run_dir: Path, *, force: bool) -> bool:
    """Keep baked run_dir/configs when resuming — avoids handoff edits mid-run."""
    if force:
        return False
    if not (run_dir / "configs" / "bc.yaml").exists():
        return False
    st_path = run_dir / "pipeline_state.json"
    if not st_path.exists():
        return False
    try:
        stages = json.loads(st_path.read_text()).get("stages", {})
    except (json.JSONDecodeError, OSError):
        return False
    for name, entry in stages.items():
        if name == "setup":
            continue
        if entry.get("status") in ("COMPLETED", "RUNNING"):
            return True
    return False


def frozen_config_paths(run_dir: Path) -> tuple[str, str, str]:
    cfg = run_dir / "configs"
    return (
        str(cfg / "bc.yaml"),
        str(cfg / "dagger_rac.yaml"),
        str(cfg / "curriculum.yaml"),
    )


def eval_scene_cap(prof: dict[str, Any], eval_cfg: EvalConfig) -> int:
    if prof.get("eval_scenes") is not None:
        return int(prof["eval_scenes"])
    return eval_cfg.val_per_level


def routes_scene_cap(prof: dict[str, Any], eval_cfg: EvalConfig, n_scenes: int) -> int:
    if prof.get("routes_scenes") is not None:
        return min(int(prof["routes_scenes"]), n_scenes)
    return min(eval_cfg.routes_scenes, n_scenes)


def skip_prereg_for(profile: str, prof: dict[str, Any]) -> bool:
    return bool(prof.get("skip_prereg") or profile == "smoke" or os.environ.get("PIPELINE_SKIP_PREREG"))


def resolve_watchdog_min(
    pipeline_cfg: dict[str, Any] | None,
    profile: str,
    stage: str,
    default_min: int,
) -> int:
    if pipeline_cfg:
        per = pipeline_cfg.get("stage_watchdog_min") or {}
        if stage in per:
            mins = int(per[stage])
            return max(5, mins // 10) if profile in ("smoke", "medium") else mins
    return default_min


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    heartbeat: Callable[[], None] | None = None,
    on_tick: Callable[[], None] | None = None,
    tick_s: float = 15.0,
    watchdog: StageWatchdog | None = None,
) -> None:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    if watchdog is not None:
        watchdog.attach_child(proc)
    while proc.poll() is None:
        if heartbeat is not None:
            heartbeat()
        if on_tick is not None:
            on_tick()
        time.sleep(tick_s)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_stage(
    name: str,
    cmd: list[str] | None,
    state: PipelineState,
    cfg_slice: dict[str, Any],
    root: Path,
    fn: Any = None,
    watchdog_min: int = 30,
    progress: PipelineProgress | None = None,
    pipeline_cfg: dict[str, Any] | None = None,
    profile: str = "full",
    *,
    force: bool = False,
) -> tuple[bool, bool]:
    """Return (success, executed). executed=False when stage was skipped (already COMPLETED)."""
    h = config_hash(cfg_slice)
    prev = state.stage_hash(name)
    if prev and prev != h:
        state.log_event("config_hash_changed", stage=name, old=prev, new=h)
        state.invalidate_downstream(name)
    if not force and state.status(name) == "COMPLETED" and prev == h:
        if progress is not None:
            progress.stage_skipped(name)
        return True, False
    state.set_status(name, "RUNNING", config_hash=h)
    state.heartbeat()
    if progress is not None:
        progress.stage_running(name)
    stage_timeout_min = resolve_watchdog_min(pipeline_cfg, profile, name, watchdog_min)
    try:
        with StageWatchdog(
            state.run_dir, timeout_s=stage_timeout_min * 60, stage=name
        ) as wd:
            if fn is not None:
                run_with_heartbeat(fn, state.heartbeat)
            elif cmd is not None:
                tick = (lambda: progress.refresh_from_disk()) if progress is not None else None
                run_cmd(
                    cmd,
                    root,
                    heartbeat=state.heartbeat,
                    on_tick=tick,
                    watchdog=wd,
                )
                sync_subprocess_state(state)
        state.set_status(name, "COMPLETED", config_hash=h)
        (state.run_dir / f"{name}.stamp").touch()
        state.save()
        if progress is not None:
            progress.stage_done(name)
        return True, True
    except subprocess.CalledProcessError as e:
        if is_cuda_oom(e):
            write_oom_backoff(state.run_dir, stage=name, detail=str(e))
        state.set_status(name, "FAILED", error=str(e), config_hash=h)
        state.log_event("stage_failed", stage=name, error=str(e))
        notify(state.run_dir, "stage_failed", stage=name, error=str(e))
        if progress is not None:
            progress.stage_failed(name, str(e))
        return False, True
    except Exception as e:
        if is_cuda_oom(e):
            write_oom_backoff(state.run_dir, stage=name, detail=str(e))
        state.set_status(name, "FAILED", error=str(e), config_hash=h)
        state.log_event("stage_failed", stage=name, error=str(e))
        notify(state.run_dir, "stage_failed", stage=name, error=str(e))
        if progress is not None:
            progress.stage_failed(name, str(e))
        return False, True


def sync_subprocess_state(state: PipelineState) -> None:
    if state.path.exists():
        disk = json.loads(state.path.read_text())
        if "dagger_rounds" in disk:
            state.data["dagger_rounds"] = disk["dagger_rounds"]
        if "events" in disk:
            seen = {e.get("ts", "") + e.get("event", "") for e in state.data.get("events", [])}
            for e in disk["events"]:
                key = e.get("ts", "") + e.get("event", "")
                if key not in seen:
                    state.data.setdefault("events", []).append(e)
                    seen.add(key)


def sync_dagger_rounds(run_dir: Path, state: PipelineState) -> None:
    rs = run_dir / "dagger" / "round_stats.json"
    if rs.exists():
        rounds = json.loads(rs.read_text())
        state.data["dagger_rounds"] = rounds
        merge_pipeline_state(state.path, {"dagger_rounds": rounds})


def smoke_div_cfg(div_cfg: DiversityConfig, profile: str) -> DiversityConfig:
    if profile != "smoke":
        return div_cfg
    return DiversityConfig(
        resample_pts=div_cfg.resample_pts,
        n_rollouts=5,
        validity_min=3,
        rollout_seeds=div_cfg.rollout_seeds,
        recovery_merge_steps=div_cfg.recovery_merge_steps,
    )


def deps_ready(runs_root: Path, deps: list[str], seed: int) -> bool:
    for dep in deps:
        parts = dep.split(".")
        cond = parts[0]
        ref = runs_root / f"{cond}_seed{seed}"
        if "eval_val" in parts:
            if not (ref / "eval/val_eval.json").exists():
                return False
            if not (ref / "evaluate_val.stamp").exists():
                return False
        elif not (ref / "pipeline_state.json").exists():
            return False
    return True


def wait_for_dep(
    runs_root: Path,
    deps: list[str],
    seed: int,
    timeout_h: float,
    state: PipelineState,
    stage: str,
    progress: PipelineProgress | None = None,
) -> bool:
    if deps_ready(runs_root, deps, seed):
        return True
    if timeout_h <= 0:
        return False
    deadline = time.time() + timeout_h * 3600
    detail = f"waiting for {', '.join(deps)}"
    while time.time() < deadline:
        state.set_status(stage, "WAITING_DEP")
        state.heartbeat()
        if progress is not None:
            progress.stage_waiting(stage, detail)
        if deps_ready(runs_root, deps, seed):
            return True
        time.sleep(30)
    return False


def restart_args(force: bool) -> list[str]:
    return ["--force-restart"] if force else []


class StageRunner:
    """Bundles the per-run context every stage needs, so call sites stay small.

    Replaces the repeated ``run_stage(..., watchdog_min=..., progress=...,
    pipeline_cfg=..., profile=..., force=...)`` boilerplate plus the
    ``if not ok: progress.close(); sys.exit(1)`` guard after every call.
    """

    def __init__(
        self,
        state: PipelineState,
        root: Path,
        progress: PipelineProgress,
        pipeline_cfg: dict[str, Any],
        profile: str,
        force: bool,
        watchdog_min: int,
    ) -> None:
        self.state = state
        self.root = root
        self.progress = progress
        self.pipeline_cfg = pipeline_cfg
        self.profile = profile
        self.force = force
        self.watchdog_min = watchdog_min

    def _run(
        self, name: str, cfg_slice: dict[str, Any], cmd: list[str] | None, fn: Any
    ) -> tuple[bool, bool]:
        return run_stage(
            name,
            cmd,
            self.state,
            cfg_slice,
            self.root,
            fn=fn,
            watchdog_min=self.watchdog_min,
            progress=self.progress,
            pipeline_cfg=self.pipeline_cfg,
            profile=self.profile,
            force=self.force,
        )

    def run(
        self,
        name: str,
        cfg_slice: dict[str, Any],
        *,
        cmd: list[str] | None = None,
        fn: Any = None,
    ) -> bool:
        """Run a required stage. On failure, close progress and exit(1).

        Returns True if the stage actually executed (vs being resume-skipped).
        """
        ok, executed = self._run(name, cfg_slice, cmd, fn)
        if not ok:
            self.progress.close(ok=False)
            sys.exit(1)
        return executed

    def run_optional(
        self,
        name: str,
        cfg_slice: dict[str, Any],
        *,
        cmd: list[str] | None = None,
        fn: Any = None,
    ) -> bool:
        """Run a stage whose failure must not abort the pipeline (e.g. the
        post-PPO re-eval). Returns True only if it executed successfully."""
        ok, executed = self._run(name, cfg_slice, cmd, fn)
        return ok and executed


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="full")
    p.add_argument("--profile", default="full")
    p.add_argument("--out", default="runs/pipeline")
    p.add_argument("--device", default="auto", help="PyTorch device: auto (cuda if available), cpu, cuda:0, ...")
    p.add_argument(
        "--force-restart",
        action="store_true",
        help="Re-run stages even when stamps exist; pass through to stage scripts",
    )
    args = p.parse_args()
    profile = args.profile
    force = args.force_restart
    os.environ.update(cuda_alloc_env())
    root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.out)
    run_dir = runs_root / f"{args.condition}_seed{args.seed}"
    reconcile_from_stamps(run_dir, STAGES)
    state = PipelineState(run_dir)
    py = project_python(root)
    device = resolve_device(args.device)
    grid = load_grid(root)
    spec = condition_spec(grid, args.condition)
    stages_allowed = set(stage_list(spec))
    progress = PipelineProgress(
        run_dir,
        args.condition,
        args.seed,
        profile,
        ordered_stages(stages_allowed),
    )
    sys.stdout.write(f"pipeline progress → {progress.status_path}\n")
    sys.stdout.write(f"python: {py}  device: {device}\n")
    sys.stdout.write(f"live status:  PYTHONPATH=src {py} scripts/watch_pipeline.py {run_dir}\n")
    sys.stdout.write(f"or:           tail -f {run_dir / 'pipeline_status.txt'}\n")
    sys.stdout.flush()
    pipeline_cfg = yaml.safe_load((root / "configs/pipeline.yaml").read_text())
    prof = dict(pipeline_cfg.get("profile", {}).get(profile, {}) or {})
    handoff_day = root / "configs/handoff/day_overrides.yaml"
    if profile == "day" and handoff_day.exists():
        prof.update(yaml.safe_load(handoff_day.read_text()) or {})
    handoff_full = root / "configs/handoff/full_overrides.yaml"
    if profile == "full" and handoff_full.exists():
        prof.update(yaml.safe_load(handoff_full.read_text()) or {})
    disk_min = pipeline_cfg.get("disk_min_gb", 50)
    watchdog_min = pipeline_cfg.get("watchdog_min", 30)

    env_cfg = yaml.safe_load((root / "configs/env/panda_reach.yaml").read_text())
    expert_cfg = yaml.safe_load((root / "configs/expert/rrt_connect.yaml").read_text())
    dagger_yaml = yaml.safe_load((root / "configs/train/dagger_rac.yaml").read_text())
    dagger_yaml.update(
        {
            k: v
            for k, v in {
                "rac_enabled": spec.get("rac_enabled"),
                "reroute_enabled": spec.get("reroute_enabled"),
            }.items()
            if k in spec
        }
    )
    if use_frozen_run_configs(run_dir, force=force):
        bc_cfg_path, dagger_cfg_path, cur_cfg_path = frozen_config_paths(run_dir)
        sys.stdout.write(f"using frozen run configs under {run_dir / 'configs'}\n")
        sys.stdout.flush()
    else:
        bc_cfg_path = profile_config_path(
            root,
            run_dir,
            "configs/train/bc.yaml",
            profile,
            (
                {
                    k: v
                    for k, v in {
                        "n_demos": prof.get("n_demos"),
                        "epochs": prof.get("bc_epochs"),
                    }.items()
                    if v is not None
                }
                if prof
                else {}
            ),
        )
        dagger_cfg_path = profile_config_path(
            root,
            run_dir,
            "configs/train/dagger_rac.yaml",
            profile,
            (
                {
                    **{
                        k: v
                        for k, v in {
                            "rounds": prof.get("dagger_rounds"),
                            "budget": prof.get("dagger_budget"),
                            "retrain_epochs": prof.get("dagger_epochs"),
                        }.items()
                        if v is not None
                    },
                    **{
                        k: dagger_yaml[k]
                        for k in ("rac_enabled", "reroute_enabled")
                        if k in dagger_yaml
                    },
                }
                if prof or any(k in spec for k in ("rac_enabled", "reroute_enabled"))
                else {}
            ),
        )
        cur_yaml = yaml.safe_load((root / "configs/train/curriculum.yaml").read_text())
        if "synthetic_obs" in spec:
            cur_yaml["synthetic_obs"] = spec["synthetic_obs"]
        cur_cfg_path = str(run_dir / "configs/curriculum.yaml")
        Path(cur_cfg_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cur_cfg_path).write_text(yaml.dump(cur_yaml, default_flow_style=False))

    write_run_meta(
        run_dir, root, {"condition": args.condition, "seed": args.seed, "profile": profile}
    )
    dump_resolved_config(
        run_dir, "pipeline", {"condition": args.condition, "profile": profile, **prof}
    )

    runner = StageRunner(
        state, root, progress, pipeline_cfg, profile, force, watchdog_min
    )

    def do_setup() -> None:
        if not disk_ok(disk_min):
            raise RuntimeError(f"disk preflight failed (< {disk_min} GB free)")
        validate_invariants(env_cfg, expert_cfg, dagger_yaml)
        skip_prereg = skip_prereg_for(profile, prof)
        setup_meta = run_setup_checks(root, run_dir, profile=profile, skip_prereg=skip_prereg)
        meta = run_dir / "run_meta.json"
        data = json.loads(meta.read_text())
        data.update(setup_meta)
        data["git_hash"] = git_hash(root)
        meta.write_text(json.dumps(data, indent=2))

    runner.run("setup", {"setup": env_cfg}, fn=do_setup)

    if "collect_bc" in stages_allowed or "evaluate_val" in stages_allowed:
        runner.run(
            "scene_sets",
            {"scene_sets": profile},
            fn=lambda: verify_scene_sets(root, profile=profile),
        )

    cal_payload: dict[str, Any] = {}

    def do_calibrate() -> None:
        nonlocal cal_payload
        cal_path = root / "calibration" / "delta.json"
        if cal_path.exists():
            cal_payload = json.loads(cal_path.read_text())
        else:
            cal_payload = run_calibration(root)
        passed, msg = gate_cal(cal_payload)
        if not passed:
            raise RuntimeError(msg)

    runner.run(
        "calibrate_delta",
        {"calibrate": pipeline_cfg.get("calibration_seed", 424242)},
        fn=do_calibrate,
    )
    delta = load_delta(root)
    cal_path = root / "calibration/delta.json"
    cal_sha = ""
    if cal_path.exists():
        cal_sha = json.loads(cal_path.read_text()).get("sha256", "")
    assert_delta_invariant(root, delta, cal_sha)

    policy_cfg = PolicyConfig(**load_yaml(root / "configs/train/bc.yaml").get("policy", {}))
    eval_cfg = load_config(root / "configs/eval/default.yaml", EvalConfig)
    div_cfg_obj = DiversityConfig(**load_yaml(root / "configs/eval/default.yaml").get("diversity", {}))
    sdiv = smoke_div_cfg(div_cfg_obj, profile)
    bc_cfg = load_config(bc_cfg_path, BCConfig)
    rng = seed_everything(args.seed)
    head_compatible = True

    if "collect_bc" in stages_allowed:
        runner.run(
            "collect_bc",
            yaml.safe_load(Path(bc_cfg_path).read_text()),
            cmd=[
                py,
                "scripts/01_collect_bc_demos.py",
                "--config", bc_cfg_path,
                "--seed", str(args.seed),
                "--out", str(run_dir / "collect"),
                "--run-dir", str(run_dir),
                *restart_args(force),
            ],
        )

    ckpt = run_dir / "bc" / "best.pt"
    if "train_bc" in stages_allowed:
        train_ran = runner.run(
            "train_bc",
            yaml.safe_load(Path(bc_cfg_path).read_text()),
            cmd=[
                py,
                "scripts/02_train_bc.py",
                "--config", bc_cfg_path,
                "--seed", str(args.seed),
                "--out", str(run_dir / "bc"),
                "--data", str(run_dir / "collect/demos.h5"),
                "--device", args.device,
                "--run-dir", str(run_dir),
                *restart_args(force),
            ],
        )
        ckpt = run_dir / "bc" / "best.pt"
        if train_ran:
            scenes_l0 = (
                load_scenes(root, "val_L0") if scene_set_path(root, "val_L0").exists() else []
            )
            cap = eval_scene_cap(prof, eval_cfg)
            if scenes_l0:
                ev = eval_success_on_scenes(
                    ckpt, scenes_l0[:cap], policy_cfg, eval_cfg, sdiv, delta
                )
                passed, branch, msg = gate_bc(
                    ev["success_rate"], bc_cfg.gate_soft, bc_cfg.gate_target
                )
                state.log_event("G-BC", passed=passed, branch=branch, msg=msg, **ev)
                if not passed:
                    ladder_ok, ladder_ckpt, diag = run_gbc_ladder(
                        root,
                        run_dir,
                        run_dir / "collect/demos.h5",
                        bc_cfg,
                        policy_cfg,
                        eval_cfg,
                        sdiv,
                        delta,
                        scenes_l0[:cap],
                        device,
                        rng,
                        force_restart=force,
                    )
                    state.log_event("G-BC_ladder", ok=ladder_ok, **diag)
                    if ladder_ok and ladder_ckpt is not None:
                        ckpt = ladder_ckpt
                        passed = True
                    else:
                        notify(state.run_dir, "G-BC_FAILED", msg=msg, **diag)
                        # Continue to DAgger/RaC if a downstream repair stage exists:
                        # those stages exist precisely to fix weak BC (§5.3). Only a
                        # terminal condition (no dagger_rac downstream, non-preflight)
                        # treats a low BC as a hard failure.
                        downstream_repair = "dagger_rac" in stages_allowed
                        if profile == "smoke" or prof.get("preflight") or downstream_repair:
                            reason = (
                                "continue_to_dagger"
                                if downstream_repair
                                else "preflight_baseline"
                            )
                            state.log_event(
                                "G-BC_continue", reason=reason, msg=msg, profile=profile, **diag
                            )
                            state.set_status(
                                "train_bc", "COMPLETED", config_hash=state.stage_hash("train_bc")
                            )
                            (run_dir / "train_bc.stamp").touch()
                        else:
                            state.set_status("train_bc", "FAILED", gate="G-BC", msg=msg)
                            progress.close(ok=False)
                            sys.exit(1)

    if "dagger_rac" in stages_allowed:
        runner.run(
            "dagger_rac",
            yaml.safe_load(Path(dagger_cfg_path).read_text()),
            cmd=[
                py,
                "scripts/03_run_dagger_rac.py",
                "--config", dagger_cfg_path,
                "--seed", str(args.seed),
                "--out", str(run_dir / "dagger"),
                "--bc-data", str(run_dir / "collect/demos.h5"),
                "--bc-ckpt", str(ckpt),
                "--delta", str(delta),
                "--run-dir", str(run_dir),
                "--profile", profile,
                "--device", args.device,
                *restart_args(force),
            ],
        )
        sync_dagger_rounds(run_dir, state)
        ckpt = run_dir / "dagger" / "best.pt"
        if not ckpt.exists():
            dcfg = load_config(dagger_cfg_path, DaggerRacConfig)
            ckpt = run_dir / "dagger" / f"ckpt_{dcfg.rounds - 1}" / "best.pt"

    merged_h5 = run_dir / "collect/demos.h5"
    if "dagger_rac" in stages_allowed:
        dcfg = load_config(dagger_cfg_path, DaggerRacConfig)
        cand = run_dir / "dagger" / f"merged_{dcfg.rounds - 1}.h5"
        if cand.exists():
            merged_h5 = cand

    if "curriculum" in stages_allowed:
        cur_extra: list[str] = []
        if prof.get("curriculum_epochs") is not None:
            cur_extra += ["--retrain-epochs", str(int(prof["curriculum_epochs"]))]
        if prof.get("dagger_budget") is not None:
            cur_extra += ["--step-budget", str(int(prof["dagger_budget"]))]
        runner.run(
            "curriculum",
            yaml.safe_load(Path(cur_cfg_path).read_text()),
            cmd=[
                py,
                "scripts/04_run_curriculum.py",
                "--config", cur_cfg_path,
                "--seed", str(args.seed),
                "--out", str(run_dir / "curriculum"),
                "--dagger-out", str(run_dir / "dagger"),
                "--delta", str(delta),
                "--profile", profile,
                "--device", args.device,
                "--run-dir", str(run_dir),
                *restart_args(force),
                *cur_extra,
            ],
        )
        ckpt = run_dir / "curriculum" / "best.pt"
        cur_merged = sorted((run_dir / "curriculum").glob("merged_cur_*.h5"))
        if cur_merged:
            merged_h5 = cur_merged[-1]

    if "curriculum" in stages_allowed or "dagger_rac" in stages_allowed:
        ckpt, head_name, head_compatible = run_dither_gate(
            root,
            run_dir,
            ckpt,
            merged_h5,
            policy_cfg,
            bc_cfg,
            eval_cfg,
            sdiv,
            delta,
            device,
            rng,
            profile,
            force_restart=force,
        )
        state.log_event("G-DITHER", head=head_name, head_compatible=head_compatible)
        if head_name == "imle":
            policy_cfg = PolicyConfig(
                **{**load_yaml(root / "configs/train/bc.yaml").get("policy", {}), "head": "imle"}
            )

    eval_ckpt = ckpt
    ppo_ran = False

    def load_val_scenes() -> tuple[list, list]:
        cap = eval_scene_cap(prof, eval_cfg)
        l0 = load_scenes(root, "val_L0")[:cap] if scene_set_path(root, "val_L0").exists() else []
        unseen = (
            load_scenes(root, "val_unseen")[:cap]
            if scene_set_path(root, "val_unseen").exists()
            else l0
        )
        return l0, unseen

    def evaluate_split(scenes: list) -> dict[str, Any]:
        from robot_routes.eval.evaluate import evaluate_checkpoint

        return evaluate_checkpoint(
            eval_ckpt,
            scenes,
            policy_cfg,
            eval_cfg,
            sdiv,
            delta=delta,
            include_routes=True,
            routes_limit=routes_scene_cap(prof, eval_cfg, len(scenes)),
            root=root,
            include_planner_ceiling=profile != "smoke",
        )

    def write_val_eval(*, post_ppo: bool) -> None:
        val_path = run_dir / "eval/val_eval.json"
        if not force and eval_artifact_valid(
            val_path, eval_ckpt, meta={"profile": profile, "post_ppo": post_ppo}
        ):
            tag = " (post-ppo)" if post_ppo else ""
            print(f"evaluate_val: {val_path} exists for {eval_ckpt}{tag} — skipping")
            return
        if not eval_ckpt.exists():
            raise FileNotFoundError(str(eval_ckpt))
        val_l0, val_unseen = load_val_scenes()
        res_l0 = evaluate_split(val_l0)
        res_unseen = evaluate_split(val_unseen)
        (run_dir / "eval").mkdir(exist_ok=True)
        val_path.write_text(
            json.dumps(
                {
                    **res_l0,
                    "val_unseen": res_unseen,
                    "validity_frac": res_unseen.get("validity_frac", 0.0),
                    "_resume_meta": eval_resume_meta(eval_ckpt, profile=profile, post_ppo=post_ppo),
                },
                indent=2,
            )
        )

    if "evaluate_val" in stages_allowed:
        runner.run(
            "evaluate_val",
            {"eval_val": profile, "ckpt": str(eval_ckpt)},
            fn=lambda: write_val_eval(post_ppo=False),
        )

    run_ppo = "ppo" in stages_allowed
    if run_ppo:
        if state.status("ppo") == "WAITING_DEP":
            state.set_status("ppo", "PENDING")
        deps = spec.get("requires") or grid.get("ppo", {}).get("requires", [])
        dep_timeout = 0 if profile == "smoke" else grid.get("dep_timeout_h", 48)
        # When PPO is forced (class deliverable) or standalone, do not block on a
        # sibling condition's eval — that cross-job wait previously cascaded a
        # single BC failure into 4 never-started jobs. Verdicts still compare
        # against whatever sibling evals exist at the end.
        if prof.get("ppo_standalone") or prof.get("ppo_force"):
            deps_ok = True
        else:
            deps_ok = not deps or wait_for_dep(
                runs_root, deps, args.seed, dep_timeout, state, "ppo", progress=progress
            )
        if not deps_ok:
            state.set_status("ppo", "SKIPPED", reason="dep_unmet")
            (state.run_dir / "ppo.stamp").touch()
            state.log_event("G-PPO", result="NO-GO", reason="dep_unmet")
            progress.stage_skipped("ppo")
            run_ppo = False
        if run_ppo:
            if prof.get("ppo_force"):
                go, reason, gmeta = True, "preflight_override", {}
            else:
                go, reason, gmeta = evaluate_g_ppo(
                    run_dir=run_dir,
                    runs_root=runs_root,
                    seed=args.seed,
                    ref_condition="rac_noreroute",
                    ppo_deadline=pipeline_cfg.get("ppo_deadline", "2099-01-01T00:00:00Z"),
                    head_compatible=head_compatible,
                    rng=np.random.default_rng(args.seed),
                )
            state.log_event("G-PPO", go=go, reason=reason, **gmeta)
            if not go:
                state.set_status("ppo", "SKIPPED", reason=reason)
                (state.run_dir / "ppo.stamp").touch()
                notify(state.run_dir, "G-PPO_NOGO", reason=reason, **gmeta)
                progress.stage_skipped("ppo")
                run_ppo = False
        if run_ppo:
            runner.run(
                "ppo",
                yaml.safe_load((root / "configs/train/rl_diversity.yaml").read_text()),
                cmd=[
                    py,
                    "scripts/05_train_rl_diversity.py",
                    "--seed", str(args.seed),
                    "--out", str(run_dir / "ppo"),
                    "--curriculum-ckpt", str(ckpt),
                    "--device", args.device,
                    "--run-dir", str(run_dir),
                    *(["--steps", str(prof["ppo_steps"])] if prof.get("ppo_steps") else []),
                    *restart_args(force),
                ],
            )
            ppo_ckpt = run_dir / "ppo" / "ppo.pt"
            if ppo_ckpt.exists():
                eval_ckpt = ppo_ckpt
                ppo_ran = True

    if ppo_ran and "evaluate_val" in stages_allowed:
        state.invalidate_downstream("evaluate_val")
        runner.run_optional(
            "evaluate_val",
            {"eval_val": profile, "ckpt": str(eval_ckpt), "post_ppo": True},
            fn=lambda: write_val_eval(post_ppo=True),
        )

    if "evaluate_test" in stages_allowed:
        test_scenes = (
            load_scenes(root, "test_unseen")[: prof.get("eval_scenes", 20)]
            if profile == "smoke"
            else load_scenes(root, "test_unseen")
        )

        def do_eval_test() -> None:
            test_path = run_dir / "eval/test_eval.json"
            if not force and eval_artifact_valid(test_path, eval_ckpt, meta={"profile": profile}):
                print(f"evaluate_test: {test_path} exists for {eval_ckpt} — skipping")
                return
            res = evaluate_split(test_scenes)
            (run_dir / "eval").mkdir(exist_ok=True)
            test_path.write_text(
                json.dumps(
                    {**res, "_resume_meta": eval_resume_meta(eval_ckpt, profile=profile)}, indent=2
                )
            )
            (run_dir / "eval/test_touched.json").write_text(
                json.dumps({"seed": args.seed, "ts": datetime.now(timezone.utc).isoformat()})
            )

        runner.run("evaluate_test", {"eval_test": profile, "ckpt": str(eval_ckpt)}, fn=do_eval_test)

    if "verdicts" in stages_allowed:

        def do_verdicts() -> None:
            eval_path = run_dir / "eval/test_eval.json"
            if not eval_path.exists():
                eval_path = run_dir / "eval/val_eval.json"
            verdict_path = run_dir / "hypotheses_verdicts.json"
            if not force and eval_artifact_valid(
                verdict_path, eval_ckpt, meta={"eval_path": str(eval_path.resolve())}
            ):
                print(f"verdicts: {verdict_path} exists — skipping")
                return
            compares = resolve_compare_evals(runs_root, args.condition, args.seed)
            cp = {k: v for k, v in compares.items() if v is not None}
            v = write_verdicts(run_dir, eval_path, mde=eval_cfg.mde_pts, compare_paths=cp or None)
            payload = json.loads(v.read_text())
            payload["_resume_meta"] = eval_resume_meta(eval_ckpt, eval_path=str(eval_path.resolve()))
            v.write_text(json.dumps(payload, indent=2))

        runner.run("verdicts", {"verdicts": args.condition}, fn=do_verdicts)

    if "report_assets" in stages_allowed:

        def do_report() -> None:
            from robot_routes.eval.plots import build_report

            report_dir = run_dir / "report"
            summary = report_dir / "summary.json"
            if not force and summary.is_file() and summary.stat().st_size > 0:
                print(f"report_assets: {summary} exists — skipping")
                return
            build_report(run_dir, report_dir)

        runner.run("report_assets", {"report": profile}, fn=do_report)

    sync_dagger_rounds(run_dir, state)
    state.set_status("pipeline", "COMPLETED")
    (run_dir / "pipeline.stamp").touch()
    (run_dir / "COMPLETED").touch()
    merge_pipeline_state(
        state.path,
        {
            "stages": state.data.get("stages", {}),
            "events": state.data.get("events", []),
            "dagger_rounds": state.data.get("dagger_rounds", []),
        },
    )
    reconcile_from_stamps(run_dir, STAGES)
    notify(state.run_dir, "pipeline_complete", condition=args.condition, seed=args.seed)
    progress.close(ok=True)
    print(f"pipeline done → {run_dir}")


if __name__ == "__main__":
    main()
