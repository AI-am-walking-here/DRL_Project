"""Live stage progress snapshots + run-level dashboard (§11.7)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from robot_routes.pipeline.conditions import condition_spec, load_grid, stage_list
from robot_routes.pipeline.progress import ordered_stages
from robot_routes.utils.progress import _bar

STAGE_LIVE = "stage_live.json"
STALE_AFTER_S = 120.0
ORPHAN_AFTER_S = 300.0

# Rough full-profile stage durations (seconds) for ETA when rate-based ETA is unreliable.
_STAGE_ETA_FULL: dict[str, float] = {
    "setup": 120,
    "scene_sets": 600,
    "calibrate_delta": 2400,
    "collect_bc": 9000,
    "train_bc": 1800,
    "dagger_rac": 57600,
    "curriculum": 14400,
    "evaluate_val": 3600,
    "ppo": 7200,
    "evaluate_test": 3600,
    "verdicts": 300,
    "report_assets": 300,
}


def stage_live_path(run_dir: Path) -> Path:
    return run_dir / STAGE_LIVE


def write_stage_live(run_dir: Path, **fields: Any) -> None:
    """Atomic write of in-stage progress (callable from subprocesses)."""
    path = stage_live_path(run_dir)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    (run_dir / "heartbeat").write_text(str(time.time()))


def read_stage_live(run_dir: Path) -> dict[str, Any]:
    path = stage_live_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _file_age_s(path: Path) -> float | None:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _heartbeat_age_s(run_dir: Path) -> float | None:
    hb = run_dir / "heartbeat"
    if not hb.exists():
        return None
    try:
        return time.time() - float(hb.read_text().strip())
    except (ValueError, OSError):
        return _file_age_s(hb)


def _pipeline_pids(run_dir: Path) -> list[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", str(run_dir.resolve())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return [p for p in pids if p != os.getpid()]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def liveness(run_dir: Path, stage_status: str) -> dict[str, Any]:
    """alive | stale | orphaned | idle"""
    if stage_status not in ("RUNNING", "WAITING_DEP"):
        return {"liveness": "idle", "heartbeat_age_s": _heartbeat_age_s(run_dir)}

    live = read_stage_live(run_dir)
    live_ts = live.get("ts")
    live_age: float | None = None
    if live_ts:
        try:
            ts = datetime.fromisoformat(str(live_ts).replace("Z", "+00:00"))
            live_age = (datetime.now(timezone.utc) - ts).total_seconds()
        except ValueError:
            live_age = _file_age_s(stage_live_path(run_dir))

    hb_age = _heartbeat_age_s(run_dir)
    ages = [a for a in (hb_age, live_age) if a is not None]
    min_age = min(ages) if ages else None
    pids = _pipeline_pids(run_dir)

    if min_age is not None and min_age <= STALE_AFTER_S:
        return {
            "liveness": "alive",
            "heartbeat_age_s": hb_age,
            "stage_live_age_s": live_age,
            "pids": pids,
        }
    if pids and min_age is not None and min_age <= ORPHAN_AFTER_S:
        return {
            "liveness": "stale",
            "heartbeat_age_s": hb_age,
            "stage_live_age_s": live_age,
            "pids": pids,
            "warning": f"no progress update for {min_age:.0f}s (process still running)",
        }
    return {
        "liveness": "orphaned",
        "heartbeat_age_s": hb_age,
        "stage_live_age_s": live_age,
        "pids": pids,
        "warning": "stage RUNNING but no live process / progress — may need restart",
    }


def _dagger_sub_progress(run_dir: Path) -> dict[str, Any]:
    dagger_dir = run_dir / "dagger"
    cfg_path = run_dir / "configs" / "dagger_rac.yaml"
    total_rounds = 6
    budget = 40000
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        total_rounds = int(cfg.get("rounds", total_rounds))
        budget = int(cfg.get("budget", budget))

    completed_rounds = 0
    rs_path = dagger_dir / "round_stats.json"
    if rs_path.exists():
        completed_rounds = len(json.loads(rs_path.read_text()))

    # Round in progress: highest round_N.h5 without matching ckpt_N
    in_progress_round: int | None = None
    round_frac = 0.0
    for k in range(total_rounds):
        shard = dagger_dir / f"round_{k}.h5"
        ckpt = dagger_dir / f"ckpt_{k}" / "best.pt"
        if shard.exists() and not ckpt.exists():
            in_progress_round = k
            meta = dagger_dir / f"round_{k}_meta.json"
            if meta.exists():
                completed_rounds = k
                round_frac = 0.85
            else:
                round_frac = max(round_frac, 0.05)
            break

    live = read_stage_live(run_dir)
    if live.get("job") == "dagger_rac":
        cur = float(live.get("current", 0))
        tot = float(live.get("total", budget))
        if tot > 0 and live.get("phase", "").endswith("collect"):
            round_frac = min(cur / tot, 0.999)
            if in_progress_round is None:
                in_progress_round = int(live.get("round", completed_rounds))

    stage_round = completed_rounds + round_frac
    if in_progress_round is not None and completed_rounds < in_progress_round:
        stage_round = in_progress_round + round_frac

    sub: dict[str, Any] = {
        "dagger_round": completed_rounds,
        "dagger_rounds_total": total_rounds,
        "dagger_round_frac": round(round_frac, 3),
        "dagger_budget": budget,
    }
    if live.get("desc"):
        sub["live_desc"] = live["desc"]
    if live.get("phase"):
        sub["dagger_phase"] = live["phase"]
    elif in_progress_round is not None and (dagger_dir / f"round_{in_progress_round}_meta.json").exists():
        sub["dagger_phase"] = f"round_{in_progress_round}_retrain_pending"
    if live.get("current") is not None:
        sub["dagger_transitions"] = f"{int(live['current'])}/{int(live.get('total', budget))}"
        cur = int(live["current"])
        tot = int(live.get("total", budget))
        if tot > 0:
            sub["dagger_collect_pct"] = round(100.0 * cur / tot, 1)
    if rs_path.exists():
        rounds = json.loads(rs_path.read_text())
        if rounds:
            sub["last_success_rate"] = rounds[-1]
    return sub


def _stage_fraction(stage: str | None, sub: dict[str, Any], run_dir: Path) -> float:
    if not stage:
        return 0.0
    if stage == "dagger_rac":
        n = float(sub.get("dagger_round", 0)) + float(sub.get("dagger_round_frac", 0))
        total = int(sub.get("dagger_rounds_total", 1))
        return min(n / max(total, 1), 0.999)
    if stage == "curriculum":
        merged = sorted(run_dir.glob("curriculum/merged_cur_*.h5"))
        cfg_path = run_dir / "configs" / "curriculum.yaml"
        total = 4
        if cfg_path.exists():
            levels = yaml.safe_load(cfg_path.read_text()).get("levels", [])
            total = max(len(levels), 1)
        return min(len(merged) / max(total, 1), 0.999)
    if stage == "collect_bc":
        shards = list((run_dir / "collect").glob("demos_w*.h5"))
        if (run_dir / "collect" / "demos.h5").exists():
            return 0.999
        return min(len(shards) / 8.0, 0.999)
    live = read_stage_live(run_dir)
    if live.get("total"):
        return min(float(live.get("current", 0)) / float(live["total"]), 0.999)
    return 0.0


def _run_elapsed_s(run_dir: Path) -> float:
    candidates = [
        run_dir / "run_meta.json",
        run_dir / "setup.stamp",
        run_dir / "heartbeat",
        run_dir / ".pipeline_progress.json",
    ]
    oldest: float | None = None
    for p in candidates:
        if p.exists():
            t = p.stat().st_mtime
            oldest = t if oldest is None else min(oldest, t)
    if oldest is None:
        return 0.0
    return time.time() - oldest


def _eta_from_profile(
    stages: list[str],
    stage_status: dict[str, str],
    current: str | None,
    stage_frac: float,
    profile: str,
) -> float:
    scale = 0.05 if profile == "smoke" else 1.0
    remaining = 0.0
    found = False
    for s in stages:
        st = stage_status.get(s, "PENDING")
        if st in ("COMPLETED", "SKIPPED"):
            continue
        dur = _STAGE_ETA_FULL.get(s, 600) * scale
        if s == current and st == "RUNNING":
            remaining += dur * (1.0 - stage_frac)
            found = True
        elif not found or st in ("PENDING", "WAITING_DEP"):
            remaining += dur
    return remaining


def _project_root_from_run(run_dir: Path) -> Path:
    for p in [run_dir, *run_dir.parents]:
        if (p / "configs" / "grid.yaml").exists():
            return p
    return run_dir.parents[min(2, len(run_dir.parents) - 1)]


def snapshot_run_progress(run_dir: Path) -> dict[str, Any]:
    """Rebuild dashboard from on-disk artifacts (works even if orchestrator died)."""
    run_dir = run_dir.resolve()
    state_path = run_dir / "pipeline_state.json"
    disk: dict[str, Any] = {}
    if state_path.exists():
        disk = json.loads(state_path.read_text())

    meta_path = run_dir / "run_meta.json"
    condition, seed, profile = "?", 0, "full"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        condition = meta.get("condition", condition)
        seed = int(meta.get("seed", seed))
        profile = meta.get("profile", profile)

    try:
        root = _project_root_from_run(run_dir)
        grid = load_grid(root)
        spec = condition_spec(grid, condition)
        stages = ordered_stages(set(stage_list(spec)))
    except (KeyError, IndexError, OSError):
        stages = ordered_stages({"collect_bc", "train_bc", "dagger_rac", "curriculum", "evaluate_val"})

    stage_status: dict[str, str] = {}
    for s in stages:
        entry = disk.get("stages", {}).get(s, {})
        st = entry.get("status", "PENDING")
        if (run_dir / f"{s}.stamp").exists():
            st = "COMPLETED"
        stage_status[s] = st

    current = None
    for s in stages:
        if stage_status.get(s) == "RUNNING":
            current = s
            break
    if current is None:
        for s in stages:
            if stage_status.get(s) in ("WAITING_DEP", "PENDING") and (
                s not in stage_status or stage_status.get(s) != "COMPLETED"
            ):
                if any(stage_status.get(x) == "COMPLETED" for x in stages[: stages.index(s)]):
                    current = s
                    break

    sub: dict[str, Any] = {}
    if current == "dagger_rac":
        sub = _dagger_sub_progress(run_dir)
    elif current == "curriculum":
        merged = sorted(run_dir.glob("curriculum/merged_cur_*.h5"))
        sub = {"curriculum_step": len(merged), "curriculum_steps_total": 4}
    elif current == "collect_bc":
        shards = list((run_dir / "collect").glob("demos_w*.h5"))
        sub = {"collect_shards_done": len(shards), "collect_shards_total": 8}

    live = read_stage_live(run_dir)
    if live:
        sub.setdefault("live_phase", live.get("phase"))
        if live.get("desc"):
            sub["live_desc"] = live["desc"]

    done_n = sum(1 for s in stages if stage_status.get(s) in ("COMPLETED", "SKIPPED"))
    stage_frac = _stage_fraction(current, sub, run_dir)
    total = len(stages)
    overall = done_n
    if current and stage_status.get(current) == "RUNNING":
        overall += stage_frac
    overall_pct = min(100.0 * overall / max(total, 1), 100.0)

    elapsed = _run_elapsed_s(run_dir)
    cached = run_dir / ".pipeline_progress.json"
    if cached.exists():
        try:
            old = json.loads(cached.read_text())
            if old.get("elapsed_s") and not current:
                elapsed = float(old["elapsed_s"])
        except (json.JSONDecodeError, TypeError):
            pass

    rate = overall_pct / max(elapsed, 1e-6)
    eta_rate = (100.0 - overall_pct) / max(rate, 1e-6) if overall_pct < 100 else 0.0
    eta_profile = _eta_from_profile(stages, stage_status, current, stage_frac, profile)
    eta_s = max(eta_rate, eta_profile) if overall_pct < 5 else min(eta_rate, eta_profile * 2)

    live_info = liveness(run_dir, stage_status.get(current or "", "PENDING"))

    stage_rows: list[dict[str, Any]] = []
    for s in stages:
        st = stage_status.get(s, "PENDING")
        if st in ("COMPLETED", "SKIPPED"):
            spct = 100.0
        elif st == "FAILED":
            spct = 0.0
        elif st == "RUNNING" and s == current:
            spct = round(min(_stage_fraction(s, sub, run_dir), 0.999) * 100.0, 1)
        else:
            spct = 0.0
        stage_rows.append({"name": s, "status": st, "pct": spct})

    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "seed": seed,
        "profile": profile,
        "current_stage": current,
        "current_status": stage_status.get(current or "", "PENDING"),
        "stages_completed": done_n,
        "stages_total": total,
        "pct": round(100.0 * done_n / max(total, 1), 2),
        "overall_pct": round(overall_pct, 2),
        "elapsed_s": round(elapsed, 1),
        "eta_s": round(eta_s, 1),
        "eta_profile_s": round(eta_profile, 1),
        "done": all(stage_status.get(s) in ("COMPLETED", "SKIPPED") for s in stages),
        "ok": None,
        "stages": stage_rows,
        "snapshot": True,
        **sub,
        **live_info,
    }
    if live_info.get("warning"):
        payload["detail"] = live_info["warning"]
    return payload
