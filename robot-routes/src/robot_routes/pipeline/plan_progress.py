"""Pre-travel plan progress: section bars from on-disk artifacts (§11.7.4)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from robot_routes.data.scene_sets import SET_SPECS, load_manifest, verify_scene_sets
from robot_routes.pipeline.stage_progress import snapshot_run_progress
from robot_routes.utils.device import resolve_device
from robot_routes.utils.progress import _bar, fmt_eta, rate_eta_s

SectionStatus = Literal["done", "running", "pending", "failed"]

POOL_COUNTS = {"smoke": 16, "medium": 64, "day": 128, "full": 256}
SET_TOTAL = len(SET_SPECS)

# Pending-section wall-clock estimates (seconds); used when no live rate exists.
SCENE_ETA_PENDING_S = {"smoke": 600, "medium": 3600, "day": 9000, "full": 10800}
PIPELINE_ETA_PROFILE_S = {"smoke": 480, "medium": 4 * 3600, "day": 16 * 3600, "full": 16 * 3600}
GRID_JOB_ETA_S = 16 * 3600
GRID_CONCURRENCY = 2
PREREG_ETA_S = 60.0


@dataclass(frozen=True)
class PlanSectionDef:
    id: str
    label: str


PLAN_SECTIONS: tuple[PlanSectionDef, ...] = (
    PlanSectionDef("env", "Environment (venv, GPU, calibration)"),
    PlanSectionDef("smoke", "Smoke pipeline"),
    PlanSectionDef("scenes_medium", "Scene sets — medium"),
    PlanSectionDef("test_medium", "Preflight run — medium (rac_noreroute)"),
    PlanSectionDef("scenes_day", "Scene sets — day"),
    PlanSectionDef("test_day", "Preflight run — day (full)"),
    PlanSectionDef("scenes_full", "Scene sets — full"),
    PlanSectionDef("prereg", "Pre-registration tag (prereg-v1)"),
    PlanSectionDef("grid", "Full experiment grid"),
)

# Milestone scopes — header ETA uses the active scope, not the whole travel plan.
PLAN_SCOPES: dict[str, tuple[str, ...]] = {
    "medium": ("env", "smoke", "scenes_medium", "test_medium"),
    "preflight": (
        "env",
        "smoke",
        "scenes_medium",
        "test_medium",
        "scenes_day",
        "test_day",
    ),
    "travel": tuple(s.id for s in PLAN_SECTIONS),
}

SCOPE_LABELS = {
    "medium": "Medium preflight",
    "preflight": "Preflight (through day run)",
    "travel": "Full travel plan",
}


def _status_icon(status: SectionStatus) -> str:
    return {"done": "✓", "running": ">", "pending": " ", "failed": "!"}.get(status, " ")


def format_section_line(section: dict[str, Any], width: int = 28) -> str:
    pct = float(section.get("pct", 0))
    label = str(section.get("label", section.get("id", "?")))
    status = section.get("status", "pending")
    detail = section.get("detail", "")
    mark = _status_icon(status)
    line = f"[{mark}] {label:<34s} {_bar(pct, width)} {pct:5.1f}%"
    eta_s = float(section.get("eta_s", 0))
    if status != "done" and eta_s > 0:
        line += f"  eta {fmt_eta(eta_s)}"
    if detail:
        line += f"  |  {detail}"
    return line


def render_plan_dashboard(data: dict[str, Any], bar_width: int = 28) -> str:
    scope = str(data.get("scope", "medium"))
    scope_label = SCOPE_LABELS.get(scope, scope)
    scope_pct = float(data.get("scope_pct", data.get("overall_pct", 0)))
    scope_eta = float(data.get("scope_eta_s", data.get("eta_live_s", 0)))
    full_eta = float(data.get("full_eta_s", scope_eta))
    lines = [
        f"{scope_label} progress",
        f"{_bar(scope_pct, 48)} {scope_pct:5.1f}%  |  eta {fmt_eta(scope_eta)}",
    ]
    if data.get("scope_completion"):
        lines.append(f"est. completion: {data['scope_completion']}")
    if scope != "travel" and full_eta > scope_eta * 1.05:
        lines.append(f"(full travel plan if continued: {fmt_eta(full_eta)})")
    lines.append("")
    for section in data.get("sections", []):
        in_scope = section.get("in_scope", True)
        line = format_section_line(section, width=bar_width)
        if not in_scope and section.get("status") == "pending":
            line += "  [later]"
        lines.append(line)
    next_up = data.get("next_up")
    if next_up:
        lines.extend(["", f"next: {next_up}"])
    ts = data.get("ts")
    if ts:
        lines.append(f"\nupdated {ts}")
    return "\n".join(lines)


def scene_profile_progress(root: Path, profile: str) -> dict[str, Any]:
    """Progress for `make scene-sets PROFILE=…` including pool_rl."""
    status_path = root / "data" / "scenes" / ".generation_status.json"
    if status_path.exists():
        try:
            live = json.loads(status_path.read_text())
        except (json.JSONDecodeError, OSError):
            live = {}
        if live.get("profile") == profile and not live.get("done"):
            set_total = int(live.get("set_total", SET_TOTAL))
            set_index = int(live.get("set_index", 1))
            cur = float(live.get("current", 0))
            tot = float(live.get("total", 1))
            phase = str(live.get("phase", ""))
            if phase == "pool_rl":
                frac = cur / max(tot, 1)
                overall = (set_total + frac) / (set_total + 1)
            elif phase == "sets":
                overall = (set_index - 1) / (set_total + 1)
            else:
                overall = (set_index - 1 + cur / max(tot, 1)) / (set_total + 1)
            pct = min(overall * 100.0, 99.9)
            desc = live.get("desc") or phase
            elapsed = float(live.get("elapsed_s", 0))
            eta_s = rate_eta_s(elapsed, pct) or SCENE_ETA_PENDING_S.get(profile, 3600)
            return {
                "status": "running",
                "pct": round(pct, 1),
                "eta_s": round(eta_s, 1),
                "detail": f"{desc} ({int(cur)}/{int(tot)})",
            }

    manifest = load_manifest(root)
    if profile not in manifest.get("profiles", {}):
        return {
            "status": "pending",
            "pct": 0.0,
            "eta_s": SCENE_ETA_PENDING_S.get(profile, 3600),
            "detail": f"run make scene-sets PROFILE={profile}",
        }

    try:
        verify_scene_sets(root, profile=profile)
    except Exception as exc:
        return {
            "status": "running",
            "pct": 5.0,
            "eta_s": SCENE_ETA_PENDING_S.get(profile, 3600),
            "detail": str(exc)[:80],
        }

    expected = POOL_COUNTS.get(profile, 256)
    pool_path = root / "data" / "pool_rl.json"
    if pool_path.exists():
        try:
            pool = json.loads(pool_path.read_text())
            n = len(pool.get("scenes", []))
        except (json.JSONDecodeError, OSError):
            n = 0
        if n >= expected:
            return {"status": "done", "pct": 100.0, "eta_s": 0.0, "detail": f"verified ({n} pool scenes)"}
        pct = min(90.0 + 10.0 * n / max(expected, 1), 99.9)
        eta_s = rate_eta_s(300.0, pct) or 600.0
        return {
            "status": "running",
            "pct": round(pct, 1),
            "eta_s": round(eta_s, 1),
            "detail": f"pool_rl {n}/{expected}",
        }

    return {
        "status": "running",
        "pct": 90.0,
        "eta_s": 600.0,
        "detail": "sets verified; pool_rl pending",
    }


def _pipeline_profile(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    if meta.exists():
        try:
            return str(json.loads(meta.read_text()).get("profile", "full"))
        except (json.JSONDecodeError, OSError):
            pass
    return "full"


def pipeline_run_progress(run_dir: Path, *, default_profile: str = "full") -> dict[str, Any]:
    run_dir = run_dir.resolve()
    pending_eta = PIPELINE_ETA_PROFILE_S.get(default_profile, GRID_JOB_ETA_S)
    if not run_dir.exists():
        return {"status": "pending", "pct": 0.0, "eta_s": pending_eta, "detail": "not started"}

    if (run_dir / "COMPLETED").exists():
        return {"status": "done", "pct": 100.0, "eta_s": 0.0, "detail": str(run_dir.name)}

    if not (run_dir / "pipeline_state.json").exists() and not (run_dir / "run_meta.json").exists():
        return {"status": "pending", "pct": 0.0, "eta_s": pending_eta, "detail": "not started"}

    snap = snapshot_run_progress(run_dir)
    profile = str(snap.get("profile", _pipeline_profile(run_dir)))
    full_eta = PIPELINE_ETA_PROFILE_S.get(profile, GRID_JOB_ETA_S)
    pct = float(snap.get("overall_pct", 0))
    failed = any(s.get("status") == "FAILED" for s in snap.get("stages", []))
    if snap.get("done"):
        status: SectionStatus = "done"
        pct = 100.0
        eta_s = 0.0
    elif failed:
        status = "failed"
        eta_s = 0.0
    elif snap.get("current_stage"):
        status = "running"
        eta_s = float(snap.get("eta_s", 0)) or rate_eta_s(float(snap.get("elapsed_s", 0)), pct) or full_eta
    else:
        status = "pending"
        eta_s = full_eta

    stage = snap.get("current_stage") or "idle"
    detail = f"{run_dir.name} — {stage} ({pct:.0f}%)"
    if snap.get("live_desc"):
        detail += f" | {snap['live_desc']}"
    return {
        "status": status,
        "pct": round(pct, 1),
        "eta_s": round(eta_s, 1),
        "detail": detail,
        "snap": snap,
    }


def grid_progress(root: Path, runs_root: Path, *, active: bool = True) -> dict[str, Any]:
    from robot_routes.pipeline.conditions import load_grid

    pending_detail = "after prereg-v1 — make grid PROFILE=full"
    if not active:
        remaining = 21
        eta_s = remaining * GRID_JOB_ETA_S / GRID_CONCURRENCY
        return {
            "status": "pending",
            "pct": 0.0,
            "eta_s": round(eta_s, 1),
            "detail": pending_detail,
        }

    if not runs_root.exists():
        remaining = 21
        eta_s = remaining * GRID_JOB_ETA_S / GRID_CONCURRENCY
        return {"status": "pending", "pct": 0.0, "eta_s": round(eta_s, 1), "detail": "not started"}

    grid = load_grid(root)
    priority = grid.get("priority", [])
    seeds = grid.get("seeds", [0, 1, 2])
    total = len(priority) * len(seeds)
    if total == 0:
        return {"status": "pending", "pct": 0.0, "eta_s": 0.0, "detail": "no grid jobs"}

    weighted = 0.0
    done_n = 0
    running_n = 0
    failed_n = 0
    remaining_eta = 0.0
    for cond in priority:
        name = cond if isinstance(cond, str) else cond["name"]
        for seed in seeds:
            run_dir = runs_root / f"{name}_seed{seed}"
            if (run_dir / "COMPLETED").exists():
                weighted += 1.0
                done_n += 1
                continue
            st_path = run_dir / "pipeline_state.json"
            if st_path.exists():
                st = json.loads(st_path.read_text())
                if st.get("stages", {}).get("pipeline", {}).get("status") == "COMPLETED":
                    weighted += 1.0
                    done_n += 1
                    continue
                snap = snapshot_run_progress(run_dir)
                if any(s.get("status") == "FAILED" for s in snap.get("stages", [])):
                    failed_n += 1
                else:
                    running_n += 1
                weighted += float(snap.get("overall_pct", 0)) / 100.0
                job_eta = float(snap.get("eta_s", 0)) or rate_eta_s(
                    float(snap.get("elapsed_s", 0)), float(snap.get("overall_pct", 0))
                ) or GRID_JOB_ETA_S * (1.0 - float(snap.get("overall_pct", 0)) / 100.0)
                remaining_eta += max(job_eta, 0.0)
            else:
                remaining_eta += GRID_JOB_ETA_S

    pct = 100.0 * weighted / total
    eta_s = remaining_eta / max(GRID_CONCURRENCY, 1)
    if done_n >= total:
        status: SectionStatus = "done"
        detail = f"{done_n}/{total} jobs complete"
        eta_s = 0.0
    elif failed_n:
        status = "failed"
        detail = f"{done_n}/{total} done, {failed_n} failed, {running_n} running"
    elif running_n or weighted > 0:
        status = "running"
        detail = f"{done_n}/{total} done, {running_n} running"
    else:
        status = "pending"
        detail = "not started"

    return {"status": status, "pct": round(pct, 1), "eta_s": round(eta_s, 1), "detail": detail}


def env_progress(root: Path) -> dict[str, Any]:
    checks: list[tuple[str, bool]] = []
    venv_py = root / ".venv" / "bin" / "python"
    checks.append(("venv", venv_py.is_file()))
    try:
        dev = resolve_device("auto")
        checks.append(("gpu", dev.type == "cuda"))
    except Exception:
        checks.append(("gpu", False))
    try:
        from robot_routes.pipeline.calibration import load_delta

        load_delta(root)
        checks.append(("calibration", True))
    except Exception:
        checks.append(("calibration", False))

    done = sum(1 for _, ok in checks if ok)
    pct = 100.0 * done / len(checks)
    missing = [name for name, ok in checks if not ok]
    if done == len(checks):
        return {"status": "done", "pct": 100.0, "eta_s": 0.0, "detail": "venv, GPU, calibration OK"}
    return {
        "status": "running" if done else "pending",
        "pct": round(pct, 1),
        "eta_s": 0.0,
        "detail": f"missing: {', '.join(missing)}" if missing else "",
    }


def prereg_progress(root: Path) -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["git", "tag", "-l", "prereg-v1"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if out:
            return {"status": "done", "pct": 100.0, "eta_s": 0.0, "detail": "prereg-v1 tagged"}
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"status": "pending", "pct": 0.0, "eta_s": PREREG_ETA_S, "detail": "run: git tag prereg-v1"}


def detect_scope(raw: dict[str, dict[str, Any]]) -> str:
    """Pick the milestone scope that matches current work."""
    if raw["test_medium"]["status"] != "done":
        return "medium"
    if raw["test_day"]["status"] != "done":
        return "preflight"
    return "travel"


def resolve_scope(raw: dict[str, dict[str, Any]], scope: str | None) -> str:
    if scope in PLAN_SCOPES:
        return scope
    return detect_scope(raw)


def snapshot_plan_progress(
    root: Path,
    *,
    smoke_run: Path | None = None,
    preflight_out: Path | None = None,
    grid_root: Path | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if smoke_run is None:
        smoke_run = root / "runs" / "local_smoke" / "full_seed0"
    if preflight_out is None:
        preflight_out = root / "runs" / "preflight"
    if grid_root is None:
        grid_root = root / "runs" / "grid"

    prereg = prereg_progress(root)
    raw: dict[str, dict[str, Any]] = {
        "env": env_progress(root),
        "smoke": pipeline_run_progress(smoke_run, default_profile="smoke"),
        "scenes_medium": scene_profile_progress(root, "medium"),
        "test_medium": pipeline_run_progress(
            preflight_out / "rac_noreroute_seed0", default_profile="medium"
        ),
        "scenes_day": scene_profile_progress(root, "day"),
        "test_day": pipeline_run_progress(preflight_out / "full_seed0", default_profile="day"),
        "scenes_full": scene_profile_progress(root, "full"),
        "prereg": prereg,
        "grid": grid_progress(root, grid_root, active=prereg["status"] == "done"),
    }

    active_scope = resolve_scope(raw, scope)
    scope_ids = set(PLAN_SCOPES[active_scope])

    sections: list[dict[str, Any]] = []
    full_eta = 0.0
    scope_eta = 0.0
    for spec in PLAN_SECTIONS:
        prog = raw[spec.id]
        eta_s = float(prog.get("eta_s", 0))
        in_scope = spec.id in scope_ids
        if prog["status"] != "done":
            full_eta += eta_s
            if in_scope:
                scope_eta += eta_s
        sections.append(
            {
                "id": spec.id,
                "label": spec.label,
                "status": prog["status"],
                "pct": prog["pct"],
                "eta_s": round(eta_s, 1),
                "detail": prog.get("detail", ""),
                "in_scope": in_scope,
            }
        )

    overall = sum(s["pct"] for s in sections) / max(len(sections), 1)
    scope_sections = [s for s in sections if s["in_scope"]]
    scope_pct = sum(s["pct"] for s in scope_sections) / max(len(scope_sections), 1)
    next_up = next(
        (s["label"] for s in sections if s["in_scope"] and s["status"] in ("pending", "running")),
        None,
    )
    now = datetime.now(timezone.utc)
    scope_completion = datetime.fromtimestamp(
        now.timestamp() + scope_eta, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")
    full_completion = datetime.fromtimestamp(
        now.timestamp() + full_eta, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "ts": now.isoformat(),
        "scope": active_scope,
        "scope_label": SCOPE_LABELS[active_scope],
        "overall_pct": round(overall, 1),
        "scope_pct": round(scope_pct, 1),
        "eta_s": round(full_eta, 1),
        "eta_live_s": round(full_eta, 1),
        "full_eta_s": round(full_eta, 1),
        "scope_eta_s": round(scope_eta, 1),
        "eta_completion": full_completion,
        "scope_completion": scope_completion,
        "sections": sections,
        "next_up": next_up,
    }
