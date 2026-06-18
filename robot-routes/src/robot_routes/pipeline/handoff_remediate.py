"""Analyze medium preflight results and apply bounded remediations before day run."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_routes.expert.oracle import ExpertOracle
from robot_routes.pipeline.bc_ladder import expert_replay_check
from robot_routes.utils.config import ExpertConfig, load_config


@dataclass
class Finding:
    id: str
    severity: str  # info | warn | critical
    detail: str
    metric: float | None = None


@dataclass
class RemediationResult:
    findings: list[Finding] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    verify_ok: bool = False
    verify_log: str = ""


def _expert_replay_rate(root: Path, demos: Path) -> float | None:
    if not demos.exists():
        return None
    expert = ExpertOracle(load_config(root / "configs/expert/rrt_connect.yaml", ExpertConfig))
    _, rate = expert_replay_check(demos, expert, min_rate=0.0, max_episodes=30)
    return rate


def _dagger_round_stats(run_dir: Path) -> list[float]:
    path = run_dir / "dagger" / "round_stats.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [float(x) for x in data] if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def analyze_medium(root: Path, run_dir: Path, report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    events = report.get("pipeline_state", {}).get("events", [])

    replay_rates = [
        float(e["expert_replay_rate"])
        for e in events
        if e.get("event") == "G-BC_ladder" and e.get("expert_replay_rate") is not None
    ]
    fresh_replay = _expert_replay_rate(root, run_dir / "collect" / "demos.h5")
    if fresh_replay is not None:
        replay_rates.append(fresh_replay)
        findings.append(
            Finding(
                "expert_replay_fresh",
                "info",
                f"expert replay on medium demos: {fresh_replay:.1%}",
                fresh_replay,
            )
        )
    if replay_rates:
        worst = min(replay_rates)
        if worst < 0.95:
            findings.append(
                Finding(
                    "expert_replay_low",
                    "warn" if worst >= 0.70 else "critical",
                    f"expert replay {worst:.1%} < 95% — demo/label quality risk",
                    worst,
                )
            )

    ev = report.get("eval") or {}
    sr = ev.get("success_rate")
    if sr is not None:
        findings.append(
            Finding("val_success", "info", f"post-pipeline val success {sr:.1%}", float(sr))
        )
        if float(sr) < 0.20:
            findings.append(
                Finding(
                    "val_success_low",
                    "warn",
                    f"val success {sr:.1%} < 20% — bumping day training budget",
                    float(sr),
                )
            )

    rounds = _dagger_round_stats(run_dir)
    if rounds:
        findings.append(
            Finding(
                "dagger_rounds",
                "info",
                f"dagger round val SR: {[round(x, 3) for x in rounds]}",
                rounds[-1] if rounds else None,
            )
        )
        if len(rounds) >= 2 and rounds[-1] < rounds[-2] - 0.05:
            findings.append(
                Finding(
                    "dagger_regress",
                    "warn",
                    f"last dagger round SR dropped ({rounds[-2]:.2f} → {rounds[-1]:.2f})",
                    rounds[-1],
                )
            )

    for e in events:
        if e.get("event") == "G-BC_preflight_baseline":
            findings.append(
                Finding(
                    "gbc_bypassed",
                    "info",
                    f"G-BC bypassed at train_bc: {e.get('msg', '')}",
                )
            )

    return findings


def _patch_yaml_key(path: Path, key: str, value: Any) -> bool:
    data = yaml.safe_load(path.read_text())
    if data.get(key) == value:
        return False
    data[key] = value
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return True


def _write_day_overrides(root: Path, overrides: dict[str, Any]) -> Path:
    out = root / "configs" / "handoff" / "day_overrides.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if out.exists():
        existing = yaml.safe_load(out.read_text()) or {}
    merged = {**existing, **overrides}
    out.write_text(yaml.dump(merged, default_flow_style=False))
    return out


def apply_remediations(root: Path, findings: list[Finding]) -> RemediationResult:
    result = RemediationResult(findings=findings)
    expert_path = root / "configs/expert/rrt_connect.yaml"
    expert_data = yaml.safe_load(expert_path.read_text())
    day_overrides: dict[str, Any] = {}

    for f in findings:
        if f.id == "expert_replay_low" and f.metric is not None:
            if f.metric < 0.95 and expert_data.get("t_label_s", 3.0) < 4.0:
                if _patch_yaml_key(expert_path, "t_label_s", 4.0):
                    result.actions.append("expert t_label_s 3.0 → 4.0 (more replan budget for DAgger labels)")
                    result.files_changed.append(str(expert_path))
            if f.metric < 0.85 and expert_data.get("t_validate_s", 10.0) < 12.0:
                if _patch_yaml_key(expert_path, "t_validate_s", 12.0):
                    result.actions.append("expert t_validate_s 10 → 12 (stricter demo validation)")
                    result.files_changed.append(str(expert_path))

        if f.id == "val_success_low":
            day_overrides["bc_epochs"] = 175
            day_overrides["dagger_epochs"] = 90
            result.actions.append("day profile: bc_epochs→175, dagger_epochs→90 via handoff overrides")

        if f.id == "dagger_regress":
            day_overrides.setdefault("dagger_rounds", 6)
            result.actions.append("day profile: dagger_rounds→6 (extra round after regress)")

    if day_overrides:
        path = _write_day_overrides(root, day_overrides)
        result.files_changed.append(str(path))

    return result


def verify_changes(root: Path, files_changed: list[str], *, run_smoke: bool = False) -> tuple[bool, str]:
    """Run unit tests + profile verify; smoke only if non-config code changed."""
    import os

    py = root / ".venv/bin/python"
    if not py.exists():
        py = Path("python3")
    env = {**os.environ, "PYTHONPATH": "src", "MUJOCO_GL": "egl"}
    lines: list[str] = []

    # Full pytest except scene manifest (profiles share files; last PROFILE= wins on disk).
    cmd = [
        str(py),
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        "not slow and not integration",
        "--ignore=tests/test_scene_sets.py",
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env)
    lines.append(f"pytest exit={proc.returncode}")
    lines.append(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        lines.append(proc.stderr[-1000:] if proc.stderr else "")
        return False, "\n".join(lines)

    from robot_routes.data.scene_sets import verify_scene_sets

    try:
        verify_scene_sets(root, profile="day")
        lines.append("verify_scene_sets(day): OK")
    except Exception as e:
        lines.append(f"verify_scene_sets(day): FAIL — {e}")
        return False, "\n".join(lines)

    config_only = files_changed and all(
        p.endswith((".yaml", ".yml", ".json")) for p in files_changed
    )
    if run_smoke and files_changed and not config_only:
        smoke = subprocess.run(
            [str(py), "scripts/00_smoke.py"],
            cwd=root,
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
        lines.append(f"smoke exit={smoke.returncode}")
        lines.append(smoke.stdout[-1500:] if smoke.stdout else "")
        if smoke.returncode != 0:
            lines.append(smoke.stderr[-1000:] if smoke.stderr else "")
            return False, "\n".join(lines)
    elif run_smoke and config_only:
        lines.append("smoke skipped (config-only handoff changes)")

    return True, "\n".join(lines)


def run_remediation_cycle(
    root: Path, run_dir: Path, report: dict[str, Any]
) -> RemediationResult:
    findings = analyze_medium(root, run_dir, report)
    result = apply_remediations(root, findings)
    ok, log = verify_changes(
        root, result.files_changed, run_smoke=bool(result.files_changed)
    )
    result.verify_ok = ok
    result.verify_log = log
    return result


def remediation_to_dict(result: RemediationResult) -> dict[str, Any]:
    return {
        "findings": [asdict(f) for f in result.findings],
        "actions": result.actions,
        "files_changed": result.files_changed,
        "verify_ok": result.verify_ok,
        "verify_log": result.verify_log,
    }


# --- Day → full grid handoff ---

FULL_PROFILE_DEFAULTS: dict[str, Any] = {
    "n_demos": 2000,
    "bc_epochs": 200,
    "dagger_rounds": 6,
    "dagger_budget": 40000,
    "dagger_epochs": 100,
    "ppo_steps": 1_000_000,
}

DAY_PROFILE_BASE: dict[str, Any] = {
    "n_demos": 2000,
    "bc_epochs": 150,
    "dagger_rounds": 5,
    "dagger_budget": 35000,
    "dagger_epochs": 80,
    "ppo_steps": 500_000,
}

# Wall-clock fraction of a day `full` run per grid condition (measured shape).
GRID_CONDITION_WALL_FRAC: dict[str, float] = {
    "full": 1.0,
    "rac_noreroute": 0.72,
    "bc_dagger": 0.55,
    "bc": 0.40,
}


def _day_profile_actual(root: Path) -> dict[str, Any]:
    """Day profile as run (pipeline.yaml day + handoff day_overrides)."""
    prof = dict(DAY_PROFILE_BASE)
    pipeline = root / "configs/pipeline.yaml"
    if pipeline.exists():
        data = yaml.safe_load(pipeline.read_text()) or {}
        prof.update(data.get("profile", {}).get("day", {}) or {})
    handoff = root / "configs/handoff/day_overrides.yaml"
    if handoff.exists():
        prof.update(yaml.safe_load(handoff.read_text()) or {})
    return prof


def day_wall_seconds(run_dir: Path) -> float:
    """Observed wall time for the day preflight run (seconds)."""
    from robot_routes.pipeline.stage_progress import snapshot_run_progress

    prog = run_dir / ".pipeline_progress.json"
    if prog.exists():
        try:
            data = json.loads(prog.read_text())
            if data.get("done") or (run_dir / "COMPLETED").exists():
                return float(data.get("elapsed_s", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return float(snapshot_run_progress(run_dir).get("elapsed_s", 0))


def _profile_train_scale(day: dict[str, Any], full: dict[str, Any]) -> float:
    """Scale day `full` wall clock → one PROFILE=full job (training budget ratio)."""
    def r(key: str, default: float = 1.0) -> float:
        dv = float(day.get(key) or default)
        fv = float(full.get(key) or default)
        return fv / max(dv, 1e-6)

    # Weight by typical wall-clock share of a full-condition run.
    return (
        0.12 * r("n_demos", 2000)
        + 0.14 * r("bc_epochs", 175)
        + 0.48
        * r("dagger_rounds", 5)
        * r("dagger_budget", 35000)
        * r("dagger_epochs", 90)
        + 0.18 * r("ppo_steps", 500_000)
        + 0.08
    )


def project_grid_eta_from_day(
    root: Path,
    run_dir: Path,
    full_overrides: dict[str, Any],
    *,
    conditions: list[str],
    seeds: int = 3,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Use completed day preflight wall time to estimate grid duration."""
    day_wall = day_wall_seconds(run_dir)
    if day_wall <= 0:
        return {"ok": False, "reason": "day wall time not available yet"}

    day_prof = _day_profile_actual(root)
    full_prof = {**FULL_PROFILE_DEFAULTS, **full_overrides}
    train_scale = _profile_train_scale(day_prof, full_prof)

    per_job_s: dict[str, float] = {}
    for cond in conditions:
        frac = GRID_CONDITION_WALL_FRAC.get(cond, 0.85)
        per_job_s[cond] = day_wall * frac * train_scale

    sequential_s = sum(per_job_s[c] * seeds for c in conditions)
    parallel_s = sequential_s / max(concurrency, 1)

    return {
        "ok": True,
        "day_wall_s": round(day_wall, 1),
        "day_wall_h": round(day_wall / 3600, 2),
        "train_scale": round(train_scale, 3),
        "day_profile": day_prof,
        "full_profile": full_prof,
        "per_job_h": {k: round(v / 3600, 2) for k, v in per_job_s.items()},
        "grid_jobs": len(conditions) * seeds,
        "sequential_h": round(sequential_s / 3600, 1),
        "sequential_d": round(sequential_s / 86400, 2),
        "parallel_h_concurrency_1": round(parallel_s / 3600, 1),
        "concurrency": concurrency,
    }



def _load_json_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        v = data.get(key)
        return float(v) if v is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _ppo_gate_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for e in reversed(events):
        if e.get("event") in ("G-PPO", "G-PPO_NOGO", "G-PPO_GO"):
            return e
    return None


def analyze_day(root: Path, run_dir: Path) -> list[Finding]:
    """Summarize completed day preflight for full-grid tuning."""
    findings: list[Finding] = []
    st_path = run_dir / "pipeline_state.json"
    events: list[dict[str, Any]] = []
    if st_path.exists():
        events = json.loads(st_path.read_text()).get("events", [])

    fresh_replay = _expert_replay_rate(root, run_dir / "collect" / "demos.h5")
    if fresh_replay is not None:
        findings.append(
            Finding(
                "expert_replay_fresh",
                "info",
                f"expert replay on day demos: {fresh_replay:.1%}",
                fresh_replay,
            )
        )
        if fresh_replay < 0.85:
            findings.append(
                Finding(
                    "expert_replay_low",
                    "critical",
                    f"expert replay {fresh_replay:.1%} < 85%",
                    fresh_replay,
                )
            )

    rounds = _dagger_round_stats(run_dir)
    if rounds:
        findings.append(
            Finding(
                "dagger_rounds",
                "info",
                f"dagger round val SR: {[round(x, 3) for x in rounds]}",
                rounds[-1],
            )
        )
        peak = max(rounds)
        if peak < 0.15:
            findings.append(
                Finding(
                    "dagger_peak_low",
                    "warn",
                    f"dagger peak val SR {peak:.1%} < 15%",
                    peak,
                )
            )
        if len(rounds) >= 2 and rounds[-1] >= rounds[0] + 0.05:
            findings.append(
                Finding(
                    "dagger_improving",
                    "info",
                    f"dagger trend up ({rounds[0]:.2f} → {rounds[-1]:.2f})",
                    rounds[-1],
                )
            )

    val_sr = _load_json_metric(run_dir / "eval" / "val_eval.json", "success_rate")
    if val_sr is not None:
        findings.append(
            Finding("val_success", "info", f"final val success {val_sr:.1%}", val_sr)
        )
        if val_sr < 0.20:
            findings.append(
                Finding(
                    "val_success_low",
                    "warn",
                    f"val success {val_sr:.1%} < 20%",
                    val_sr,
                )
            )
        elif val_sr >= 0.40:
            findings.append(
                Finding(
                    "val_success_high",
                    "info",
                    f"val success {val_sr:.1%} ≥ 40% — near gate target, can trim budget",
                    val_sr,
                )
            )

    if rounds and len(rounds) >= 2:
        if abs(rounds[-1] - rounds[-2]) < 0.03 and rounds[-1] >= 0.25:
            findings.append(
                Finding(
                    "dagger_plateau",
                    "info",
                    f"dagger plateaued at {rounds[-1]:.1%}",
                    rounds[-1],
                )
            )

    test_sr = _load_json_metric(run_dir / "eval" / "test_eval.json", "success_rate")
    if test_sr is not None:
        findings.append(
            Finding("test_success", "info", f"test success {test_sr:.1%}", test_sr)
        )

    ppo_ev = events and _ppo_gate_event(events)
    if ppo_ev:
        findings.append(
            Finding(
                "ppo_gate",
                "info",
                f"PPO gate: {ppo_ev.get('event')} — {ppo_ev.get('msg', '')[:120]}",
            )
        )

    for e in events:
        if e.get("event") == "G-BC_preflight_baseline":
            findings.append(Finding("gbc_bypassed", "info", e.get("msg", "G-BC bypassed")))

    return findings


def _write_full_overrides(root: Path, overrides: dict[str, Any]) -> Path:
    out = root / "configs/handoff/full_overrides.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(overrides, default_flow_style=False, sort_keys=False))
    return out


def apply_full_grid_remediations(
    root: Path, findings: list[Finding]
) -> tuple[RemediationResult, dict[str, Any]]:
    """Derive PROFILE=full overrides — bump if weak, trim if already trained."""
    result = RemediationResult(findings=findings)
    overrides = dict(FULL_PROFILE_DEFAULTS)

    val_sr = next((f.metric for f in findings if f.id == "val_success"), None)
    peak_dagger = next((f.metric for f in findings if f.id == "dagger_rounds"), None)
    ids = {f.id for f in findings}

    # Not learning — increase budget.
    if "val_success_low" in ids or (val_sr is not None and val_sr < 0.25):
        overrides["bc_epochs"] = 220
        overrides["dagger_epochs"] = 110
        overrides["dagger_budget"] = 45_000
        result.actions.append(
            "full: val weak — bc_epochs→220, dagger_epochs→110, dagger_budget→45000"
        )
    elif peak_dagger is not None and peak_dagger < 0.15:
        overrides["dagger_rounds"] = 7
        overrides["dagger_budget"] = 45_000
        result.actions.append("full: dagger peak <15% — dagger_rounds→7, budget→45000")

    # Already close to trained — trim to fit 7-day window (keep quality).
    elif "val_success_high" in ids or (
        val_sr is not None and val_sr >= 0.40 and "dagger_plateau" in ids
    ):
        overrides["bc_epochs"] = 160
        overrides["dagger_epochs"] = 85
        overrides["dagger_rounds"] = 5
        overrides["dagger_budget"] = 35_000
        overrides["ppo_steps"] = 750_000
        result.actions.append(
            "full: day near target / plateau — trim bc→160, dagger→85ep×5r, ppo→750k"
        )
    elif val_sr is not None and val_sr >= 0.50:
        overrides["bc_epochs"] = 150
        overrides["dagger_epochs"] = 80
        overrides["dagger_rounds"] = 5
        overrides["ppo_steps"] = 500_000
        result.actions.append("full: val ≥50% — spec-min budgets to save wall clock")

    elif "dagger_improving" in ids and val_sr is not None and val_sr >= 0.25:
        result.actions.append("full: day trend OK — keeping spec defaults (200/6/40k/100)")

    if "expert_replay_low" in ids:
        expert_path = root / "configs/expert/rrt_connect.yaml"
        expert_data = yaml.safe_load(expert_path.read_text())
        if expert_data.get("t_label_s", 3.0) < 4.5:
            if _patch_yaml_key(expert_path, "t_label_s", 4.5):
                result.actions.append("expert t_label_s → 4.5 (day replay still low)")
                result.files_changed.append(str(expert_path))

    path = _write_full_overrides(root, overrides)
    result.files_changed.append(str(path))
    return result, overrides


def run_day_to_full_cycle(
    root: Path, run_dir: Path
) -> tuple[RemediationResult, dict[str, Any], dict[str, Any]]:
    findings = analyze_day(root, run_dir)
    result, overrides = apply_full_grid_remediations(root, findings)
    eta = project_grid_eta_from_day(
        root,
        run_dir,
        overrides,
        conditions=["full", "rac_noreroute", "bc_dagger"],
        seeds=3,
        concurrency=1,
    )
    ok, log = verify_changes(root, result.files_changed, run_smoke=False)
    try:
        from robot_routes.data.scene_sets import verify_scene_sets

        verify_scene_sets(root, profile="full")
        log += "\nverify_scene_sets(full): OK"
    except Exception as e:
        log += f"\nverify_scene_sets(full): pending — {e}"
        ok = False
    result.verify_ok = ok
    result.verify_log = log
    return result, overrides, eta


def day_report(run_dir: Path) -> dict[str, Any]:
    """Artifact summary for handoff review."""
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "completed": (run_dir / "COMPLETED").exists(),
    }
    for rel in ("eval/val_eval.json", "eval/test_eval.json", "dagger/round_stats.json"):
        p = run_dir / rel
        if p.exists():
            report[rel] = json.loads(p.read_text())
    if (run_dir / "pipeline_state.json").exists():
        st = json.loads((run_dir / "pipeline_state.json").read_text())
        report["gate_events"] = [
            e for e in st.get("events", []) if str(e.get("event", "")).startswith("G-")
        ]
    return report

