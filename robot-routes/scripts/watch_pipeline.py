#!/usr/bin/env python3
"""Live pipeline status dashboard (reads on-disk artifacts; never goes stale)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.stage_progress import snapshot_run_progress
from robot_routes.utils.device import project_python
from robot_routes.utils.progress import _bar, fmt_eta


def _liveness_banner(data: dict) -> str:
    live = data.get("liveness", "idle")
    if live == "orphaned":
        return "*** ORPHANED — stage marked RUNNING but no live progress ***"
    if live == "stale":
        return f"** STALE — no update for {data.get('stage_live_age_s') or data.get('heartbeat_age_s', '?'):.0f}s **"
    if live == "alive":
        age = data.get("stage_live_age_s") or data.get("heartbeat_age_s")
        if age is not None:
            return f"live (updated {age:.0f}s ago)"
    return ""


def render(data: dict) -> str:
    overall = float(data.get("overall_pct", data.get("pct", 0)))
    done = int(data.get("stages_completed", 0))
    total = int(data.get("stages_total", 1))
    stage = data.get("current_stage") or "idle"
    status = data.get("current_status", "PENDING")
    elapsed = float(data.get("elapsed_s", 0))
    eta = float(data.get("eta_s", 0))
    eta_prof = float(data.get("eta_profile_s", 0))
    lines = [
        f"{data.get('condition', '?')} seed{data.get('seed', '?')} ({data.get('profile', '?')})",
        _liveness_banner(data),
        f"{_bar(overall, 48)} {overall:5.1f}%",
        f"stages {done}/{total}  |  current: {stage} ({status})",
        f"elapsed {fmt_eta(elapsed)}  |  eta {fmt_eta(eta)}"
        + (f"  (profile est {fmt_eta(eta_prof)})" if eta_prof > 0 else ""),
    ]
    if "dagger_round" in data:
        frac = float(data.get("dagger_round_frac", 0))
        r = data["dagger_round"]
        rt = data.get("dagger_rounds_total", "?")
        lines.append(
            f"dagger round {r}+{frac:.2f}/{rt}"
            + (
                f"  last_sr={data['last_success_rate']:.3f}"
                if data.get("last_success_rate") is not None
                else ""
            )
        )
    if data.get("dagger_transitions"):
        pct = data.get("dagger_collect_pct")
        pct_s = f" ({pct}%)" if pct is not None else ""
        lines.append(f"  transitions {data['dagger_transitions']}{pct_s}")
    if data.get("dagger_phase") or data.get("live_phase"):
        lines.append(f"  phase: {data.get('dagger_phase') or data.get('live_phase')}")
    if data.get("live_desc"):
        lines.append(f"  {data['live_desc']}")
    if "curriculum_step" in data:
        lines.append(
            f"curriculum step {data['curriculum_step']}/{data.get('curriculum_steps_total', '?')}"
        )
    if data.get("collect_shards_done") is not None:
        lines.append(
            f"collect shards {data['collect_shards_done']}/{data.get('collect_shards_total', 8)}"
        )
    if data.get("detail"):
        lines.append(f"detail: {data['detail']}")
    pids = data.get("pids") or []
    if pids:
        lines.append(f"processes: {', '.join(str(p) for p in pids[:5])}")
    lines.append("")
    lines.append("stages:")
    for s in data.get("stages", []):
        mark = {"COMPLETED": "✓", "RUNNING": ">", "FAILED": "!", "WAITING_DEP": "…"}.get(
            s.get("status", ""), " "
        )
        spct = float(s.get("pct", 0))
        if s.get("status") in ("COMPLETED", "SKIPPED"):
            spct = 100.0
        lines.append(
            f"  [{mark}] {s.get('name', '?'):16s} {_bar(spct, 24)} {spct:5.1f}%  {s.get('status', 'PENDING')}"
        )
    if data.get("done"):
        lines.append("")
        lines.append(f"DONE ok={data.get('ok')}")
    ts = data.get("ts", "")
    if ts:
        snap = " (snapshot)" if data.get("snapshot") else ""
        lines.append(f"\nupdated {ts}{snap}")
    return "\n".join(line for line in lines if line is not None)


def main() -> None:
    p = argparse.ArgumentParser(description="Live pipeline progress dashboard")
    p.add_argument("run_dir", type=Path, help="e.g. runs/grid/rac_noreroute_seed0")
    p.add_argument("-n", "--interval", type=float, default=3.0, help="refresh seconds")
    p.add_argument("--once", action="store_true", help="print once and exit")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    root = run_dir.parents[1] if (run_dir.parents[1] / "configs").exists() else run_dir.parents[2]
    py = project_python(root)
    status = run_dir / ".pipeline_progress.json"
    if not status.exists() and not (run_dir / "pipeline_state.json").exists():
        print(f"waiting for run dir {run_dir} ...", file=sys.stderr)
    try:
        while True:
            data = snapshot_run_progress(run_dir)
            try:
                status.parent.mkdir(parents=True, exist_ok=True)
                tmp = status.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2))
                tmp.replace(status)
                human = run_dir / "pipeline_status.txt"
                human.write_text(
                    f"{data['overall_pct']:5.1f}% | {data.get('current_stage', 'idle')} | "
                    f"{data['stages_completed']}/{data['stages_total']} | "
                    f"eta {data['eta_s']/3600:.1f}h | {data.get('liveness', '')}\n"
                )
            except OSError:
                pass
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(render(data))
            sys.stdout.write(f"\n\nwatch: {py} scripts/watch_pipeline.py {run_dir}\n")
            sys.stdout.write(f"tail:  tail -f {run_dir / 'pipeline_status.txt'}\n")
            sys.stdout.flush()
            if data.get("done") or args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
