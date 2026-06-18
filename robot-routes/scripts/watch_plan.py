#!/usr/bin/env python3
"""Live pre-travel plan dashboard (reads on-disk artifacts; never goes stale)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.plan_progress import render_plan_dashboard, snapshot_plan_progress
from robot_routes.utils.device import project_python


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-travel plan progress dashboard")
    p.add_argument("--root", type=Path, default=Path("."), help="project root")
    p.add_argument("--smoke-run", type=Path, default=None, help="smoke pipeline run dir")
    p.add_argument("--preflight-out", type=Path, default=None, help="preflight runs root")
    p.add_argument("--grid-root", type=Path, default=None, help="grid runs root")
    p.add_argument(
        "--scope",
        choices=["auto", "medium", "preflight", "travel"],
        default="auto",
        help="milestone for header ETA (auto detects from progress)",
    )
    p.add_argument("-n", "--interval", type=float, default=5.0, help="refresh seconds")
    p.add_argument("--once", action="store_true", help="print once and exit")
    args = p.parse_args()

    root = args.root.resolve()
    py = project_python(root)
    status_path = root / "runs" / ".plan_progress.json"

    try:
        while True:
            scope = None if args.scope == "auto" else args.scope
            data = snapshot_plan_progress(
                root,
                smoke_run=args.smoke_run,
                preflight_out=args.preflight_out,
                grid_root=args.grid_root,
                scope=scope,
            )
            try:
                status_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2))
                tmp.replace(status_path)
                (root / "runs" / "plan_status.txt").write_text(
                    f"{data.get('scope_label', '?')} {data.get('scope_pct', 0):5.1f}% | "
                    f"eta {data.get('scope_eta_s', 0)/3600:.1f}h | "
                    f"done ~{data.get('scope_completion', '?')} | next: {data.get('next_up') or 'done'}\n"
                )
            except OSError:
                pass

            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(render_plan_dashboard(data))
            sys.stdout.write(f"\n\nwatch: {py} scripts/watch_plan.py --root {root}\n")
            sys.stdout.write(f"json:  {status_path}\n")
            sys.stdout.flush()

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped)")


if __name__ == "__main__":
    main()
