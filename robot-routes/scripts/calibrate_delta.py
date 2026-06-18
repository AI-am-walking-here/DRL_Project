#!/usr/bin/env python3
"""Live δ_distinct calibration with progress (§9.1, §11.7.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.pipeline.calibration import gate_cal, run_calibration


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument(
        "--status-file",
        default="calibration/.calibration_status.json",
        help="JSON snapshot updated during run (watch with: watch -n1 cat …)",
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    status = root / args.status_file
    print(f"calibrating δ_distinct → {root / 'calibration/delta.json'}")
    print(f"status file → {status}")
    payload = run_calibration(root, status_path=status)
    ok, msg = gate_cal(payload)
    print(
        f"delta_distinct={payload['delta_distinct']:.4f}  "
        f"same_p95={payload.get('same_p95', 0):.4f}  "
        f"cross_p5={payload.get('cross_p5', 0):.4f}  "
        f"G-CAL={'PASS' if ok else 'FAIL'} ({msg})"
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
