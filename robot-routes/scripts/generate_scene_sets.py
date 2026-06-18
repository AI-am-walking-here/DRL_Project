#!/usr/bin/env python3
"""Generate frozen scene sets + manifest (§10.2, §11.7.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_routes.data.scene_sets import generate_all, generate_pool_rl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="smoke", choices=["smoke", "medium", "day", "full"])
    p.add_argument("--root", default=".")
    p.add_argument(
        "--status-file",
        default="data/scenes/.generation_status.json",
        help="JSON snapshot updated during run (watch with: watch -n1 cat …)",
    )
    args = p.parse_args()
    root = Path(args.root).resolve()
    status = root / args.status_file
    print(f"generating scene sets profile={args.profile} → {root / 'data/scenes/'}")
    print(f"status file → {status}")
    print(f"  watch -n1 cat {status}")
    manifest = generate_all(root, profile=args.profile, status_path=status)
    pool_counts = {"smoke": 16, "medium": 64, "day": 128, "full": 256}
    pool_count = pool_counts.get(args.profile, 256)
    pool_hash = generate_pool_rl(
        root, count=pool_count, profile=args.profile, status_path=status
    )
    print(f"manifest → {root / 'data/scenes/manifest.json'}")
    print(f"pool_rl sha256={pool_hash[:16]}... ({pool_count} requested)")


if __name__ == "__main__":
    main()
