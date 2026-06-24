#!/usr/bin/env bash
# MuJoCo needs system libstdc++ (conda/anaconda lib breaks Mesa GL drivers).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export DISPLAY="${DISPLAY:-:0}"
export MUJOCO_GL=glfw
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
exec .venv/bin/python scripts/render_mujoco_showcase.py "$@"
