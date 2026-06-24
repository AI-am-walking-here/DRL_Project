# robot-routes

Deep RL final project: teaching a Franka Panda arm many routes to a goal via BC, DAgger+RaC, curriculum, and PPO diversity reward.

See [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md) for the full specification.

## Quick start

```bash
# Recommended: uv + CUDA PyTorch (see pyproject.toml)
uv sync
source .venv/bin/activate
make check
make smoke
```

## Profiles

| Profile | Purpose | Wall time (4090) |
|---------|---------|------------------|
| `smoke` | CI / sanity | < 1 h |
| `medium` | Preflight learning path | ~3–5 h |
| `day` | Single-condition overnight | ~17 h |
| `full` | Core hypothesis grid (3-day budget) | ~2.5–3 days |

## Core hypothesis grid (3-day budget)

The grid tests the pre-registered hypotheses H1 (`full` vs `bc_dagger`) and H2
(`full` vs `rac_noreroute`) across 3 seeds — 9 sequential jobs on one 4090. Each
`full` job includes the PPO diversity-reward stage.

```bash
make scene-sets PROFILE=full     # one-time, expert-verified scenes (~hours, CPU)
make grid PROFILE=full           # 9-job core grid (configs/grid.yaml), sequential
```

Monitor a single run: `make watch RUN_DIR=runs/grid/full_seed0`
(or `tail -f runs/grid/*/pipeline_status.txt`).

## Single run / reproduction

```bash
make pipeline   CONDITION=full PROFILE=smoke SEED=0   # one condition, end-to-end
make reproduce  CONDITION=full PROFILE=full  SEED=0
```

## Sharing with a collaborator

Repo: **https://github.com/AI-am-walking-here/DRL_Project**

```bash
git clone https://github.com/AI-am-walking-here/DRL_Project.git
cd DRL_Project
uv sync && source .venv/bin/activate
make setup PROFILE=smoke    # scenes + calibrate for smoke
make smoke
```

Large artifacts (`runs/`, checkpoints, `.venv/`) are gitignored. Scene sets for `full` are generated locally (`make scene-sets PROFILE=full`).

## Layout

| Path | Role |
|------|------|
| `src/robot_routes/{envs,expert,agents,data,diversity,eval}` | Learning core (env, RRT expert, policies, datasets, metrics) |
| `src/robot_routes/pipeline/` | Orchestration: state, gates, resume, progress, watchdog |
| `scripts/run_pipeline.py` | Single-run DAG orchestrator |
| `scripts/07_launch_grid.py` | Sequential single-GPU grid launcher |
| `scripts/0X_*.py` | Per-stage entry points (smoke, collect, BC, DAgger, curriculum, PPO, eval) |
| `configs/` | YAML configs (env, expert, train, eval, grid) |

To grant access: GitHub repo → **Settings → Collaborators** → add your partner's GitHub username.
