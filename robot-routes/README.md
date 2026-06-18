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
| `full` | Paper grid (via `grid-5day`) | ~5 days |

## 5-day grid (current plan)

After day preflight completes:

```bash
make plan-5day              # analyze day → write configs/handoff/full_overrides.yaml
make handoff-5day           # full scenes → launch 9-job grid unattended
# or manually:
make scene-sets PROFILE=full
PIPELINE_SKIP_PREREG=1 make grid-5day OUT=runs/grid
```

Monitor: `tail -f runs/preflight/grid_5day_handoff.log` and `runs/grid/*/pipeline_status.txt`

## Reproduction

```bash
make reproduce SEED=0 CONDITION=full PROFILE=smoke
make pipeline PROFILE=smoke SEED=0 CONDITION=full
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

To grant access: GitHub repo → **Settings → Collaborators** → add your partner's GitHub username.
