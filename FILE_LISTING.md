# `/testing` file listing

Generated from the cluster workspace. Large generated trees are summarized.

## Summary

| Path | Role |
|------|------|
| `TECHNICAL_SPEC.md` | Project specification |
| `robot-routes/` | DRL pipeline codebase ([DRL_Project](https://github.com/AI-am-walking-here/DRL_Project)) |
| `.git/` | Repository metadata (root-level; `robot-routes/` is no longer a separate git repo) |
| `robot-routes/.venv/` | Local Python venv (~4.9 GB, **do not commit**) |
| `robot-routes/runs/` | Pipeline artifacts (~1.2 GB, **do not commit**) |

Approximate total size on disk: **6.02 GB** (29,610 files).

### `.venv/` (summarized)

- **28,935** files, **2,457** directories, **4.83 GB**
- Omitted from the detailed listing below.

| Entry | Type |
|-------|------|
| `robot-routes/.venv/.gitignore` | file |
| `robot-routes/.venv/.lock` | file |
| `robot-routes/.venv/CACHEDIR.TAG` | file |
| `robot-routes/.venv/bin/` | dir (31 files) |
| `robot-routes/.venv/lib/` | dir (28,898 files) |
| `robot-routes/.venv/lib64/` | dir (28,898 files) |
| `robot-routes/.venv/pyvenv.cfg` | file |
| `robot-routes/.venv/share/` | dir (2 files) |

### `runs/` (summarized)

- **122** files, **35** directories, **1.11 GB**
- Omitted from the detailed listing below.

| Entry | Type |
|-------|------|
| `robot-routes/runs/grid/` | dir (60 files) |
| `robot-routes/runs/pipeline/` | dir (62 files) |

## Full file listing (excluding `.venv/` and `runs/`)

```
TECHNICAL_SPEC.md
robot-routes/.cursor/rules
.github/workflows/ci.yml
robot-routes/.gitignore
robot-routes/.mypy_cache/.gitignore
robot-routes/.mypy_cache/3.10/cache.0.db
robot-routes/.mypy_cache/3.10/cache.1.db
robot-routes/.mypy_cache/3.10/cache.10.db
robot-routes/.mypy_cache/3.10/cache.11.db
robot-routes/.mypy_cache/3.10/cache.12.db
robot-routes/.mypy_cache/3.10/cache.13.db
robot-routes/.mypy_cache/3.10/cache.14.db
robot-routes/.mypy_cache/3.10/cache.15.db
robot-routes/.mypy_cache/3.10/cache.2.db
robot-routes/.mypy_cache/3.10/cache.3.db
robot-routes/.mypy_cache/3.10/cache.4.db
robot-routes/.mypy_cache/3.10/cache.5.db
robot-routes/.mypy_cache/3.10/cache.6.db
robot-routes/.mypy_cache/3.10/cache.7.db
robot-routes/.mypy_cache/3.10/cache.8.db
robot-routes/.mypy_cache/3.10/cache.9.db
robot-routes/.mypy_cache/CACHEDIR.TAG
robot-routes/.pytest_cache/.gitignore
robot-routes/.pytest_cache/CACHEDIR.TAG
robot-routes/.pytest_cache/README.md
robot-routes/.pytest_cache/v/cache/lastfailed
robot-routes/.pytest_cache/v/cache/nodeids
robot-routes/.ruff_cache/.gitignore
robot-routes/.ruff_cache/0.15.17/13629899778492707436
robot-routes/.ruff_cache/0.15.17/14810835238189452651
robot-routes/.ruff_cache/0.15.17/14966821655193529280
robot-routes/.ruff_cache/0.15.17/17002152189653858269
robot-routes/.ruff_cache/0.15.17/2687662880254650944
robot-routes/.ruff_cache/0.15.17/7652697723522003430
robot-routes/.ruff_cache/CACHEDIR.TAG
robot-routes/BLOCKED.md
robot-routes/Dockerfile
robot-routes/ISSUES.md
robot-routes/Makefile
robot-routes/README.md
robot-routes/TASKS/wp1.yaml
robot-routes/TASKS/wp2.yaml
robot-routes/TASKS/wp3.yaml
robot-routes/TASKS/wp4.yaml
robot-routes/TASKS/wp5.yaml
robot-routes/TASKS/wp6.yaml
robot-routes/TASKS/wp7.yaml
robot-routes/TASKS/wp8.yaml
robot-routes/TECHNICAL_SPEC.md
robot-routes/calibration/.calibration_status.json
robot-routes/calibration/delta.json
robot-routes/configs/env/panda_reach.yaml
robot-routes/configs/eval/default.yaml
robot-routes/configs/expert/rrt_connect.yaml
robot-routes/configs/grid.yaml
robot-routes/configs/pipeline.yaml
robot-routes/configs/train/bc.yaml
robot-routes/configs/train/curriculum.yaml
robot-routes/configs/train/dagger_rac.yaml
robot-routes/configs/train/rl_diversity.yaml
robot-routes/data/pool_rl.json
robot-routes/data/scenes/manifest.json
robot-routes/data/scenes/test_L0.json
robot-routes/data/scenes/test_L1.json
robot-routes/data/scenes/test_L2.json
robot-routes/data/scenes/test_L3.json
robot-routes/data/scenes/test_unseen.json
robot-routes/data/scenes/val_L0.json
robot-routes/data/scenes/val_L1.json
robot-routes/data/scenes/val_L2.json
robot-routes/data/scenes/val_L3.json
robot-routes/data/scenes/val_unseen.json
robot-routes/pyproject.toml
robot-routes/scripts/00_smoke.py
robot-routes/scripts/01_collect_bc_demos.py
robot-routes/scripts/02_train_bc.py
robot-routes/scripts/03_run_dagger_rac.py
robot-routes/scripts/04_run_curriculum.py
robot-routes/scripts/05_train_rl_diversity.py
robot-routes/scripts/06_evaluate.py
robot-routes/scripts/07_launch_grid.py
robot-routes/scripts/__pycache__/01_collect_bc_demos.cpython-310.pyc
robot-routes/scripts/__pycache__/02_train_bc.cpython-310.pyc
robot-routes/scripts/__pycache__/03_run_dagger_rac.cpython-310.pyc
robot-routes/scripts/__pycache__/04_run_curriculum.cpython-310.pyc
robot-routes/scripts/__pycache__/05_train_rl_diversity.cpython-310.pyc
robot-routes/scripts/__pycache__/run_pipeline.cpython-310.pyc
robot-routes/scripts/__pycache__/watch_pipeline.cpython-310.pyc
robot-routes/scripts/calibrate_delta.py
robot-routes/scripts/check_constants.py
robot-routes/scripts/check_spec_constants.py
robot-routes/scripts/generate_scene_sets.py
robot-routes/scripts/run_pipeline.py
robot-routes/scripts/watch_pipeline.py
robot-routes/src/robot_routes/__init__.py
robot-routes/src/robot_routes/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/__pycache__/contracts.cpython-310.pyc
robot-routes/src/robot_routes/agents/__init__.py
robot-routes/src/robot_routes/agents/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/agents/__pycache__/bc_trainer.cpython-310.pyc
robot-routes/src/robot_routes/agents/__pycache__/dagger_rac.cpython-310.pyc
robot-routes/src/robot_routes/agents/__pycache__/policy.cpython-310.pyc
robot-routes/src/robot_routes/agents/__pycache__/ppo_diversity.cpython-310.pyc
robot-routes/src/robot_routes/agents/bc_trainer.py
robot-routes/src/robot_routes/agents/dagger_rac.py
robot-routes/src/robot_routes/agents/policy.py
robot-routes/src/robot_routes/agents/ppo_diversity.py
robot-routes/src/robot_routes/contracts.py
robot-routes/src/robot_routes/data/__init__.py
robot-routes/src/robot_routes/data/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/data/__pycache__/buffer.cpython-310.pyc
robot-routes/src/robot_routes/data/__pycache__/scene_sets.cpython-310.pyc
robot-routes/src/robot_routes/data/__pycache__/schema.cpython-310.pyc
robot-routes/src/robot_routes/data/buffer.py
robot-routes/src/robot_routes/data/collect_demos.py
robot-routes/src/robot_routes/data/merge_shards.py
robot-routes/src/robot_routes/data/scene_sets.py
robot-routes/src/robot_routes/data/schema.py
robot-routes/src/robot_routes/diversity/__init__.py
robot-routes/src/robot_routes/diversity/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/diversity/__pycache__/route_metrics.cpython-310.pyc
robot-routes/src/robot_routes/diversity/route_metrics.py
robot-routes/src/robot_routes/envs/__init__.py
robot-routes/src/robot_routes/envs/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/envs/__pycache__/panda_reach_env.cpython-310.pyc
robot-routes/src/robot_routes/envs/__pycache__/scene_gen.cpython-310.pyc
robot-routes/src/robot_routes/envs/assets/assets/finger_0.obj
robot-routes/src/robot_routes/envs/assets/assets/finger_1.obj
robot-routes/src/robot_routes/envs/assets/assets/hand.stl
robot-routes/src/robot_routes/envs/assets/assets/hand_0.obj
robot-routes/src/robot_routes/envs/assets/assets/hand_1.obj
robot-routes/src/robot_routes/envs/assets/assets/hand_2.obj
robot-routes/src/robot_routes/envs/assets/assets/hand_3.obj
robot-routes/src/robot_routes/envs/assets/assets/hand_4.obj
robot-routes/src/robot_routes/envs/assets/assets/link0.stl
robot-routes/src/robot_routes/envs/assets/assets/link0_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_10.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_11.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_3.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_4.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_5.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_7.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_8.obj
robot-routes/src/robot_routes/envs/assets/assets/link0_9.obj
robot-routes/src/robot_routes/envs/assets/assets/link1.obj
robot-routes/src/robot_routes/envs/assets/assets/link1.stl
robot-routes/src/robot_routes/envs/assets/assets/link2.obj
robot-routes/src/robot_routes/envs/assets/assets/link2.stl
robot-routes/src/robot_routes/envs/assets/assets/link3.stl
robot-routes/src/robot_routes/envs/assets/assets/link3_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link3_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link3_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link3_3.obj
robot-routes/src/robot_routes/envs/assets/assets/link4.stl
robot-routes/src/robot_routes/envs/assets/assets/link4_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link4_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link4_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link4_3.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_collision_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_collision_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link5_collision_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link6.stl
robot-routes/src/robot_routes/envs/assets/assets/link6_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_10.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_11.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_12.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_13.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_14.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_15.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_16.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_3.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_4.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_5.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_6.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_7.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_8.obj
robot-routes/src/robot_routes/envs/assets/assets/link6_9.obj
robot-routes/src/robot_routes/envs/assets/assets/link7.stl
robot-routes/src/robot_routes/envs/assets/assets/link7_0.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_1.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_2.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_3.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_4.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_5.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_6.obj
robot-routes/src/robot_routes/envs/assets/assets/link7_7.obj
robot-routes/src/robot_routes/envs/assets/panda.xml
robot-routes/src/robot_routes/envs/assets/reach_scene.xml
robot-routes/src/robot_routes/envs/panda_reach_env.py
robot-routes/src/robot_routes/envs/scene_gen.py
robot-routes/src/robot_routes/eval/__init__.py
robot-routes/src/robot_routes/eval/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/eval/__pycache__/evaluate.cpython-310.pyc
robot-routes/src/robot_routes/eval/__pycache__/plots.cpython-310.pyc
robot-routes/src/robot_routes/eval/evaluate.py
robot-routes/src/robot_routes/eval/plots.py
robot-routes/src/robot_routes/expert/__init__.py
robot-routes/src/robot_routes/expert/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/expert/__pycache__/collision.cpython-310.pyc
robot-routes/src/robot_routes/expert/__pycache__/oracle.cpython-310.pyc
robot-routes/src/robot_routes/expert/__pycache__/rrt_connect.cpython-310.pyc
robot-routes/src/robot_routes/expert/collision.py
robot-routes/src/robot_routes/expert/oracle.py
robot-routes/src/robot_routes/expert/rrt_connect.py
robot-routes/src/robot_routes/pipeline/__init__.py
robot-routes/src/robot_routes/pipeline/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/artifacts.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/bc_ladder.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/calibration.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/conditions.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/dagger_resume.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/dither.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/g_ppo.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/gates.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/notify.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/progress.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/setup_checks.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/stage_progress.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/stage_resume.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/state_merge.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/verdicts.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/videos.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/__pycache__/watchdog.cpython-310.pyc
robot-routes/src/robot_routes/pipeline/artifacts.py
robot-routes/src/robot_routes/pipeline/bc_ladder.py
robot-routes/src/robot_routes/pipeline/calibration.py
robot-routes/src/robot_routes/pipeline/conditions.py
robot-routes/src/robot_routes/pipeline/dagger_resume.py
robot-routes/src/robot_routes/pipeline/dither.py
robot-routes/src/robot_routes/pipeline/g_ppo.py
robot-routes/src/robot_routes/pipeline/gates.py
robot-routes/src/robot_routes/pipeline/notify.py
robot-routes/src/robot_routes/pipeline/progress.py
robot-routes/src/robot_routes/pipeline/setup_checks.py
robot-routes/src/robot_routes/pipeline/stage_progress.py
robot-routes/src/robot_routes/pipeline/stage_resume.py
robot-routes/src/robot_routes/pipeline/state_merge.py
robot-routes/src/robot_routes/pipeline/verdicts.py
robot-routes/src/robot_routes/pipeline/videos.py
robot-routes/src/robot_routes/pipeline/watchdog.py
robot-routes/src/robot_routes/utils/__init__.py
robot-routes/src/robot_routes/utils/__pycache__/__init__.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/config.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/device.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/gpu_alloc.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/mj_state.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/progress.cpython-310.pyc
robot-routes/src/robot_routes/utils/__pycache__/seeding.cpython-310.pyc
robot-routes/src/robot_routes/utils/config.py
robot-routes/src/robot_routes/utils/device.py
robot-routes/src/robot_routes/utils/gpu_alloc.py
robot-routes/src/robot_routes/utils/logging.py
robot-routes/src/robot_routes/utils/mj_state.py
robot-routes/src/robot_routes/utils/progress.py
robot-routes/src/robot_routes/utils/seeding.py
robot-routes/tests/__init__.py
robot-routes/tests/__pycache__/__init__.cpython-310.pyc
robot-routes/tests/__pycache__/test_api_assumptions.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_api_assumptions.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_contracts.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_contracts.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_dagger_resume.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_device.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_diversity_metrics.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_diversity_metrics.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_env.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_env.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_expert.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_expert.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_gates.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_gpu_alloc.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_gpu_alloc.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_integration.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_integration.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_pipeline_progress.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_pipeline_progress.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_ppo.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_ppo.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_progress.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_progress.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_rewind.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_rewind.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_scene_sets.cpython-310-pytest-9.0.3.pyc
robot-routes/tests/__pycache__/test_scene_sets.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_stage_progress.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_stage_resume.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/__pycache__/test_watchdog.cpython-310-pytest-9.1.0.pyc
robot-routes/tests/test_api_assumptions.py
robot-routes/tests/test_contracts.py
robot-routes/tests/test_dagger_resume.py
robot-routes/tests/test_device.py
robot-routes/tests/test_diversity_metrics.py
robot-routes/tests/test_env.py
robot-routes/tests/test_expert.py
robot-routes/tests/test_gates.py
robot-routes/tests/test_gpu_alloc.py
robot-routes/tests/test_integration.py
robot-routes/tests/test_pipeline_progress.py
robot-routes/tests/test_ppo.py
robot-routes/tests/test_progress.py
robot-routes/tests/test_rewind.py
robot-routes/tests/test_scene_sets.py
robot-routes/tests/test_stage_progress.py
robot-routes/tests/test_stage_resume.py
robot-routes/tests/test_watchdog.py
robot-routes/uv.lock
```

**553** paths listed.
