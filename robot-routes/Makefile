.PHONY: lint type check check-constants check-spec test smoke check-wp reproduce pipeline grid grid-7day grid-5day install setup scene-sets preflight test-medium test-day prep-full watch watch-plan handoff-day handoff-grid analyze-day plan-5day handoff-5day

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PROFILE ?= full
WP ?= 1
SEED ?= 0
CONDITION ?= full
PREFLIGHT_OUT ?= runs/preflight
RUN_DIR ?= runs/preflight/rac_noreroute_seed0

export MUJOCO_GL ?= egl
export PYTHONPATH ?= src
export WANDB_MODE ?= offline
export HDF5_USE_FILE_LOCKING ?= FALSE

install:
	$(PYTHON) -m pip install -e ".[dev]"

setup: install scene-sets calibrate
	@echo "setup complete (local). For container: docker build -t robot-routes ."

preflight:
	$(PYTHON) scripts/preflight.py

# Medium preflight (~3–5 h): core learning path without grid deps.
test-medium:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition rac_noreroute --profile medium --out $(PREFLIGHT_OUT) --device auto

# Day preflight (~14–18 h): near-full single seed; exercises PPO + resume.
test-day:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition full --profile day --out $(PREFLIGHT_OUT) --device auto

# Before travel: generate full scenes (hours) then launch grid when tagged.
prep-full:
	@echo "Step 1: make scene-sets PROFILE=full   # ~2–4 h CPU"
	@echo "Step 2: git tag prereg-v1              # required for full grid"
	@echo "Step 3: PIPELINE_SKIP_PREREG=1 make grid PROFILE=full OUT=runs/grid"

watch:
	PYTHONPATH=src $(PYTHON) scripts/watch_pipeline.py $(RUN_DIR)

watch-plan:
	PYTHONPATH=src $(PYTHON) scripts/watch_plan.py --root .

# Unattended: wait for medium → review → launch day (~14–18 h).
handoff-day:
	nohup $(PYTHON) scripts/autonomous_handoff.py >> runs/preflight/handoff.log 2>&1 & echo $$! > runs/preflight/handoff.pid && echo "handoff pid=$$(cat runs/preflight/handoff.pid) log=runs/preflight/handoff.log"

scene-sets:
	PYTHONPATH=src MUJOCO_GL=$(MUJOCO_GL) $(PYTHON) scripts/generate_scene_sets.py --profile $(PROFILE)

calibrate:
	PYTHONPATH=src $(PYTHON) scripts/calibrate_delta.py

lint:
	ruff check src tests scripts
	black --check src tests scripts

type:
	mypy src/robot_routes --ignore-missing-imports

check-constants:
	$(PYTHON) scripts/check_constants.py

check-spec:
	$(PYTHON) scripts/check_spec_constants.py

test:
	PYTHONPATH=src MUJOCO_GL=$(MUJOCO_GL) pytest tests -q -m "not slow and not integration"

smoke:
	PYTHONPATH=src MUJOCO_GL=$(MUJOCO_GL) $(PYTHON) scripts/00_smoke.py

check: lint type check-constants check-spec test

check-wp:
	pytest tests -q -m "wp$(WP) or not wp1 and not wp2 and not wp3 and not wp4 and not wp5 and not wp6 and not wp7 and not wp8" --ignore=tests/test_integration.py 2>/dev/null || pytest tests -q
	$(PYTHON) scripts/00_smoke.py

reproduce:
	$(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition $(CONDITION) --profile $(PROFILE) --out $(or $(OUT),runs/reproduce)

pipeline:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition $(CONDITION) --profile $(PROFILE) --out $(or $(OUT),runs/pipeline)

grid:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/07_launch_grid.py --profile $(PROFILE) --runs-root $(or $(OUT),runs/grid)

# 9-run core grid for 7-day window (H1/H2 conditions × 3 seeds).
grid-7day:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/07_launch_grid.py --config configs/grid_7day.yaml --profile full --runs-root $(or $(OUT),runs/grid)

# 5-day calibrated grid (same 9 jobs; PROFILE=full uses configs/handoff/full_overrides.yaml).
grid-5day:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/07_launch_grid.py --config configs/grid_7day.yaml --profile full --runs-root $(or $(OUT),runs/grid)

plan-5day:
	$(PYTHON) scripts/plan_5day_grid.py

# Unattended: full scenes → 5-day grid (~114h sequential on 4090).
handoff-5day:
	nohup $(PYTHON) scripts/grid_5day_handoff.py >> runs/preflight/grid_5day_handoff.log 2>&1 & echo $$! > runs/preflight/grid_5day_handoff.pid && echo "handoff-5day pid=$$(cat runs/preflight/grid_5day_handoff.pid)"

# Wait for day → analyze → write configs/handoff/full_overrides.yaml (no grid launch).
handoff-grid:
	nohup $(PYTHON) scripts/day_to_grid_handoff.py >> runs/preflight/day_to_grid.log 2>&1 & echo $$! > runs/preflight/day_to_grid.pid && echo "handoff-grid pid=$$(cat runs/preflight/day_to_grid.pid)"

# One-shot if day already finished.
analyze-day:
	$(PYTHON) scripts/day_to_grid_handoff.py --skip-wait
