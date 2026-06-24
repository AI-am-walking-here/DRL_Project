.PHONY: install setup scene-sets calibrate lint type check-constants check-spec test smoke check check-wp pipeline reproduce grid watch

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PROFILE ?= full
WP ?= 1
SEED ?= 0
CONDITION ?= full
GRID ?= configs/grid.yaml
RUN_DIR ?= runs/grid/full_seed0

export MUJOCO_GL ?= egl
export PYTHONPATH ?= src
export WANDB_MODE ?= offline
export HDF5_USE_FILE_LOCKING ?= FALSE

# ---- setup ----------------------------------------------------------------
install:
	$(PYTHON) -m pip install -e ".[dev]"

setup: install scene-sets calibrate
	@echo "setup complete. For container: docker build -t robot-routes ."

scene-sets:
	$(PYTHON) scripts/generate_scene_sets.py --profile $(PROFILE)

calibrate:
	$(PYTHON) scripts/calibrate_delta.py

# ---- quality gates --------------------------------------------------------
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
	pytest tests -q -m "not slow and not integration"

smoke:
	$(PYTHON) scripts/00_smoke.py

check: lint type check-constants check-spec test

check-wp:
	pytest tests -q -m "wp$(WP) or not wp1 and not wp2 and not wp3 and not wp4 and not wp5 and not wp6 and not wp7 and not wp8" --ignore=tests/test_integration.py 2>/dev/null || pytest tests -q
	$(PYTHON) scripts/00_smoke.py

# ---- pipeline -------------------------------------------------------------
# Single condition end-to-end (e.g. make pipeline CONDITION=full PROFILE=full).
pipeline:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition $(CONDITION) --profile $(PROFILE) --out $(or $(OUT),runs/pipeline)

reproduce:
	$(PYTHON) scripts/run_pipeline.py --seed $(SEED) --condition $(CONDITION) --profile $(PROFILE) --out $(or $(OUT),runs/reproduce)

# Core hypothesis grid: 3 conditions x 3 seeds, sequential, 3-day budget.
grid:
	PIPELINE_SKIP_PREREG=1 $(PYTHON) scripts/07_launch_grid.py --config $(GRID) --profile $(PROFILE) --runs-root $(or $(OUT),runs/grid)

watch:
	$(PYTHON) scripts/watch_pipeline.py $(RUN_DIR)
