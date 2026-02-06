PYTHON ?= python3

RUNNER_FILES = \
	gate_validator.py \
	runner_common.py \
	l1_topic_research_runner.py \
	l2_l3_profile_runner.py \
	l4_hook_runner.py \
	l5_script_session_runner.py \
	l6_scene_planning_runner.py \
	l7_scene_prompt_map_runner.py \
	l8_downstream_asset_runner.py \
	profile_lock_rules.py

.PHONY: qa test test-golden golden-update lint typecheck check-ruff check-mypy

qa: lint typecheck test

test:
	$(PYTHON) -m unittest discover -s tests -q

test-golden:
	$(PYTHON) -m unittest tests.test_pipeline_golden_e2e -q

golden-update:
	UPDATE_GOLDEN=1 $(PYTHON) -m unittest tests.test_pipeline_golden_e2e -q

lint: check-ruff
	RUFF_CACHE_DIR=/tmp/ruff_cache_project_i3 $(PYTHON) -m ruff check . --select E9,F63,F7,F82

typecheck: check-mypy
	$(PYTHON) -m mypy --config-file mypy.ini --cache-dir /tmp/mypy_cache_project_i3

check-ruff:
	@$(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)" || \
		(echo "ruff is not installed. Install with: $(PYTHON) -m pip install -r requirements-dev.txt" >&2; exit 1)

check-mypy:
	@$(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('mypy') else 1)" || \
		(echo "mypy is not installed. Install with: $(PYTHON) -m pip install -r requirements-dev.txt" >&2; exit 1)
