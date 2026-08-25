# TBIE — Temporal Behavioural Intelligence Engine
#
# Every target is reproducible from `data/train/` plus the two frozen models.
# Run `make help` for the list.
#
# Windows: install GNU Make, or copy the command out of the target you want.

PYTHON      ?= python
OBS_DATE    ?= 2025-12-31
DATA_DIR    ?= ./data/train/
OUTPUT_DIR  ?= ./outputs/
PORT        ?= 8000

# -u keeps stdout unbuffered so a redirected log shows progress live rather
# than staying empty until the process exits.
PY := $(PYTHON) -u -X utf8

.DEFAULT_GOAL := help
.PHONY: help install test lint fix check predict train calibrate thresholds \
        explain drift tune cluster-search serve docker-build clean-outputs all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

install:  ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

# ── Quality gates ────────────────────────────────────────────────────────────

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

lint:  ## Lint without modifying anything
	$(PYTHON) -m ruff check .

fix:  ## Apply the safe lint fixes
	$(PYTHON) -m ruff check . --fix

check: lint test  ## Lint + tests + the model bundle contract
	$(PY) scripts/check_model_contract.py

# ── Inference ────────────────────────────────────────────────────────────────

predict:  ## Score every member at OBS_DATE (override: make predict OBS_DATE=2026-01-31)
	$(PY) pipeline.py \
		--data_dir $(DATA_DIR) \
		--observation_date $(OBS_DATE) \
		--output_dir $(OUTPUT_DIR)

# ── Training and analysis ────────────────────────────────────────────────────

train:  ## Regenerate state labels, then retrain the transition model
	$(PY) src/07_lifecycle_states.py
	$(PY) src/08_transition_prediction.py

calibrate:  ## Measure probability reliability; fit isotonic on validation
	$(PY) src/09_calibration.py

thresholds:  ## Expected-value operating point and campaign economics
	$(PY) src/10_cost_thresholds.py

explain:  ## Global + per-member SHAP attributions
	$(PY) src/11_shap_explain.py

drift:  ## Drift vs the frozen fit window (exit 1 past threshold)
	$(PY) src/12_drift_monitor.py --current $(OBS_DATE)

tune:  ## Optuna hyperparameter search (validation only; long)
	$(PY) src/13_tune_hyperparams.py --trials 25

cluster-search:  ## Silhouette vs downstream F1 sweep (long)
	$(PY) src/14_clustering_search.py --stage 1
	$(PY) src/14_clustering_search.py --stage 2

analysis: calibrate thresholds explain  ## All post-training analysis

# ── Serving ──────────────────────────────────────────────────────────────────

serve:  ## Start the API + dashboard on PORT
	TBIE_OBSERVATION_DATE=$(OBS_DATE) $(PYTHON) -m uvicorn serving.api:app \
		--port $(PORT) --reload

docker-build:  ## Build the inference image
	docker build -t tbie .

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean-outputs:  ## Delete generated outputs (models and raw data are untouched)
	rm -rf outputs/*.csv outputs/*.json outputs/*.md outputs/*.parquet outputs/*.txt

all: check predict analysis  ## Gates, then score, then analyse
