DELIVERABLE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CAPSTONE_UPSTREAM_CACHE_DIR ?= $(DELIVERABLE_DIR)/.private_cache
CAPSTONE_MODULE24_CACHE_DIR ?= $(DELIVERABLE_DIR)/.cache/module24
export CAPSTONE_UPSTREAM_CACHE_DIR
export CAPSTONE_MODULE24_CACHE_DIR
export PYTHONDONTWRITEBYTECODE := 1

UV_RUN = UV_CACHE_DIR=/tmp/uv-cache uv --project "$(DELIVERABLE_DIR)" run
NOTEBOOK_DIR := $(DELIVERABLE_DIR)/notebooks
KERNEL_PREFIX := $(DELIVERABLE_DIR)/.jupyter
KERNEL_NAME := module24-final
JUPYTER_PATH := $(KERNEL_PREFIX)/share/jupyter
EXECUTE = JUPYTER_PATH="$(JUPYTER_PATH)" $(UV_RUN) jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=$(KERNEL_NAME)

.PHONY: setup kernel notebooks sync test lint privacy artifacts public-check provenance package validate full-check

setup:
	UV_CACHE_DIR=/tmp/uv-cache uv sync --project "$(DELIVERABLE_DIR)" --locked

kernel: setup
	$(UV_RUN) python -m ipykernel install --prefix "$(KERNEL_PREFIX)" --name "$(KERNEL_NAME)" --display-name "Module 24 final"

notebooks: kernel
	cd "$(NOTEBOOK_DIR)" && $(EXECUTE) 05_peer_strategy_baseline.ipynb
	cd "$(NOTEBOOK_DIR)" && $(EXECUTE) 06_hurdle_cadence_model.ipynb
	cd "$(NOTEBOOK_DIR)" && $(EXECUTE) 07_alert_episodes_and_pilot.ipynb
	cd "$(NOTEBOOK_DIR)" && $(EXECUTE) 99_final_findings.ipynb

sync:
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/sync_metrics.py"

test:
	$(UV_RUN) pytest "$(DELIVERABLE_DIR)/tests" -q -p no:cacheprovider

lint:
	$(UV_RUN) ruff check "$(DELIVERABLE_DIR)/src" "$(DELIVERABLE_DIR)/tests"

privacy:
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/check_privacy.py"

artifacts:
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/validate_artifacts.py"

public-check: setup lint privacy artifacts
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/validate_artifacts.py" --manifest

provenance: artifacts
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/generate_provenance_manifest.py"

validate: sync test lint privacy provenance
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/validate_artifacts.py" --manifest

package: validate
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/package_submission.py"
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/check_privacy.py" --archive "$(DELIVERABLE_DIR)/dist/module24_deliverable.zip"
	$(UV_RUN) python "$(DELIVERABLE_DIR)/src/validate_artifacts.py" --manifest --archive "$(DELIVERABLE_DIR)/dist/module24_deliverable.zip"

full-check: notebooks package
