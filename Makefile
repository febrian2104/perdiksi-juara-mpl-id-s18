.PHONY: setup audit semantic-audit normalize canonicalize quality-report eda prediction-policy analysis build-features baseline modeling build-match-features backtest models sync-season18 train-final simulate-season18 update-predictions explain-season18 update-season18 season18 verify docs-check local-train local-update local-pipeline test lint format dashboard

OBSERVED_AT ?=

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-dev.txt
	.venv/bin/python -m pip install --no-deps --editable .

audit:
	.venv/bin/mpl-predictor audit

semantic-audit:
	.venv/bin/mpl-predictor semantic-audit

normalize:
	.venv/bin/mpl-predictor normalize

canonicalize:
	.venv/bin/mpl-predictor canonicalize

quality-report:
	.venv/bin/mpl-predictor quality-report

eda:
	.venv/bin/mpl-predictor eda

prediction-policy:
	.venv/bin/mpl-predictor prediction-policy

analysis: quality-report eda prediction-policy

build-features:
	.venv/bin/mpl-predictor build-features

baseline:
	.venv/bin/mpl-predictor baseline

modeling: build-features baseline

build-match-features:
	.venv/bin/mpl-predictor build-match-features

backtest:
	.venv/bin/mpl-predictor backtest

models: modeling build-match-features backtest

sync-season18:
	.venv/bin/mpl-predictor sync-season18 $(if $(OBSERVED_AT),--observed-at $(OBSERVED_AT),)

train-final:
	.venv/bin/mpl-predictor train-final

simulate-season18:
	.venv/bin/mpl-predictor simulate-season18

update-predictions:
	.venv/bin/mpl-predictor update-season18-predictions

explain-season18:
	.venv/bin/mpl-predictor explain-season18

update-season18: sync-season18 train-final update-predictions explain-season18

season18: update-season18

docs-check:
	test -s README.md
	test -s docs/OPERATIONS.md

verify: audit semantic-audit lint test docs-check

local-train:
	./scripts/run_local_pipeline.sh train

local-update:
	./scripts/run_local_pipeline.sh update $(if $(OBSERVED_AT),$(OBSERVED_AT),)

local-pipeline:
	./scripts/run_local_pipeline.sh all $(if $(OBSERVED_AT),$(OBSERVED_AT),)

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

dashboard:
	.venv/bin/streamlit run src/mpl_predictor/dashboard.py
