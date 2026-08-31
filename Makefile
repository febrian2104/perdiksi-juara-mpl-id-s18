.PHONY: setup audit semantic-audit normalize canonicalize quality-report eda prediction-policy analysis test lint format dashboard

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

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

dashboard:
	.venv/bin/streamlit run src/mpl_predictor/dashboard.py
