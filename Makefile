.PHONY: setup lint format typecheck test test-integration engine serve bench-e1 bench-smoke fetch-data split baselines eval-smoke eval-dev check-data-text

HOST ?= 127.0.0.1
PORT ?= 8000

setup:
	uv sync

lint:
	uv run ruff check src tests evals scripts bench
	uv run ruff format --check src tests evals scripts bench

format:
	uv run ruff check --fix src tests evals scripts bench
	uv run ruff format src tests evals scripts bench

typecheck:
	uv run mypy src evals scripts bench

test:
	uv run pytest

test-integration:
	uv run pytest -m integration tests/integration

engine:
	./scripts/run_engine.sh

serve:
	uv run uvicorn --factory inferscale.api.app:app_factory --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 35

bench-e1:
	uv run python -m bench.run_experiment configs/experiments/e1_baseline.yaml

bench-smoke:
	uv run python -m bench.run_experiment configs/experiments/bench_smoke.yaml

fetch-data:
	uv run python scripts/fetch_pag.py

split:
	uv run python scripts/make_pag_split.py

baselines:
	uv run python -m evals.baselines

eval-smoke:
	uv run python -m evals.run_eval --split dev --limit 10

eval-dev:
	uv run python -m evals.run_eval --split dev

check-data-text:
	uv run python scripts/check_no_dataset_text.py
