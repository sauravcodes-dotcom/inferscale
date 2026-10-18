.PHONY: setup lint format typecheck test test-integration engine serve eval-smoke

HOST ?= 127.0.0.1
PORT ?= 8000

setup:
	uv sync

lint:
	uv run ruff check src tests evals scripts
	uv run ruff format --check src tests evals scripts

format:
	uv run ruff check --fix src tests evals scripts
	uv run ruff format src tests evals scripts

typecheck:
	uv run mypy src evals scripts

test:
	uv run pytest

test-integration:
	uv run pytest -m integration tests/integration

engine:
	./scripts/run_engine.sh

serve:
	uv run uvicorn --factory inferscale.api.app:app_factory --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 35

eval-smoke:
	uv run python -m evals.run_eval --split dev --limit 5 --response-format $(or $(RF),none)
