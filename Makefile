.PHONY: setup lint format typecheck test test-integration engine serve

HOST ?= 127.0.0.1
PORT ?= 8000

setup:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

test-integration:
	uv run pytest -m integration tests/integration

engine:
	./scripts/run_engine.sh

serve:
	uv run uvicorn --factory inferscale.api.app:app_factory --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 35
