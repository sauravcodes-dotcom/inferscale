.PHONY: setup lint format typecheck test test-integration engine serve bench-e1 bench-smoke fetch-data split baselines eval-smoke eval-dev check-data-text freeze-prompt eval-test preflight bfcl-setup engine-bfcl serve-bfcl bfcl-test reliability bfcl-requests m43 m43-smoke

HOST ?= 127.0.0.1
PORT ?= 8000

setup:
	uv sync

lint:
	uv run ruff check src tests evals scripts bench tools/bfcl
	uv run ruff format --check src tests evals scripts bench tools/bfcl

format:
	uv run ruff check --fix src tests evals scripts bench tools/bfcl
	uv run ruff format src tests evals scripts bench tools/bfcl

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

# Freeze a prompt for the held-out test split, e.g.
#   make freeze-prompt PROMPT=pag_v1.p2 REASON="..."
freeze-prompt:
	@test -n "$(PROMPT)" || (echo "usage: make freeze-prompt PROMPT=pag_v1.p1 REASON='...'" && exit 1)
	uv run python -m evals.freeze --prompt $(PROMPT) --reason "$(or $(REASON),frozen via make freeze-prompt)"

preflight:
	uv run python scripts/preflight.py

# Final held-out test run (needs: fresh restart, AC power, browsers closed). Stops both servers.
eval-test:
	./scripts/eval_test.sh

# --- BFCL (isolated environment, pinned and hash-locked) ---
bfcl-setup:
	uv venv tools/bfcl/.venv --python 3.12 --allow-existing -q
	VIRTUAL_ENV=tools/bfcl/.venv uv pip sync tools/bfcl/requirements.lock --require-hashes

engine-bfcl:
	ENGINE_PROFILE=bfcl ./scripts/run_engine.sh

serve-bfcl:
	INFERSCALE_CONFIG_FILE=configs/serving/mac-llamacpp-bfcl.yaml $(MAKE) serve

# One-time BFCL held-out run with the frozen config (needs restart, AC power, no browsers).
bfcl-test:
	./scripts/bfcl_test.sh

# Reliability validation against the real engine (starts/stops its own processes).
reliability:
	uv run python -m bench.reliability

# M4.3 inference sweeps (start/stop their own engine + gateway; need ports 8000/8081 free).
bfcl-requests:
	tools/bfcl/.venv/bin/python tools/bfcl/export_requests.py --partition dev

# Clean environment required (restart, AC, no browsers). Order matters: the cache sweep's 8 GiB
# variant is expected to swap, so it runs last. ~4.5 h in total. Start it with
#   uv run python scripts/detach.py logs/m43-run.log -- make m43
# so it survives the launching session; an interrupted sweep continues with
#   uv run python -m bench.sweep <config> --resume results/bench/<sweep>/<ts>.json
m43: bfcl-requests
	uv run python -m bench.sweep configs/sweeps/m43_slots.yaml
	uv run python -m bench.sweep configs/sweeps/m43_quant.yaml
	uv run python -m bench.sweep configs/sweeps/m43_cache_ram.yaml

m43-smoke: bfcl-requests
	uv run python -m bench.sweep configs/sweeps/m43_smoke.yaml
