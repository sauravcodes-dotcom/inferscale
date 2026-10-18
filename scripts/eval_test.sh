#!/usr/bin/env bash
# Final held-out test run: preflight -> start engine + gateway -> run_eval --split test -> stop both.
# Uses the frozen prompt (configs/eval/frozen.yaml); run_eval refuses anything else.
# Extra arguments are passed to run_eval (e.g. --concurrency is NOT recommended for quality runs).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

uv run python scripts/preflight.py

ENGINE_PID=""
GATEWAY_PID=""
cleanup() {
  [[ -n "$GATEWAY_PID" ]] && kill -TERM "$GATEWAY_PID" 2>/dev/null || true
  [[ -n "$ENGINE_PID" ]] && kill -TERM "$ENGINE_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "stopped engine and gateway"
}
trap cleanup EXIT

./scripts/run_engine.sh > logs/engine-test.log 2>&1 &
ENGINE_PID=$!
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8081/health | grep -q '"ok"' && break
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine exited; see logs/engine-test.log" >&2; exit 1; }
  sleep 1
done
curl -sf http://127.0.0.1:8081/health | grep -q '"ok"' || { echo "engine not healthy after 120 s" >&2; exit 1; }

uv run uvicorn --factory inferscale.api.app:app_factory --host 127.0.0.1 --port 8000 \
  --timeout-graceful-shutdown 35 > logs/gateway-test.log 2>&1 &
GATEWAY_PID=$!
for _ in $(seq 1 60); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/readyz)" == 200 ]] && break
  kill -0 "$GATEWAY_PID" 2>/dev/null || { echo "gateway exited; see logs/gateway-test.log" >&2; exit 1; }
  sleep 0.5
done

uv run python -m evals.run_eval --split test --i-am-running-final-test "$@"
