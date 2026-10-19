#!/usr/bin/env bash
# One-time BFCL held-out run with the frozen configuration (configs/eval/bfcl_frozen.yaml):
# preflight -> engine (ENGINE_PROFILE=bfcl) + gateway (gateway_validated profile) -> held-out
# partition -> stop both. evals.bfcl.run refuses anything but the frozen configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SERVING=configs/serving/mac-llamacpp-bfcl-gwv.yaml

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

ENGINE_PROFILE=bfcl ./scripts/run_engine.sh > logs/engine-bfcl-test.log 2>&1 &
ENGINE_PID=$!
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8081/health | grep -q '"ok"' && break
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "engine exited; see logs/engine-bfcl-test.log" >&2; exit 1; }
  sleep 1
done

INFERSCALE_CONFIG_FILE=$SERVING uv run uvicorn --factory inferscale.api.app:app_factory \
  --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 35 > logs/gateway-bfcl-test.log 2>&1 &
GATEWAY_PID=$!
for _ in $(seq 1 60); do
  [[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/readyz)" == 200 ]] && break
  sleep 0.5
done

uv run python -m evals.bfcl.run --partition test --i-am-running-final-test --serving-profile "$SERVING" "$@"
