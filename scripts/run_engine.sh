#!/usr/bin/env bash
# Start llama-server (llama.cpp build 11669) for InferScale. All values overridable via env.
set -euo pipefail

ENGINE_BIN="${ENGINE_BIN:-llama-server}"
ENGINE_MODEL="${ENGINE_MODEL:-$HOME/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf}"
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-8081}"
ENGINE_CTX="${ENGINE_CTX:-8192}"
ENGINE_PARALLEL="${ENGINE_PARALLEL:-2}"
ENGINE_GPU_LAYERS="${ENGINE_GPU_LAYERS:-all}"
ENGINE_REASONING="${ENGINE_REASONING:-off}"

if [[ ! -f "$ENGINE_MODEL" ]]; then
  echo "model not found: $ENGINE_MODEL" >&2
  exit 1
fi

exec "$ENGINE_BIN" \
  -m "$ENGINE_MODEL" \
  --host "$ENGINE_HOST" \
  --port "$ENGINE_PORT" \
  -c "$ENGINE_CTX" \
  -np "$ENGINE_PARALLEL" \
  -ngl "$ENGINE_GPU_LAYERS" \
  --metrics \
  --reasoning "$ENGINE_REASONING" \
  "$@"
