#!/usr/bin/env bash
# Start llama-server (llama.cpp build 11669) for InferScale. All values overridable via env.
set -euo pipefail

# Optional profile (configs/engine/<name>.env) supplies defaults; explicit ENGINE_* env still wins.
ENGINE_PROFILE="${ENGINE_PROFILE:-default}"
PROFILE_FILE="$(dirname "$0")/../configs/engine/${ENGINE_PROFILE}.env"
if [[ -f "$PROFILE_FILE" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ -z "${!key:-}" ]] && export "$key=$value"
  done < "$PROFILE_FILE"
else
  echo "unknown engine profile: $ENGINE_PROFILE ($PROFILE_FILE)" >&2
  exit 1
fi
echo "engine profile: $ENGINE_PROFILE (ctx=${ENGINE_CTX} parallel=${ENGINE_PARALLEL} reasoning=${ENGINE_REASONING} cache_ram=${ENGINE_CACHE_RAM:-1024})" >&2

ENGINE_BIN="${ENGINE_BIN:-llama-server}"
ENGINE_MODEL="${ENGINE_MODEL:-$HOME/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf}"
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-8081}"
ENGINE_CTX="${ENGINE_CTX:-8192}"
ENGINE_PARALLEL="${ENGINE_PARALLEL:-2}"
ENGINE_GPU_LAYERS="${ENGINE_GPU_LAYERS:-all}"
ENGINE_REASONING="${ENGINE_REASONING:-off}"
# Host-RAM prompt cache (llama.cpp --cache-ram, MiB). llama.cpp defaults to 8192, which on a
# 16 GB Mac pushes the system into swap next to the model and KV cache; we bound it.
ENGINE_CACHE_RAM="${ENGINE_CACHE_RAM:-1024}"

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
  --cache-ram "$ENGINE_CACHE_RAM" \
  "$@"
