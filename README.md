# InferScale

A minimal LLM inference gateway in front of a local llama.cpp engine (`llama-server`).
It exposes an OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming) and adds:

- **Admission control**: at most `max_inflight` requests on the engine (= engine slots), a
  bounded wait queue, and an immediate `429` + `Retry-After` when full.
- **Deadlines**: connect, first-token and total timeouts → `504`, with the upstream request cancelled.
  A client disconnect mid-stream closes the upstream stream so the engine slot frees.
- **Error mapping**: engine error → `502`, engine unreachable/loading → `503`.
- **Graceful shutdown**: on `SIGTERM`, `/readyz` turns `503`, new requests get `503`, and
  in-flight requests get up to `drain_timeout_s` (default 30s) to finish.
- **Observability**: Prometheus metrics at `/metrics` (`inferscale_*`) and one structured JSON
  log line per request. Prompt and response content is never logged.

Current target: Qwen3-8B Q4_K_M on an Apple M1 Pro (Metal), thinking disabled.
The model is pinned in `configs/models/registry.yaml` by HF revision and sha256.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and `brew install llama.cpp`, plus the model file at
`~/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf`.

```bash
make setup             # uv sync (Python 3.12)
make engine            # llama-server on 127.0.0.1:8081 (scripts/run_engine.sh, ENGINE_* env overrides)
make serve             # gateway on 127.0.0.1:8000
make test              # unit + contract tests (FakeBackend, no engine needed)
make test-integration  # against the running engine
make lint typecheck
```

```bash
curl -N localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}'
```

Configuration lives in `configs/serving/mac-llamacpp.yaml`. Any field can be overridden with
`INFERSCALE_<FIELD>` (nested: `INFERSCALE_ADMISSION__MAX_INFLIGHT=4`), and
`INFERSCALE_CONFIG_FILE` selects another profile.

Endpoints: `/v1/chat/completions`, `/healthz` (process alive), `/readyz` (engine healthy,
registry loaded and not draining), `/metrics`.

## Status

**No benchmark results yet.** The numbers in the bring-up notes are single smoke-test requests,
not benchmarks.
