# InferScale

An LLM inference and reliability platform around a local llama.cpp engine (`llama-server`):
an OpenAI-compatible gateway, measured against the real engine for serving performance,
failure behaviour and task quality (BFCL function calling, scored with the official scorer).

The gateway exposes `POST /v1/chat/completions` (streaming and non-streaming, including OpenAI
`tools` / `tool_calls`) and adds:

- **Admission control**: at most `max_inflight` requests on the engine (= engine slots), a
  bounded wait queue, and an immediate `429` + `Retry-After` when full. A permit is held until
  the engine's `/slots` shows the slot free, because llama.cpp keeps prefilling a cancelled
  request (see [Reliability](#reliability)).
- **Deadlines**: connect, first-token and total timeouts → `504`, with the upstream request cancelled.
  A client disconnect mid-stream closes the upstream stream so the engine slot frees.
- **Error mapping**: engine context overflow → `400 context_length_exceeded`, other engine 4xx →
  `400`, engine 5xx → `502`, engine unreachable/loading → `503`.
- **Tool-call modes** (`tool_call_mode`, see [Tool calling](#tool-calling)).
- **Graceful shutdown**: on `SIGTERM`, `/readyz` turns `503`, new requests get `503`, and
  in-flight requests get up to `drain_timeout_s` (default 30s) to finish.
- **Observability**: Prometheus metrics at `/metrics` (`inferscale_*`: latency, queue, held
  permits, finish reasons, tool-call fallbacks) and one structured JSON log line per request.
  Prompt and response content is never logged.

Current target: Qwen3-8B Q4_K_M on an Apple M1 Pro 16 GB (Metal), llama.cpp b11669, thinking
disabled. The model is pinned in `configs/models/registry.yaml` by HF revision and sha256.

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

Engine profiles live in `configs/engine/*.env` (`ENGINE_PROFILE=bfcl make engine`: 2 slots x
8192 tokens, host prompt cache capped at 1 GiB, since llama.cpp's 8 GiB default swaps on 16 GB).

## Tool calling

llama.cpp b11669 turns each tool's JSON schema into a decoding grammar with a fixed property
order. When the model writes valid arguments in another order, the grammar forces other tokens
and corrupts them. The gateway therefore offers three modes:

| `tool_call_mode` | how | |
|---|---|---|
| `engine` | llama.cpp `/v1/chat/completions` with its grammar | default |
| `gateway` | prompt rendered with the model's chat template (`/apply-template`), unconstrained `/completion`, `<tool_call>` blocks parsed in the gateway (`backends/tool_parsing.py`); malformed blocks are kept as content and counted, never repaired | |
| `gateway_validated` | `gateway`, then each call is validated against its tool's JSON schema; if any call is invalid, the request is re-run on the engine's constrained path | BFCL frozen config |

Responses carry `x_inferscale` (`tool_call_mode`, `fallback`, `fallback_reasons`,
`malformed_tool_calls`). In `gateway_validated` mode a streamed response is sent only after
validation, so its time to first byte equals its completion time.

## Evaluation: BFCL single-turn function calling

[BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
v4 single-turn categories (non-live and live AST, irrelevance), generated and scored by the
official `bfcl-eval==2026.3.23` (Apache-2.0) in an isolated, hash-locked venv
(`tools/bfcl/`), through the gateway. The adapter (`tools/bfcl/inferscale_bfcl.py`) only points
BFCL's own Qwen FC handler at the gateway; prompts, decoding and scoring are BFCL's.

```bash
make bfcl-setup                 # tools/bfcl/.venv from requirements.lock (--require-hashes)
make engine-bfcl                # engine, bfcl profile (one terminal)
INFERSCALE_CONFIG_FILE=configs/serving/mac-llamacpp-bfcl-gwv.yaml make serve   # gateway (another)
uv run python -m evals.bfcl.run --partition dev --serving-profile configs/serving/mac-llamacpp-bfcl-gwv.yaml
uv run python -m evals.bfcl.compare <A.json> <B.json>   # paired bootstrap + exact McNemar
make bfcl-test                  # held-out test, frozen config only (see below)
```

- **Split** (`splits/bfcl_v4_single_turn_v1.json`, `scripts/make_bfcl_split.py`): dev 347 /
  test 3,294, about 10% of each category in dev, seed 20261018. It is leakage-aware: entries
  with the same normalised question, or from the same live source group, form components that
  are assigned whole, so no question or live source group spans dev and test.
- **Metrics** (`evals/bfcl/metrics.py`): leaderboard aggregates (non-live AST = unweighted mean
  of simple, multiple, parallel, parallel_multiple; live = count-weighted), category-stratified
  bootstrap 95% CIs, and an error class per failure.
- **Baselines** (`results/bfcl/baselines/dev.json`): never-call (0% AST, 100% irrelevance);
  BM25 function choice with no arguments (~0% AST, although it picks the right function in
  90% of `multiple` and 88.6% of `live_multiple` entries, so argument filling is the hard part).
- **Freeze**: `configs/eval/bfcl_frozen.yaml` pins a config hash over the adapter, lock file,
  engine and serving profiles, temperature and engine `/props`. `evals.bfcl.run` refuses the
  test partition unless that hash matches, and `make bfcl-test` adds the clean-environment
  preflight.

Dev results (347 entries, official scorer, %):

| tool_call_mode | non-live AST | live | irrelevance | live irrelevance |
|---|---|---|---|---|
| engine | 87.29 | 75.00 | 75.0 | 79.55 |
| gateway | 85.42 | 84.17 | 75.0 | 79.55 |
| **gateway_validated** (frozen) | 87.08 | 84.17 | 75.0 | 79.55 |
| published Qwen3-8B (FC), full set | 87.58 | 80.53 | | |

`gateway_validated` vs `engine`: live +9.17 points, 95% CI [+5.0, +15.0]. The whole difference
is in `live_multiple`: 11 entries are fixed and 0 are broken (exact McNemar p = 0.001). No other
category changes. Dev numbers guided the choice of mode, so they are not a final result. The
held-out test has not been run.

**Label quality.** Every one of the 57 dev failures of the frozen config was reviewed against the
tool schema and BFCL's possible answers (`results/bfcl/audit/`): 35 model errors, 8 label
errors, 14 ambiguous. Headline numbers are always the official scorer's. If label errors were
counted correct, non-live AST would be 92.50 (vs 87.08) and live 85.83 (vs 84.17).

## Reliability

`make reliability` (`bench/reliability.py`) starts its own engine and gateway, injects
failures, and checks behaviour with evidence (statuses, metric deltas, timings). Latest run
(`results/reliability/20261019T020838Z.json`): **26/26 checks pass**.

| scenario | what is checked |
|---|---|
| overload | Poisson arrivals at ~3x capacity: 82 of 90 shed with 429 + `Retry-After` (queue-full p95 5 ms; queue timeouts at 5.0 s), no 5xx, engine never above its 2 slots, queue never above its bound |
| deadline | prefill longer than a 2 s first-token deadline: 504 at 2.01 s; the permit is held while llama.cpp keeps prefilling (~30 s) and released 0.12 s after the slot frees |
| cancel | client disconnect mid-stream: engine idle within 0.23 s, 25 of 1024 tokens generated |
| engine_crash | SIGKILL mid-stream: in-band stream error, `/readyz` 503, fast 503 `engine_unavailable` during the outage, recovery after restart |
| drain | SIGTERM under load: `/readyz` 503, new requests 503, both in-flight streams complete, exit in 11.9 s |

The deadline scenario found that llama.cpp checks for cancellation only after prefill, so a
cancelled long prompt kept its slot busy while the gateway had already admitted a replacement.
Holding admission permits until the engine reports the slot free fixed the over-admission.

## Benchmarks

`bench/` measures serving through the gateway with streaming clients and a 1 Hz sampler
(engine RSS, swap, memory pressure, thermal state, queue depth). Runs are refused unless the
machine is clean: AC power, swap <= 64 MB, no browsers or Ollama. A run is invalid if swap grows
by more than 32 MB. `bench/sweep.py` restarts the engine for each configuration, and for each
level and repeat, so no run inherits another's prompt cache. It records end-to-end latency
(bootstrap CIs), throughput, llama.cpp's prefill and decode rates, prompt-cache reuse, peak RSS,
computed KV size and swap growth.

```bash
make bfcl-requests     # BFCL dev requests as BFCL builds them (git-ignored)
make m43-smoke         # pipeline check, not a result
make m43               # slots x concurrency, Q4_K_M vs Q8_0, host prompt-cache size (~4.5 h)
```

No sweep results yet: they need a clean-environment window.

One measurement note from the BFCL runs: BFCL's own `latency` times a streaming `create()`
call, i.e. time to response headers. The 1.5 s (streaming modes) vs 3.75 s
(`gateway_validated`) means therefore compare first byte with completion. All three dev runs
took 1302-1308 s wall time for the 347 entries.

## Secondary evaluation: PAG incident analysis

The model gets the incident fields of a PAG record (summary, description, detection method, impact,
region, timestamps) and must return JSON with severity, root-cause type and team, a root-cause
summary, and corrective and preventive actions (`evals/schemas/pag_output_v1.json`). It sees no
labels. Requests go through the gateway, streaming, at temperature 0.

```bash
make fetch-data        # download PAG at the pinned commit into data/pag/ (git-ignored), verify sha256
make split             # rebuild splits/pag_v1.json (deterministic; committed)
make baselines         # generic + retrieval baselines -> results/baselines/pag_v1.json
make eval-smoke        # 10 dev items (smoke, not a result)
make eval-dev          # full dev split (103); --prompt-version pag_v1.p1|pag_v1.p2
```

The test split (495) is held out. `run_eval` refuses it unless you pass
`--i-am-running-final-test` and the prompt is the one frozen in `configs/eval/frozen.yaml`
(with an unchanged sha256 and a referenced full dev result). To freeze another prompt, run
`make freeze-prompt PROMPT=pag_v1.p2 REASON="..."`. `make eval-test` runs the clean-environment
preflight (AC power, swap <= 64 MB, no Chrome/Safari/Ollama, ports free), starts the engine and
gateway, runs the test split with the frozen prompt, and stops both.
`make check-data-text` (`scripts/check_no_dataset_text.py`) fails if any tracked file contains PAG text.

## Data and licence

- **Source:** PAG (Postmortem Action Generation) synthetic cloud-incident dataset,
  <https://github.com/sancharichowdhuri2/PAG>, file `PAG-Synthetic-Cloud-Incident-Dataset.json`,
  pinned to commit `ea4860499ab2fd8b0f7d1ac835257b70bb6e4d3a`, sha256
  `5a68a736d9ab53a9049daabda27818388812c734c2f8eba836d37279e88c7be7` (950 records; 850 used
  after excluding source indices 0-99, which duplicate later records).
- **Licence:** the PAG repository has **no LICENSE file**, so this repository does not
  redistribute any PAG records, descriptions, summaries or reference actions. The data is
  synthetic, and its labels are not known to be human-validated.
- **Getting the data:** `make fetch-data` downloads the file from the pinned commit into the
  git-ignored `data/pag/` and verifies the sha256.
- **What is committed:**
  - split membership by Incident_ID and source index (`splits/pag_v1.json`);
  - aggregate metrics, environment fingerprints and ID-list hashes (`results/baselines/`,
    `results/eval/<model>/*.json`);
  - code, prompts, the output schema and the grounding lexicon.

  Per-example predictions are written to the git-ignored `results/eval/raw/`, because they sit
  next to reference text. Model outputs are committed only if they contain no PAG text.

BFCL data is Apache-2.0 and comes from the pinned `bfcl-eval` package. Its per-entry outputs
are written to the git-ignored `results/bfcl/raw/`.

## Status

- Gateway, tool-call modes and reliability suite: implemented and validated (26/26).
- BFCL: dev results above; the held-out test (3,294 entries, frozen config) has not been run yet.
- Serving sweeps: tooling done, runs pending a clean environment.
- PAG: dev-split numbers are for prompt development only; its held-out test has not been run.
