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

## Evaluation (PAG incident analysis)

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

## Status

**No final benchmark results yet.** Dev-split numbers are for prompt development, not final
results. The held-out test split has not been run.
