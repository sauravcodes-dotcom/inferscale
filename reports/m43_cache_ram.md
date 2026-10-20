# Sweep m43_cache_ram

Host prompt-cache size vs latency, prompt-token reuse and memory/swap, all 347 BFCL dev
requests in evaluation order (Qwen3-8B Q4_K_M, 2 slots x 8192 tokens, gateway_validated).

Started 2026-10-20T00:46:58+00:00, git c921c1a03a2d (dirty). Valid (no swap growth): False.

| variant | conc | ok/n | e2e p50 s [95% CI] | e2e p95 s | req/s | prefill tok/s | decode tok/s/req | prompt reuse | peak RSS GB | KV MiB (computed) | swap growth MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cache0 | 1 | 347/347 | 3.09 [2.89, 3.34] | 9.19 | 0.258 | 228 | 24.3 | 54% | 7.46 | 2304 | 0 |
| cache1024 | 1 | 347/347 | 3.09 [2.91, 3.33] | 9.40 | 0.255 | 223 | 23.8 | 55% | 9.05 | 2304 | 0 |
| cache4096 | 1 | 347/347 | 3.16 [2.88, 3.35] | 9.41 | 0.253 | 219 | 23.6 | 56% | 9.25 | 2304 | 2754 |
| cache8192 | 1 | 347/347 | 2.97 [2.74, 3.16] | 8.85 | 0.268 | 234 | 24.7 | 57% | 9.16 | 2304 | 6802 |

e2e: request sent to last byte received, as the client sees it (in gateway_validated mode the gateway sends a tool call only after validating it, so time to first token equals e2e). prefill/decode: llama.cpp counters over the run (per-request rates). Prompt reuse: share of client prompt tokens the engine did not evaluate (prompt cache). Each run starts from a restarted engine, so reuse is within-run only. KV MiB: computed from total context (ENGINE_CTX) and the model's KV bytes per token.

