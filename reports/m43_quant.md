# Sweep m43_quant

Q4_K_M vs Q8_0 weights on BFCL dev requests: latency, decode rate, memory
(Qwen3-8B, 2 slots x 8192 tokens, 1 GiB host prompt cache, gateway_validated).

Started 2026-10-19T20:00:08+00:00, git a595e33f0c90 (dirty). Valid (no swap growth): False.

| variant | conc | ok/n | e2e p50 s [95% CI] | e2e p95 s | req/s | prefill tok/s | decode tok/s/req | prompt reuse | peak RSS GB | KV MiB (computed) | swap growth MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_k_m | 1 | 120/120 | 3.74 [3.06, 4.20] | 10.48 | 0.229 | 239 | 24.9 | 24% | 9.80 | 2304 | 0 |
| q4_k_m | 1 | 120/120 | 3.73 [3.04, 4.18] | 10.49 | 0.230 | 240 | 24.9 | 24% | 9.85 | 2304 | 0 |
| q4_k_m | 2 | 120/120 | 7.13 [6.18, 8.62] | 17.65 | 0.227 | 235 | 8.3 | 18% | 9.94 | 2304 | 0 |
| q4_k_m | 2 | 120/120 | 7.14 [6.21, 8.64] | 17.64 | 0.227 | 235 | 8.3 | 18% | 9.84 | 2304 | 0 |
| q8_0 | 1 | 120/120 | 3.70 [3.22, 4.34] | 10.64 | 0.221 | 274 | 21.1 | 24% | 11.71 | 2304 | 909 |
| q8_0 | 1 | 120/120 | 3.70 [3.25, 4.37] | 10.64 | 0.221 | 275 | 21.1 | 24% | 11.89 | 2304 | 122 |
| q8_0 | 2 | 120/120 | 6.07 [4.99, 6.64] | 15.21 | 0.275 | 269 | 10.6 | 20% | 11.70 | 2304 | 229 |
| q8_0 | 2 | 120/120 | 6.08 [4.92, 6.62] | 15.17 | 0.275 | 269 | 10.6 | 20% | 11.70 | 2304 | 45 |

e2e: request sent to last byte received, as the client sees it (in gateway_validated mode the gateway sends a tool call only after validating it, so time to first token equals e2e). prefill/decode: llama.cpp counters over the run (per-request rates). Prompt reuse: share of client prompt tokens the engine did not evaluate (prompt cache). Each run starts from a restarted engine, so reuse is within-run only. KV MiB: computed from total context (ENGINE_CTX) and the model's KV bytes per token.

