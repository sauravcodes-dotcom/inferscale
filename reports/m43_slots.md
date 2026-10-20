# Sweep m43_slots

Engine slots (1/2/4) x offered concurrency on BFCL dev requests: throughput, latency,
KV memory (Qwen3-8B Q4_K_M, 1 GiB host prompt cache, gateway_validated).

Started 2026-10-19T16:54:19+00:00, git a595e33f0c90 (dirty). Valid (no swap growth): False.

| variant | conc | ok/n | e2e p50 s [95% CI] | e2e p95 s | req/s | prefill tok/s | decode tok/s/req | prompt reuse | peak RSS GB | KV MiB (computed) | swap growth MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| slots1 | 1 | 120/120 | 3.79 [3.16, 4.47] | 11.37 | 0.221 | 240 | 24.9 | 19% | 8.39 | 1152 | 0 |
| slots1 | 1 | 119/120 | 3.88 [3.18, 4.53] | 10.63 | 0.218 | 54 | 21.8 | 17% | 8.29 | 1152 | 0 |
| slots1 | 2 | 120/120 | 8.40 [7.58, 9.29] | 15.18 | 0.221 | 240 | 24.9 | 19% | 8.60 | 1152 | 0 |
| slots1 | 2 | 120/120 | 8.39 [7.47, 9.29] | 15.17 | 0.221 | 240 | 24.9 | 19% | 8.50 | 1152 | 0 |
| slots2 | 1 | 120/120 | 3.73 [3.05, 4.20] | 10.49 | 0.230 | 240 | 24.9 | 24% | 9.27 | 2304 | 0 |
| slots2 | 1 | 120/120 | 3.72 [3.04, 4.19] | 10.48 | 0.230 | 240 | 24.9 | 24% | 9.65 | 2304 | 0 |
| slots2 | 2 | 120/120 | 7.13 [6.19, 8.66] | 17.65 | 0.227 | 235 | 8.3 | 18% | 9.62 | 2304 | 0 |
| slots2 | 2 | 120/120 | 7.13 [6.23, 8.62] | 17.63 | 0.226 | 235 | 8.3 | 18% | 9.75 | 2304 | 0 |
| slots2 | 4 | 120/120 | 16.87 [15.66, 18.21] | 26.07 | 0.227 | 235 | 8.3 | 18% | 9.44 | 2304 | 0 |
| slots2 | 4 | 120/120 | 16.88 [15.67, 18.21] | 26.06 | 0.227 | 235 | 8.3 | 18% | 9.48 | 2304 | 0 |
| slots4 | 1 | 120/120 | 3.79 [3.17, 4.47] | 10.55 | 0.223 | 240 | 24.9 | 20% | 11.29 | 4608 | 779 |
| slots4 | 1 | 120/120 | 3.80 [3.22, 4.47] | 11.34 | 0.221 | 238 | 24.7 | 20% | 11.08 | 4608 | 670 |
| slots4 | 2 | 120/120 | 7.12 [6.17, 8.67] | 17.96 | 0.227 | 235 | 8.3 | 21% | 11.24 | 4608 | 38 |
| slots4 | 2 | 120/120 | 7.13 [6.20, 8.68] | 17.97 | 0.227 | 235 | 8.3 | 21% | 11.26 | 4608 | 0 |
| slots4 | 4 | 120/120 | 11.77 [10.38, 13.89] | 29.93 | 0.298 | 232 | 5.2 | 21% | 11.29 | 4608 | 250 |
| slots4 | 4 | 120/120 | 12.06 [10.45, 13.91] | 29.82 | 0.298 | 232 | 5.1 | 21% | 11.31 | 4608 | 0 |

e2e: request sent to last byte received, as the client sees it (in gateway_validated mode the gateway sends a tool call only after validating it, so time to first token equals e2e). prefill/decode: llama.cpp counters over the run (per-request rates). Prompt reuse: share of client prompt tokens the engine did not evaluate (prompt cache). Each run starts from a restarted engine, so reuse is within-run only. KV MiB: computed from total context (ENGINE_CTX) and the model's KV bytes per token.

- slots1 errors: {'deadline_exceeded': 1}
