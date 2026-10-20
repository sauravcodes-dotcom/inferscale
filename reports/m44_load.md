# Sweep m44_load

Open-loop Poisson load (0.10-0.40 req/s) on BFCL dev requests: achieved throughput,
429 shedding, tail latency and goodput (e2e <= 15 s). Qwen3-8B Q4_K_M, 2 slots, cache-ram 0.

Started 2026-10-20T20:03:46+00:00, git bb2328095c8a (dirty). Valid (no swap growth): False.

| variant | load | ok/n | e2e p50 s [95% CI] | e2e p95 s | req/s | prefill tok/s | decode tok/s/req | prompt reuse | peak RSS GB | KV MiB (computed) | swap growth MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_2slots_nocache | 0.1/s | 56/60 | 8.60 [6.91, 10.72] | 23.67 | 0.109 | 231 | 11.0 | 7% | 4.70 | 2304 | 654 |
| q4_2slots_nocache | 0.15/s | 49/60 | 10.22 [6.44, 12.14] | 23.42 | 0.142 | 233 | 9.9 | 7% | 7.56 | 2304 | 16 |
| q4_2slots_nocache | 0.2/s | 47/60 | 11.55 [9.54, 12.67] | 23.05 | 0.177 | 233 | 9.3 | 7% | 7.25 | 2304 | 296 |
| q4_2slots_nocache | 0.25/s | 41/60 | 12.62 [9.87, 14.25] | 25.51 | 0.192 | 232 | 9.3 | 8% | 7.57 | 2304 | 134 |
| q4_2slots_nocache | 0.3/s | 37/60 | 13.60 [10.23, 16.35] | 26.00 | 0.200 | 234 | 9.1 | 8% | 7.52 | 2304 | 0 |
| q4_2slots_nocache | 0.4/s | 30/60 | 14.12 [12.15, 17.71] | 23.87 | 0.209 | 234 | 8.3 | 8% | 7.23 | 2304 | 21 |

e2e: request sent to last byte received, as the client sees it (in gateway_validated mode the gateway sends a tool call only after validating it, so time to first token equals e2e). prefill/decode: llama.cpp counters over the run (per-request rates). Prompt reuse: share of client prompt tokens the engine did not evaluate (prompt cache). Each run starts from a restarted engine, so reuse is within-run only. KV MiB: computed from total context (ENGINE_CTX) and the model's KV bytes per token.


Open-loop load (Poisson arrivals). Goodput: requests completed OK within 15 s, per second; shed = 429 queue_full/queue_timeout.

| variant | offered req/s | n | achieved req/s | shed | other errors | p50 s | p95 s | p99 s | goodput req/s | within SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| q4_2slots_nocache | 0.1 | 60 | 0.109 | 7% |  | 8.74 | 25.04 | 29.26 | 0.094 | 80% |
| q4_2slots_nocache | 0.15 | 60 | 0.142 | 18% |  | 10.22 | 23.85 | 26.16 | 0.113 | 65% |
| q4_2slots_nocache | 0.2 | 60 | 0.177 | 22% |  | 11.55 | 23.14 | 35.77 | 0.132 | 58% |
| q4_2slots_nocache | 0.25 | 60 | 0.192 | 32% |  | 12.62 | 25.51 | 36.39 | 0.136 | 48% |
| q4_2slots_nocache | 0.3 | 60 | 0.200 | 38% |  | 13.60 | 32.83 | 37.46 | 0.119 | 37% |
| q4_2slots_nocache | 0.4 | 60 | 0.209 | 50% |  | 14.24 | 24.46 | 38.72 | 0.111 | 27% |

- q4_2slots_nocache errors: {'queue_timeout': 97, 'queue_full': 3}
