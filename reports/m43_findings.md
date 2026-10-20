# M4.3 serving sweeps: findings

Qwen3-8B on an M1 Pro (16 GB), llama.cpp b11669 (Metal), gateway in `gateway_validated` mode.
The workload is 120 BFCL dev requests with their tools (seeded shuffle), closed loop, 2 repeats.
The engine is restarted before every run. Full tables: [m43_slots.md](m43_slots.md) and
[m43_quant.md](m43_quant.md). Raw data: `results/bench/m43_slots/20261019T165419Z.json` and
`results/bench/m43_quant/20261019T200008Z.json`.

**Conditions (not clean, recorded in the result files):**
- The slots sweep started on battery at the owner's request, and later moved to AC.
- The quant sweep started with 1.25 GB of swap left over from the 4-slot runs.
- Runs whose swap grew more than 32 MB are flagged invalid: 4 of 6 runs with 4 slots, and all
  4 Q8_0 runs.
- Repeats agree within 0.1 s at the median (for example 7.13 s vs 7.13 s), so run-to-run noise
  is small next to the effects below.

## 1. Parallel slots barely help throughput

| slots | concurrency | e2e p50 s | req/s | decode tok/s per request | peak RSS GB |
|---|---|---|---|---|---|
| 1 | 1 | 3.79 | 0.221 | 24.9 | 8.4 |
| 2 | 1 | 3.73 | 0.230 | 24.9 | 9.3 |
| 2 | 2 | 7.13 | 0.227 | 8.3 | 9.6 |
| 2 | 4 | 16.87 | 0.227 | 8.3 | 9.4 |
| 4 | 4 | 11.77 | 0.298 | 5.2 | 11.3 (swaps) |

- **Two slots:** with Q4_K_M, running 2 requests at once does not raise throughput (0.230 →
  0.227 req/s), and it roughly doubles latency (3.7 → 7.1 s). Per-request decode drops from
  24.9 to 8.3 tok/s, so aggregate decode falls from 24.9 to about 16.6 tok/s. One request at a
  time is faster than two.
- **Four slots:** they do help (+30% req/s, aggregate decode about 21 tok/s), but the median
  latency is 11.8 s, the 4.6 GiB KV cache pushes peak RSS to 11.3 GB, and the Mac swaps.
- **Hypothesis, not verified here:** llama.cpp's Metal kernels for batched Q4_K matrix
  multiplication are less efficient at batch 2 than its single-sequence path. A llama-bench
  batch-size scan would confirm or refute this.
- **Single-stream latency** does not depend on the slot count (3.73–3.88 s).
- **Practical setting:** for this workload on this machine, 2 slots behave like 1 slot with
  queueing. Admission at 2 slots with a bounded queue is what protects latency. Each extra
  concurrent client adds about one service time to the median.

## 2. Q8_0 vs Q4_K_M: same single-stream latency, better under concurrency, more memory

| model | concurrency | e2e p50 s | req/s | prefill tok/s | decode tok/s per request | peak RSS GB |
|---|---|---|---|---|---|---|
| Q4_K_M | 1 | 3.73 | 0.230 | 240 | 24.9 | 9.8 |
| Q8_0 | 1 | 3.70 | 0.221 | 274 | 21.1 | 11.7 (swaps) |
| Q4_K_M | 2 | 7.13 | 0.227 | 235 | 8.3 | 9.9 |
| Q8_0 | 2 | 6.07 | 0.275 | 269 | 10.6 | 11.7 (swaps) |

- **Single stream:** Q8_0 prefills 14% faster (no k-quant dequantisation work) and decodes 15%
  slower (1.7x the bytes per token from memory). For these prompt-heavy tool-calling requests
  the two effects cancel: the median is 3.70 vs 3.73 s.
- **Two concurrent requests:** Q8_0 is 21% faster in throughput (0.275 vs 0.227 req/s) with a
  lower median (6.07 vs 7.13 s). This is consistent with the batched Q4_K kernel being the
  bottleneck in finding 1.
- **Memory:** Q8_0 peaks at 11.7 GB and swaps on 16 GB in every run, so it is not a safe
  production setting on this machine.
- **Quality:** BFCL accuracy of Q8_0 has not been measured.

## Not measured

- **Prompt-cache size sweep:** stopped. The Mac slept repeatedly on battery with the lid closed,
  so requests timed out across sleeps (16 timeouts in the first 162). It needs a rerun on AC.
- **Per-request time to first token:** in `gateway_validated` mode it equals end-to-end
  latency, because the gateway sends a tool call only after validating it.
