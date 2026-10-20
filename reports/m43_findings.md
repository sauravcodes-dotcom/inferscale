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
- **Cause (confirmed by the batch scan below):** llama.cpp's batch-2 Q4_K decode on Metal is
  barely faster than batch 1, and prefill does not batch at all.
- **Single-stream latency** does not depend on the slot count (3.73–3.88 s).
- **Practical setting:** for this workload on this machine, 2 slots behave like 1 slot with
  queueing. Admission at 2 slots with a bounded queue is what protects latency. Each extra
  concurrent client adds about one service time to the median.

### Why: batch-2 decoding is slow for Q4_K, and prefill does not batch

`llama-batched-bench` (llama.cpp b11669, no gateway) runs a 1,024-token prompt plus 64
generated tokens per sequence, with 1, 2, 4 and 8 parallel sequences
(`results/bench/batched/*.jsonl`; command in its README note below).

| parallel sequences | Q4_K_M decode tok/s (total) | Q8_0 decode tok/s (total) | prefill tok/s (Q4 / Q8) |
|---|---|---|---|
| 1 | 25.0 | 20.1 | 250 / 271 |
| 2 | **27.3 (+9%)** | 33.5 (+67%) | 252 / 282 |
| 4 | 58.3 | 63.1 | 251 / 279 |
| 8 | 106.0 | 107.2 | 251 / 285 |

- **Batch 2:** decoding 2 sequences together is barely faster than 1 for Q4_K_M, while Q8_0
  gains 67% and both scale well from 4 upwards. This confirms the hypothesis above for this
  llama.cpp build on Metal.
- **Prefill:** throughput is flat in the number of sequences (compute-bound). A BFCL request
  (~800 prompt tokens, ~55 generated) spends more time in prefill (~4 s at 1,024 tokens) than in
  decode (~2.6 s). So parallelism can at best speed up the smaller part of the work. This is
  also why the serving sweep shows lower per-request decode under concurrency than the batch
  scan: one request's prefill stalls the other's decode steps.

Command: `llama-batched-bench -m <model> -c 9216 -b 2048 -ub 512 -ngl 99 -npp 1024 -ntg 64
-npl 1,2,4,8 --output-format jsonl`.

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

## 3. The host prompt cache buys almost nothing; disable it

All 347 BFCL dev requests in evaluation order (category-grouped, so consecutive requests share
tool definitions), single stream, one run per setting. The run started at 00:47 UTC on AC with
886 MB of swap left from earlier runs. Full table: [m43_cache_ram.md](m43_cache_ram.md).

| `--cache-ram` MiB | wall s | e2e p50 s [95% CI] | prompt tokens reused | peak RSS GB | swap growth |
|---|---|---|---|---|---|
| 0 | 1346 | 3.09 [2.89, 3.34] | 54.0% | 7.46 | 0 |
| 1024 | 1361 | 3.09 [2.91, 3.33] | 55.4% | 9.05 | 0 |
| 4096 | 1374 | 3.16 [2.88, 3.35] | 55.9% | 9.25 | +2.75 GB |
| 8192 (llama.cpp default) | 1295 | 2.97 [2.74, 3.16] | 56.9% | 9.16 | +6.80 GB |

- **Reuse:** 54% of prompt tokens are reused with no host cache at all. That reuse is llama.cpp
  keeping a slot's KV prefix between consecutive requests that share a system prompt and tools.
  The host cache adds only 1.4 to 2.9 points.
- **Latency:** the medians overlap across all four settings. The 8 GiB setting finished 4%
  faster, but this is one run per setting and it swapped 6.8 GB doing it.
- **Memory:** even the 1 GiB cap costs about 1.6 GB of extra peak RSS (7.46 → 9.05 GB). At
  4–8 GiB, peak RSS stops growing only because the cache is pushed out to swap.
- **Recommendation:** `--cache-ram 0` for this workload on a 16 GB Mac: same latency, about
  1.6 GB less memory, no swap. The current default (1024) is safe but buys nothing measurable.
  This also confirms that the 9.5 GB of swap in the first BFCL dev run came from llama.cpp's
  8 GiB default.

## Not measured

- **Q8_0 accuracy on BFCL.**
- **Per-request time to first token:** in `gateway_validated` mode it equals end-to-end
  latency, because the gateway sends a tool call only after validating it.
