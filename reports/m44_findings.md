# M4.4 open-loop load test: findings

Poisson arrivals of BFCL dev requests (with tools) at 0.10–0.40 req/s, 60 requests per rate,
through the gateway with its normal admission settings: 2 in flight, a queue of 8 and a 10 s
queue timeout. Qwen3-8B Q4_K_M, 2 slots × 8192 tokens, `--cache-ram 0`, `gateway_validated`
mode, SLO 15 s end to end. Full table: [m44_load.md](m44_load.md). Raw data:
`results/bench/m44_load/20261020T200346Z.json`.

**Not a clean run (recorded in the result file).** The owner chose to run it on a busy machine:
Chrome was using about 3 CPU cores, Ollama was running, swap stood at 3.5 GB, and swap grew
during 3 of the 6 runs. Service time roughly doubled as a result: the median was 8.6 s at
0.1 req/s, against 3.7 s single-stream on a clean machine. Read the absolute latencies as
pessimistic. The shape of the curves, and the admission behaviour, are what this run shows.

| offered req/s | achieved req/s | shed (429) | p50 s | p95 s | p99 s | goodput req/s (≤ 15 s) | within SLO |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.109 | 7% | 8.74 | 25.04 | 29.26 | 0.094 | 80% |
| 0.15 | 0.142 | 18% | 10.22 | 23.85 | 26.16 | 0.113 | 65% |
| 0.20 | 0.177 | 22% | 11.55 | 23.14 | 35.77 | 0.132 | 58% |
| 0.25 | 0.192 | 32% | 12.62 | 25.51 | 36.39 | **0.136** | 48% |
| 0.30 | 0.200 | 38% | 13.60 | 32.83 | 37.46 | 0.119 | 37% |
| 0.40 | 0.209 | 50% | 14.24 | 24.46 | 38.72 | 0.111 | 27% |

There were no 5xx responses and no errors other than 429s. Of the 100 rejections (of 360
requests), 97 were `queue_timeout` and 3 `queue_full`.

## What it shows

1. **Throughput saturates near 0.21 req/s, and the gateway degrades gracefully.** Achieved
   throughput levels off at 0.20–0.21 req/s. Excess load is rejected with 429 + `Retry-After`
   rather than piling up without bound or failing with 5xx. Those rejections are mostly not
   fast, though: see point 3.
2. **Goodput peaks at about 0.25 req/s offered, then falls.** Past the knee, goodput drops (0.136
   → 0.111 req/s) even though raw throughput keeps rising. Admitted requests increasingly miss
   the SLO, because each one waits in the queue for up to 10 s before getting a slot. That wait
   plus the service time pushes it past 15 s.
3. **The shedding comes from the queue timeout, not from the queue bound.** The queue (depth 8)
   almost never filled (3 `queue_full`). Requests were shed after waiting the full 10 s. That
   wasted 10 s of each rejected client's time, and requests admitted near the deadline still
   blew the SLO.
4. **Tuning implication.** The queue timeout should be derived from the SLO: queue timeout ≈
   SLO − typical service time, here about 15 − 4 = 11 s on a clean machine, and less under
   contention. The queue depth should be about the number of requests the slots can drain
   within that time (≈ 2 slots × 11 s / 4 s ≈ 5). With a timeout that leaves no room for
   service, the gateway admits work it then serves too late.

## Next

- Rerun on a clean machine for real absolute latencies.
- Run the same load with the admission settings derived above (for example a 6 s queue timeout
  and a queue of 4) as a second variant, and compare goodput past the knee.
