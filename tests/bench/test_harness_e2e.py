"""End-to-end harness runs against the gateway + FakeBackend under a real uvicorn server."""

from __future__ import annotations

from typing import Any

import pytest

from bench.analyze import decode_rate, itls, run_metrics, ttft
from bench.harness import Target, poisson_offsets, run_load
from inferscale.backends.fake import FakeBackend
from tests.bench.conftest import LiveFactory

PROMPTS = [[{"role": "user", "content": f"alert {i}"}] for i in range(3)]
MODEL = "qwen3-8b-q4km-v1"


def max_overlap(records: list[dict[str, Any]]) -> int:
    events = sorted(
        [(r["send_t"], 1) for r in records] + [(r["end_t"], -1) for r in records],
        key=lambda e: (e[0], e[1]),
    )
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


async def test_closed_loop_concurrency_1(live_gateway: LiveFactory) -> None:
    url = await live_gateway(FakeBackend(first_token_delay_s=0.05, token_delay_s=0.02))
    run = await run_load(
        Target("gateway", url, MODEL), PROMPTS, n_requests=4, max_tokens=32, warmup=2
    )
    recs = run["records"]
    assert [r["phase"] for r in recs] == ["warmup"] * 2 + ["measure"] * 4
    measured = [r for r in recs if r["phase"] == "measure"]
    assert all(r["ok"] and r["status"] == 200 for r in measured)
    for r in measured:
        assert ttft(r) is not None and ttft(r) >= 0.05  # first-token delay is honoured
        assert len(r["delta_ts"]) == 6  # six content deltas; the empty role chunk is not counted
        assert r["completion_tokens"] == 6 and r["prompt_tokens"] == 7  # from the usage chunk
        assert all(0.01 < g < 0.2 for g in itls(r))
        assert 20 < (decode_rate(r) or 0) < 60  # 5 tokens / ~0.1 s
    assert max_overlap(measured) == 1
    m = run_metrics(run, n_boot=50)
    assert m["n_ok"] == 4 and m["errors"] == {}
    assert m["throughput_tok_s"] and m["throughput_tok_s"] > 0


async def test_closed_loop_concurrency_4(live_gateway: LiveFactory) -> None:
    url = await live_gateway(
        FakeBackend(first_token_delay_s=0.1, token_delay_s=0.02), admission={"max_inflight": 4}
    )
    run = await run_load(
        Target("gateway", url, MODEL), PROMPTS, n_requests=12, max_tokens=32, concurrency=4
    )
    measured = [r for r in run["records"] if r["phase"] == "measure"]
    assert all(r["ok"] for r in measured)
    assert max_overlap(measured) == 4
    assert run["concurrency"] == 4


async def test_errors_are_recorded(live_gateway: LiveFactory) -> None:
    url = await live_gateway(
        FakeBackend(first_token_delay_s=0.3), admission={"max_inflight": 1, "max_queue": 0}
    )
    run = await run_load(
        Target("gateway", url, MODEL), PROMPTS, n_requests=3, max_tokens=8, concurrency=3
    )
    errors = sorted(r["error"] for r in run["records"] if not r["ok"])
    assert errors == ["queue_full", "queue_full"]
    assert {r["status"] for r in run["records"] if not r["ok"]} == {429}


async def test_open_loop_poisson(live_gateway: LiveFactory) -> None:
    url = await live_gateway(
        FakeBackend(token_delay_s=0.005), admission={"max_inflight": 8, "max_queue": 8}
    )
    run = await run_load(
        Target("gateway", url, MODEL),
        PROMPTS,
        n_requests=8,
        max_tokens=16,
        mode="open",
        rate=20,
        seed=7,
    )
    measured = [r for r in run["records"] if r["phase"] == "measure"]
    sched = [r["scheduled_t"] for r in measured]
    expected = poisson_offsets(8, 20, 7)
    assert [s - sched[0] for s in sched] == pytest.approx(expected, abs=1e-9)
    assert all(r["send_t"] >= r["scheduled_t"] - 1e-6 for r in measured)
    assert all(r["send_t"] - r["scheduled_t"] < 0.05 for r in measured)  # sent on schedule
    assert all(r["ok"] for r in measured)
    assert run["mode"] == "open" and run["rate"] == 20


def test_poisson_offsets_seeded() -> None:
    a, b = poisson_offsets(50, 10, 3), poisson_offsets(50, 10, 3)
    assert a == b and a[0] == 0.0 and a == sorted(a)
    assert poisson_offsets(50, 10, 4) != a
    assert 2.5 < a[-1] < 8.0  # ~49 gaps of mean 0.1 s


async def test_open_loop_needs_rate() -> None:
    with pytest.raises(ValueError):
        await run_load(
            Target("x", "http://127.0.0.1:1"), PROMPTS, n_requests=1, max_tokens=1, mode="open"
        )
