"""Hand-computed fixtures for the benchmark math."""

from __future__ import annotations

from typing import Any

import pytest

from bench.analyze import decode_rate, e2e, itls, percentile, run_metrics, ttft


@pytest.mark.parametrize(
    ("xs", "p", "expected"),
    [
        ([1, 2, 3, 4], 50, 2.5),  # k = 1.5
        ([1, 2, 3, 4], 90, 3.7),  # k = 2.7 -> 3 + 0.7
        ([1, 2, 3, 4], 99, 3.97),  # k = 2.97
        ([1, 2, 3, 4], 0, 1.0),
        ([1, 2, 3, 4], 100, 4.0),
        ([4, 1, 3, 2], 50, 2.5),  # unsorted input
        ([10, 20, 30, 40, 50], 95, 48.0),  # k = 3.8 -> 40 + 0.8*10
        ([7], 99, 7.0),
    ],
)
def test_percentile(xs: list[float], p: float, expected: float) -> None:
    assert percentile(xs, p) == pytest.approx(expected)


def test_percentile_empty_is_none() -> None:
    assert percentile([], 50) is None


def rec(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "phase": "measure",
        "ok": True,
        "send_t": 1.0,
        "first_token_t": 1.25,
        "end_t": 3.0,
        "delta_ts": [1.25, 1.35, 1.55, 2.25],
        "completion_tokens": 4,
        "prompt_tokens": 10,
        "status": 200,
        "error": None,
    }
    base.update(kw)
    return base


def test_latencies_and_itl() -> None:
    r = rec()
    assert ttft(r) == pytest.approx(0.25)
    assert e2e(r) == pytest.approx(2.0)
    assert itls(r) == pytest.approx([0.1, 0.2, 0.7])


def test_decode_rate() -> None:
    # (4 - 1) tokens over 2.25 - 1.25 = 1.0 s -> 3 tok/s
    assert decode_rate(rec()) == pytest.approx(3.0)
    # 11 tokens with first/last deltas 2 s apart -> 5 tok/s
    assert decode_rate(rec(completion_tokens=11, delta_ts=[0.0, 0.5, 2.0])) == pytest.approx(5.0)
    assert decode_rate(rec(completion_tokens=1, delta_ts=[1.25])) is None
    assert decode_rate(rec(completion_tokens=None)) is None
    assert decode_rate(rec(delta_ts=[1.0, 1.0])) is None  # zero span


def test_run_metrics_excludes_warmup_and_errors() -> None:
    run = {
        "measure_window_s": [0.0, 10.0],
        "records": [
            rec(phase="warmup", first_token_t=9.0),
            rec(),
            rec(send_t=2.0, first_token_t=2.75, end_t=4.0),
            rec(ok=False, status=429, error="queue_full", first_token_t=None),
        ],
    }
    m = run_metrics(run, n_boot=50)
    assert (m["n"], m["n_ok"]) == (3, 2)
    assert m["errors"] == {"queue_full": 1}
    assert m["ttft_s"]["p50"] == pytest.approx(0.5)  # median of 0.25, 0.75
    assert m["completion_tokens_total"] == 8
    assert m["throughput_tok_s"] == pytest.approx(0.8)  # 8 tokens / 10 s
    assert m["throughput_req_s"] == pytest.approx(0.2)


def test_missing_fields_stay_none() -> None:
    m = run_metrics(
        {"measure_window_s": [0.0, 1.0], "records": [rec(ok=False, first_token_t=None, error="x")]},
        n_boot=10,
    )
    assert m["ttft_s"]["p50"] is None and m["decode_tok_s"]["p50"] is None
    assert m["ttft_s"]["p50_ci95"] is None
