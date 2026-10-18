from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from bench.memory_sampler import MemorySampler, parse_prom, swap_validity


def samples(*swaps: float | None) -> list[dict[str, Any]]:
    return [{"swap_used_mb": s} for s in swaps]


def test_swap_growth_above_threshold_is_invalid() -> None:
    v = swap_validity(samples(100.0, 110.0, 133.0, 120.0))
    assert v["valid"] is False and v["swap_growth_mb"] == pytest.approx(33.0)
    assert "33.0 MB" in v["reason"]


def test_swap_growth_at_threshold_is_valid() -> None:
    v = swap_validity(samples(100.0, 132.0))  # exactly 32 MB: "more than 32" is the rule
    assert v["valid"] is True and v["swap_growth_mb"] == pytest.approx(32.0)


def test_swap_shrinking_is_valid_and_missing_is_unknown() -> None:
    assert swap_validity(samples(200.0, 150.0))["valid"] is True
    assert swap_validity(samples(None, None))["valid"] is None
    # the first *measured* sample is the base
    assert swap_validity(samples(None, 10.0, 50.0))["valid"] is False


def test_parse_prom() -> None:
    text = (
        "# HELP x\nllamacpp:requests_processing 2\nllamacpp:requests_deferred 0\n"
        "inferscale_inflight 1.0\nbad line\n"
    )
    out = parse_prom(
        text, ("llamacpp:requests_processing", "llamacpp:requests_deferred", "missing")
    )
    assert out == {
        "llamacpp:requests_processing": 2.0,
        "llamacpp:requests_deferred": 0.0,
        "missing": None,
    }


async def test_sampler_collects_with_stub_probes() -> None:
    swaps = iter([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

    async def scraper(
        _c: httpx.AsyncClient, _u: str | None, names: tuple[str, ...]
    ) -> dict[str, float | None]:
        return dict.fromkeys(names, 1.0)

    sampler = MemorySampler(
        None,
        None,
        interval_s=0.01,
        engine_pid=-1,
        scraper=scraper,
        probes={
            "swap_used_mb": lambda: next(swaps),
            "memory_pressure": lambda: 1,
            "thermal": lambda: {"thermal_warning": False},
            "engine_rss_bytes": lambda: 123,
        },
    )
    async with sampler:
        while len(sampler.samples) < 3:
            await asyncio.sleep(0.005)
    s = sampler.samples
    assert len(s) >= 4  # loop samples + final sample on exit
    assert s[0]["engine_rss_bytes"] == 123 and s[0]["inferscale_queue_depth"] == 1.0
    assert s[0]["llamacpp:requests_processing"] == 1.0
    assert [x["swap_used_mb"] for x in s] == sorted(x["swap_used_mb"] for x in s)
