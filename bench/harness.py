"""Streaming load generator for the gateway (or the engine directly).

Per request it records send time, first content-delta time (TTFT), every content-delta
timestamp (for inter-token latency), end time, usage tokens, HTTP status and error code.
All times are seconds on time.perf_counter(), relative to the run's t0.

Modes:
- closed loop: `concurrency` workers each send the next request as soon as theirs finishes;
- open loop: Poisson arrivals at `rate` req/s from a seeded RNG, regardless of completions.
The first `warmup` requests run first (at the same concurrency) and are marked phase="warmup".
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import httpx

Mode = Literal["closed", "open"]


@dataclass
class RequestRecord:
    index: int
    phase: Literal["warmup", "measure"]
    scheduled_t: float | None = None
    send_t: float | None = None
    first_token_t: float | None = None
    end_t: float | None = None
    delta_ts: list[float] = field(default_factory=list)
    status: int | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None and self.first_token_t is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        d["delta_ts"] = [round(t, 6) for t in self.delta_ts]
        return d


@dataclass
class Target:
    name: str
    base_url: str
    model: str | None = None  # sent as "model" (gateway); the engine ignores it

    def body(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        b: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.model:
            b["model"] = self.model
        return b


def _now(t0: float) -> float:
    return time.perf_counter() - t0


async def stream_one(
    client: httpx.AsyncClient, body: dict[str, Any], rec: RequestRecord, t0: float
) -> RequestRecord:
    rec.send_t = _now(t0)
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            rec.status = resp.status_code
            if resp.status_code != 200:
                raw = await resp.aread()
                try:
                    rec.error = str(json.loads(raw)["error"]["code"])
                except (ValueError, KeyError, TypeError):
                    rec.error = f"http_{resp.status_code}"
                return rec
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    err = chunk["error"]
                    rec.error = (
                        str(err.get("code", "stream_error"))
                        if isinstance(err, dict)
                        else "stream_error"
                    )
                    break
                for choice in chunk.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        t = _now(t0)
                        if rec.first_token_t is None:
                            rec.first_token_t = t
                        rec.delta_ts.append(t)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    rec.prompt_tokens = usage.get("prompt_tokens")
                    rec.completion_tokens = usage.get("completion_tokens")
    except httpx.HTTPError as e:
        rec.error = type(e).__name__
    finally:
        rec.end_t = _now(t0)
    return rec


def poisson_offsets(n: int, rate: float, seed: int) -> list[float]:
    """Arrival offsets (seconds from start) of a seeded Poisson process; first arrival at 0."""
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(n):
        if i:
            t += rng.expovariate(rate)
        out.append(t)
    return out


async def _closed(
    client: httpx.AsyncClient,
    bodies: Sequence[dict[str, Any]],
    recs: list[RequestRecord],
    concurrency: int,
    t0: float,
) -> None:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(bodies)):
        queue.put_nowait(i)

    async def worker() -> None:
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await stream_one(client, bodies[i], recs[i], t0)

    await asyncio.gather(*(worker() for _ in range(concurrency)))


async def _open(
    client: httpx.AsyncClient,
    bodies: Sequence[dict[str, Any]],
    recs: list[RequestRecord],
    rate: float,
    seed: int,
    t0: float,
) -> None:
    start = _now(t0)
    offsets = poisson_offsets(len(bodies), rate, seed)

    async def fire(i: int) -> None:
        scheduled = start + offsets[i]
        recs[i].scheduled_t = scheduled
        delay = scheduled - _now(t0)
        if delay > 0:
            await asyncio.sleep(delay)
        await stream_one(client, bodies[i], recs[i], t0)

    await asyncio.gather(*(fire(i) for i in range(len(bodies))))


async def run_load(
    target: Target,
    prompts: Sequence[list[dict[str, str]]],
    *,
    n_requests: int,
    max_tokens: int,
    temperature: float = 0.0,
    mode: Mode = "closed",
    concurrency: int = 1,
    rate: float | None = None,
    warmup: int = 0,
    seed: int = 1234,
    timeout_s: float = 300.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Run warm-up then the measured requests. Prompts are cycled if n_requests > len(prompts)."""
    if mode == "open" and not rate:
        raise ValueError("open-loop mode needs a rate")
    limits = httpx.Limits(
        max_connections=max(concurrency, 64), max_keepalive_connections=max(concurrency, 64)
    )
    async with httpx.AsyncClient(
        base_url=target.base_url, timeout=timeout_s, limits=limits, transport=transport
    ) as client:
        t0 = time.perf_counter()
        wall_start = time.time()
        warm_bodies = [
            target.body(prompts[i % len(prompts)], max_tokens, temperature) for i in range(warmup)
        ]
        warm_recs = [RequestRecord(index=i, phase="warmup") for i in range(warmup)]
        if warmup:
            await _closed(client, warm_bodies, warm_recs, max(1, min(concurrency, warmup)), t0)
        bodies = [
            target.body(prompts[(warmup + i) % len(prompts)], max_tokens, temperature)
            for i in range(n_requests)
        ]
        recs = [RequestRecord(index=i, phase="measure") for i in range(n_requests)]
        measure_start = _now(t0)
        if mode == "closed":
            await _closed(client, bodies, recs, concurrency, t0)
        else:
            assert rate is not None
            await _open(client, bodies, recs, rate, seed, t0)
        measure_end = _now(t0)
    return {
        "target": target.name,
        "base_url": target.base_url,
        "mode": mode,
        "concurrency": concurrency if mode == "closed" else None,
        "rate": rate if mode == "open" else None,
        "seed": seed,
        "warmup": warmup,
        "wall_start": wall_start,
        "measure_window_s": [round(measure_start, 6), round(measure_end, 6)],
        "records": [r.to_dict() for r in warm_recs + recs],
    }
