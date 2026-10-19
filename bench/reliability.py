"""Reliability validation of InferScale against the real llama.cpp engine.

Starts its own engine and gateway processes so it can overload, cancel, crash, restart and
drain them, and records pass/fail checks with evidence (HTTP statuses, metric deltas, timings):

  overload     Poisson arrivals above capacity: immediate 429 + Retry-After, no 5xx, engine
               concurrency never above its slots, queue depth never above its bound.
  deadline     a prompt whose prefill exceeds a short first-token deadline: fast 504 and the
               engine slot is released (upstream request cancelled).
  cancel       a client disconnects mid-stream: the engine stops generating for it.
  engine_crash engine SIGKILLed mid-stream: in-band stream error, /readyz -> 503, new requests
               get a fast 503 engine_unavailable; after restart /readyz -> 200 and requests succeed.
  drain        SIGTERM to the gateway under load: in-flight requests complete, new ones get 503,
               the process exits within the drain budget.

  uv run python -m bench.reliability [--scenarios overload deadline ...]
Requires ports 8000/8081 free (it starts and stops the processes itself).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.harness import RequestRecord, Target, poisson_offsets, stream_one
from bench.memory_sampler import parse_prom
from evals.dataset import ROOT
from evals.fingerprint import git_state, host_state

GW = "http://127.0.0.1:8000"
ENGINE = "http://127.0.0.1:8081"
LOGS = ROOT / "logs"
OUT = ROOT / "results" / "reliability"
SHORT = [{"role": "user", "content": "Reply with the word ok."}]
LONG_GEN = [{"role": "user", "content": "Count from 1 to 400, one number per line, nothing else."}]


class Procs:
    def __init__(self) -> None:
        self.engine: subprocess.Popen[bytes] | None = None
        self.gateway: subprocess.Popen[bytes] | None = None

    def start_engine(
        self,
        profile: str = "bfcl",
        env_overrides: dict[str, str] | None = None,
        log_name: str = "reliability-engine.log",
    ) -> Path:
        env = {**os.environ, "ENGINE_PROFILE": profile, **(env_overrides or {})}
        LOGS.mkdir(exist_ok=True)
        log_path = LOGS / log_name
        log = log_path.open("ab")
        self.engine = subprocess.Popen(
            [str(ROOT / "scripts" / "run_engine.sh")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        return log_path

    def start_gateway(
        self,
        overrides: dict[str, str] | None = None,
        config_file: str = "configs/serving/mac-llamacpp-bfcl-gwv.yaml",
        log_name: str = "reliability-gateway.log",
    ) -> None:
        env = {**os.environ, "INFERSCALE_CONFIG_FILE": config_file, **(overrides or {})}
        LOGS.mkdir(exist_ok=True)
        log = (LOGS / log_name).open("ab")
        self.gateway = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--factory",
                "inferscale.api.app:app_factory",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--timeout-graceful-shutdown",
                "35",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )

    @staticmethod
    def _stop(
        p: subprocess.Popen[bytes] | None, sig: int = signal.SIGTERM, timeout: float = 40
    ) -> float | None:
        if p is None or p.poll() is not None:
            return None
        t0 = time.monotonic()
        p.send_signal(sig)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        return time.monotonic() - t0

    def stop_gateway(self, sig: int = signal.SIGTERM) -> float | None:
        out = self._stop(self.gateway, sig)
        self.gateway = None
        return out

    def stop_engine(self, sig: int = signal.SIGTERM) -> float | None:
        out = self._stop(self.engine, sig)
        self.engine = None
        return out

    def stop_all(self) -> None:
        self.stop_gateway()
        self.stop_engine()


async def wait_until(
    pred: Callable[[], Awaitable[bool]], within_s: float, interval: float = 0.25
) -> float | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < within_s:
        try:
            if await pred():
                return time.monotonic() - t0
        except httpx.HTTPError:
            pass
        await asyncio.sleep(interval)
    return None


async def engine_healthy() -> bool:
    async with httpx.AsyncClient(timeout=2) as c:
        return bool((await c.get(f"{ENGINE}/health")).json().get("status") == "ok")


async def ready_status() -> int:
    async with httpx.AsyncClient(timeout=3) as c:
        return (await c.get(f"{GW}/readyz")).status_code


async def gateway_ready() -> bool:
    return await ready_status() == 200


async def metrics(url: str) -> str:
    async with httpx.AsyncClient(timeout=3) as c:
        return (await c.get(f"{url}/metrics")).text


def prom(text: str, *names: str) -> dict[str, float | None]:
    return parse_prom(text, names)


def counter(text: str, name: str) -> dict[str, float]:
    """All label sets of a counter: {'outcome="ok"': 3.0, ...}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith(name + "{"):
            labels = line[len(name) + 1 : line.index("}")]
            out[labels] = float(line.split()[-1])
    return out


def check(name: str, ok: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "pass": bool(ok), "detail": detail}


async def up(procs: Procs, overrides: dict[str, str] | None = None) -> None:
    if procs.engine is None:
        procs.start_engine()
        if await wait_until(engine_healthy, 180) is None:
            raise SystemExit("engine did not become healthy")
    procs.start_gateway(overrides)
    if await wait_until(gateway_ready, 30) is None:
        raise SystemExit("gateway did not become ready")


# --- scenarios -----------------------------------------------------------------------


async def overload(procs: Procs) -> dict[str, Any]:
    """Arrivals at ~3x capacity for 30 s; admission must shed load instead of queueing forever."""
    await up(
        procs,
        {"INFERSCALE_ADMISSION__MAX_QUEUE": "4", "INFERSCALE_ADMISSION__QUEUE_TIMEOUT_S": "5"},
    )
    rate, duration = 3.0, 30.0
    n = int(rate * duration)
    offsets = poisson_offsets(n, rate, seed=7)
    target = Target("gateway", GW, model="qwen3-8b-q4km-v1")
    body = target.body(LONG_GEN, 96, 0.0)
    before = await metrics(GW)
    samples: list[dict[str, float | None]] = []
    stop = asyncio.Event()

    async def sample() -> None:
        while not stop.is_set():
            try:
                g = prom(await metrics(GW), "inferscale_inflight", "inferscale_queue_depth")
                e = prom(await metrics(ENGINE), "llamacpp:requests_processing")
                samples.append({**g, **e})
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)

    recs = [RequestRecord(index=i, phase="measure") for i in range(n)]
    statuses: Counter[str] = Counter()
    retry_after: list[str] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=GW, timeout=120) as client:
        sampler = asyncio.create_task(sample())

        async def fire(i: int) -> None:
            await asyncio.sleep(max(0.0, offsets[i] - (time.perf_counter() - t0)))
            if i % 3 == 0:  # a share of non-stream requests to read Retry-After headers
                r = await client.post("/v1/chat/completions", json={**body, "stream": False})
                recs[i].status = r.status_code
                if r.status_code == 429:
                    retry_after.append(r.headers.get("retry-after", ""))
                    recs[i].error = r.json()["error"]["code"]
            else:
                await stream_one(client, body, recs[i], t0)

        await asyncio.gather(*(fire(i) for i in range(n)))
        stop.set()
        await sampler
    after = await metrics(GW)
    for r in recs:
        statuses[str(r.status)] += 1
    rejected = {
        k: v - counter(before, "inferscale_rejected_total").get(k, 0.0)
        for k, v in counter(after, "inferscale_rejected_total").items()
    }

    def lat(code: str) -> list[float]:
        return sorted(r.end_t - r.send_t for r in recs if r.error == code and r.end_t and r.send_t)

    full, timed_out = lat("queue_full"), lat("queue_timeout")
    ok = [r for r in recs if r.status == 200 and r.error is None]
    max_engine = (
        max((s.get("llamacpp:requests_processing") or 0) for s in samples) if samples else None
    )
    max_queue = max((s.get("inferscale_queue_depth") or 0) for s in samples) if samples else None
    procs.stop_gateway()
    return {
        "load": {"rate_rps": rate, "duration_s": duration, "requests": n, "max_tokens": 96},
        "statuses": dict(statuses),
        "rejected_delta": rejected,
        "max_engine_processing": max_engine,
        "max_queue_depth": max_queue,
        "rejections_by_code": {"queue_full": len(full), "queue_timeout": len(timed_out)},
        "checks": [
            check("some requests were shed with 429", statuses.get("429", 0) > 0, statuses),
            check("no 5xx responses", not any(k.startswith("5") for k in statuses), statuses),
            check(
                "queue_full 429s are immediate (p95 < 0.5 s)",
                bool(full) and full[int(0.95 * (len(full) - 1))] < 0.5,
                {"n": len(full), "p95_s": full[int(0.95 * (len(full) - 1))] if full else None},
            ),
            check(
                "queue_timeout 429s arrive at the 5 s queue timeout (4.5-6.5 s)",
                all(4.5 <= x <= 6.5 for x in timed_out),
                {
                    "n": len(timed_out),
                    "min_s": timed_out[0] if timed_out else None,
                    "max_s": timed_out[-1] if timed_out else None,
                },
            ),
            check(
                "429 responses carry Retry-After",
                bool(retry_after) and all(retry_after),
                retry_after[:5],
            ),
            check(
                "engine never exceeded its 2 slots",
                max_engine is not None and max_engine <= 2,
                max_engine,
            ),
            check(
                "queue depth never exceeded its bound (4)",
                max_queue is not None and max_queue <= 4,
                max_queue,
            ),
            check("admitted requests completed", len(ok) > 0, len(ok)),
        ],
    }


async def deadline(procs: Procs) -> dict[str, Any]:
    """First-token deadline 2 s vs a ~6.5k-token prompt (>20 s prefill): 504, slot released."""
    await up(
        procs,
        {
            "INFERSCALE_DEADLINES__FIRST_TOKEN_TIMEOUT_S": "2",
            "INFERSCALE_LIMITS__MAX_INPUT_CHARS": "40000",
        },
    )
    filler = "The quick brown fox jumps over the lazy dog. " * 650  # ~6.5k tokens
    body = {
        "messages": [{"role": "user", "content": filler + "\nSummarize in one word."}],
        "max_tokens": 16,
        "stream": True,
    }
    async with httpx.AsyncClient(base_url=GW, timeout=60) as c:
        t0 = time.perf_counter()
        r = await c.post("/v1/chat/completions", json=body)
        elapsed = time.perf_counter() - t0
    await asyncio.sleep(0.5)
    g = prom(await metrics(GW), "inferscale_held_permits", "inferscale_inflight")
    e = prom(await metrics(ENGINE), "llamacpp:requests_processing")
    held_after_504 = g.get("inferscale_held_permits")
    # Track engine occupancy vs gateway permits until the engine finishes the abandoned prefill.
    undercount = 0
    freed: float | None = None
    t_idle = time.monotonic()
    while time.monotonic() - t_idle < 90:
        g = prom(await metrics(GW), "inferscale_inflight")
        e = prom(await metrics(ENGINE), "llamacpp:requests_processing")
        busy, inflight = (
            e.get("llamacpp:requests_processing") or 0,
            g.get("inferscale_inflight") or 0,
        )
        if busy > inflight:
            undercount += 1
        if busy == 0:
            freed = time.monotonic() - t_idle
            break
        await asyncio.sleep(0.2)
    released = await wait_until(lambda: _gateway_inflight_zero(), 5, 0.1)
    m = await metrics(GW)
    hold = prom(m, "inferscale_permit_hold_seconds_sum", "inferscale_permit_hold_seconds_count")
    procs.stop_gateway()
    code = r.json().get("error", {}).get("code") if r.status_code != 200 else None
    return {
        "status": r.status_code,
        "error_code": code,
        "elapsed_s": round(elapsed, 2),
        "observation_engine_busy_after_504_s": None if freed is None else round(freed + 0.5, 1),
        "held_permits_after_504": held_after_504,
        "permit_hold_s": hold.get("inferscale_permit_hold_seconds_sum"),
        "samples_engine_busier_than_gateway": undercount,
        "checks": [
            check(
                "504 deadline_exceeded",
                r.status_code == 504 and code == "deadline_exceeded",
                [r.status_code, code],
            ),
            check("returned close to the 2 s deadline (< 4 s)", elapsed < 4, round(elapsed, 2)),
            check(
                "gateway held the permit while the engine kept prefilling",
                held_after_504 == 1,
                held_after_504,
            ),
            check(
                "gateway never counted fewer requests than the engine was running",
                undercount == 0,
                undercount,
            ),
            check(
                "permit released within 5 s of the engine slot freeing",
                released is not None,
                released,
            ),
        ],
    }


async def _gateway_inflight_zero() -> bool:
    v = prom(await metrics(GW), "inferscale_inflight").get("inferscale_inflight")
    return v is not None and v == 0


async def _engine_idle() -> bool:
    v = prom(await metrics(ENGINE), "llamacpp:requests_processing").get(
        "llamacpp:requests_processing"
    )
    return v is not None and v == 0


async def cancel(procs: Procs) -> dict[str, Any]:
    """Client disconnects 1.5 s into a ~400-number stream; the engine must stop that generation."""
    await up(procs)
    before = (
        prom(await metrics(ENGINE), "llamacpp:tokens_predicted_total")[
            "llamacpp:tokens_predicted_total"
        ]
        or 0
    )
    body = {"messages": LONG_GEN, "max_tokens": 1024, "temperature": 0, "stream": True}
    got = 0
    async with httpx.AsyncClient(base_url=GW, timeout=60) as c:
        try:
            async with asyncio.timeout(1.5):
                async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and '"content"' in line:
                            got += 1
        except TimeoutError:
            pass
    idle = await wait_until(lambda: _engine_idle(), 10, 0.2)
    await asyncio.sleep(1.0)
    after = (
        prom(await metrics(ENGINE), "llamacpp:tokens_predicted_total")[
            "llamacpp:tokens_predicted_total"
        ]
        or 0
    )
    procs.stop_gateway()
    generated = after - before
    return {
        "chunks_received_before_disconnect": got,
        "engine_tokens_generated": generated,
        "engine_idle_after_s": None if idle is None else round(idle, 2),
        "checks": [
            check("stream had started before the disconnect", got > 0, got),
            check("engine became idle within 10 s of the disconnect", idle is not None, idle),
            check(
                "engine stopped well short of the 1024-token budget (< 300)",
                generated < 300,
                generated,
            ),
        ],
    }


async def engine_crash(procs: Procs) -> dict[str, Any]:
    await up(procs)
    body = {"messages": LONG_GEN, "max_tokens": 1024, "temperature": 0, "stream": True}
    events: list[str] = []
    stream_status: int | None = None

    async def long_stream() -> None:
        nonlocal stream_status
        async with httpx.AsyncClient(base_url=GW, timeout=60) as c:
            async with c.stream("POST", "/v1/chat/completions", json=body) as resp:
                stream_status = resp.status_code
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        events.append(line[6:])

    task = asyncio.create_task(long_stream())
    await asyncio.sleep(2.0)
    crash_t = time.monotonic()
    procs.stop_engine(signal.SIGKILL)
    await asyncio.wait_for(task, 30)
    in_band = next(
        (json.loads(e)["error"]["code"] for e in events if e.startswith("{") and '"error"' in e),
        None,
    )
    not_ready = await wait_until(lambda: _ready_is(503), 15, 0.2)
    async with httpx.AsyncClient(base_url=GW, timeout=10) as c:
        t0 = time.perf_counter()
        r = await c.post("/v1/chat/completions", json={"messages": SHORT, "max_tokens": 8})
        fail_fast = time.perf_counter() - t0
    m = prom(await metrics(GW), "inferscale_engine_up")
    procs.start_engine()
    restart_t = time.monotonic()
    healthy = await wait_until(engine_healthy, 180)
    ready = await wait_until(lambda: _ready_is(200), 30, 0.2)
    async with httpx.AsyncClient(base_url=GW, timeout=60) as c:
        ok = await c.post("/v1/chat/completions", json={"messages": SHORT, "max_tokens": 8})
    procs.stop_gateway()
    return {
        "stream_status": stream_status,
        "in_band_error": in_band,
        "readyz_503_after_crash_s": None if not_ready is None else round(not_ready, 2),
        "request_during_outage": {
            "status": r.status_code,
            "code": r.json().get("error", {}).get("code"),
            "elapsed_s": round(fail_fast, 3),
        },
        "engine_up_gauge_during_outage": m.get("inferscale_engine_up"),
        "engine_restart_healthy_s": None if healthy is None else round(healthy, 1),
        "recovered_request_status": ok.status_code,
        "crash_to_recovered_s": round(time.monotonic() - crash_t, 1),
        "restart_to_ready_s": round(time.monotonic() - restart_t, 1),
        "checks": [
            check(
                "in-flight stream ended with an in-band engine error",
                in_band in ("engine_error", "engine_unavailable"),
                in_band,
            ),
            check("/readyz reported 503 within 15 s", not_ready is not None, not_ready),
            check(
                "requests during the outage fail fast with 503 engine_unavailable",
                r.status_code == 503 and fail_fast < 3,
                [r.status_code, round(fail_fast, 3)],
            ),
            check(
                "engine_up gauge reads 0 during the outage",
                m.get("inferscale_engine_up") == 0,
                m.get("inferscale_engine_up"),
            ),
            check("/readyz back to 200 after engine restart", ready is not None, ready),
            check("requests succeed after recovery", ok.status_code == 200, ok.status_code),
        ],
    }


async def _ready_is(code: int) -> bool:
    return await ready_status() == code


async def drain(procs: Procs) -> dict[str, Any]:
    await up(procs, {"INFERSCALE_DRAIN_TIMEOUT_S": "30"})
    body = {"messages": LONG_GEN, "max_tokens": 160, "temperature": 0, "stream": True}
    target = Target("gateway", GW)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=GW, timeout=120) as c:
        inflight = [
            asyncio.create_task(
                stream_one(c, body, RequestRecord(index=i, phase="measure"), t0, True)
            )
            for i in range(2)
        ]
        await asyncio.sleep(1.5)
        assert procs.gateway is not None
        term_t = time.monotonic()
        procs.gateway.send_signal(signal.SIGTERM)
        await asyncio.sleep(0.5)
        try:
            ready = await ready_status()
            r = await c.post("/v1/chat/completions", json={"messages": SHORT, "max_tokens": 8})
            new_req = (r.status_code, r.json().get("error", {}).get("code"))
        except httpx.HTTPError as e:
            ready, new_req = -1, (-1, repr(e))
        done = await asyncio.gather(*inflight)
    exit_s = None
    if procs.gateway is not None:
        try:
            procs.gateway.wait(timeout=40)
            exit_s = time.monotonic() - term_t
        except subprocess.TimeoutExpired:
            pass
        procs.gateway = None
    del target
    complete = [
        d.status == 200 and d.error is None and (d.content or "").count("\n") > 50 for d in done
    ]
    return {
        "readyz_during_drain": ready,
        "new_request_during_drain": new_req,
        "inflight_completed": complete,
        "exit_after_sigterm_s": None if exit_s is None else round(exit_s, 1),
        "checks": [
            check("/readyz 503 while draining", ready == 503, ready),
            check("new requests rejected 503 draining", new_req == (503, "draining"), new_req),
            check("both in-flight streams completed fully", all(complete), complete),
            check(
                "process exited within the 30 s drain budget (+5 s)",
                exit_s is not None and exit_s < 35,
                exit_s,
            ),
        ],
    }


SCENARIOS: dict[str, Callable[[Procs], Awaitable[dict[str, Any]]]] = {
    "overload": overload,
    "deadline": deadline,
    "cancel": cancel,
    "engine_crash": engine_crash,
    "drain": drain,
}


async def run(names: list[str]) -> Path:
    procs = Procs()
    results: dict[str, Any] = {}
    started = datetime.now(UTC)
    try:
        for name in names:
            print(f"== {name}", file=sys.stderr)
            t = time.monotonic()
            try:
                res = await SCENARIOS[name](procs)
            except Exception as e:  # record and continue with the next scenario
                res = {"error": repr(e), "checks": [check("scenario ran", False, repr(e))]}
                procs.stop_gateway()
            res["duration_s"] = round(time.monotonic() - t, 1)
            res["pass"] = all(c["pass"] for c in res["checks"])
            results[name] = res
            for c in res["checks"]:
                print(
                    f"   [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}",
                    file=sys.stderr,
                )
    finally:
        procs.stop_all()
    out = {
        "started_at": started.isoformat(timespec="seconds"),
        "fingerprint": {"git": git_state(), "host": host_state()},
        "all_pass": all(r["pass"] for r in results.values()),
        "scenarios": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reliability validation against the real engine")
    ap.add_argument("--scenarios", nargs="*", choices=list(SCENARIOS), default=list(SCENARIOS))
    args = ap.parse_args(argv)
    path = asyncio.run(run(args.scenarios))
    print(f"wrote {path}")
    return 0 if json.loads(path.read_text())["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
