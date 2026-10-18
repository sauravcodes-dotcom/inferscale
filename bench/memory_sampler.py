"""1 Hz background sampler: engine RSS, swap, memory pressure, thermal state, queue metrics.

Each probe is a plain function so tests can substitute it. A run is invalid when swap used grows
by more than `max_swap_growth_mb` (default 32 MB) over the first sample of the run.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import psutil

SWAP_RE = re.compile(r"used = ([\d.]+)M")
DEFAULT_MAX_SWAP_GROWTH_MB = 32.0


def _run(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def swap_used_mb() -> float | None:
    out = _run(["sysctl", "vm.swapusage"])
    m = SWAP_RE.search(out or "")
    return float(m.group(1)) if m else None


def memory_pressure_level() -> int | None:
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical."""
    out = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    return int(out.strip()) if out and out.strip().isdigit() else None


def thermal_state() -> dict[str, Any] | None:
    out = _run(["pmset", "-g", "therm"])
    if out is None:
        return None
    limits = {k: int(v) for k, v in re.findall(r"(\w+_Limit)\s*=\s*(\d+)", out)}
    return {
        "thermal_warning": "No thermal warning level has been recorded" not in out,
        "performance_warning": "No performance warning level has been recorded" not in out,
        **limits,
    }


def find_engine_pid(name: str = "llama-server") -> int | None:
    for p in psutil.process_iter(["name", "pid"]):
        if p.info["name"] == name:
            return int(p.info["pid"])
    return None


def process_rss(pid: int | None) -> int | None:
    if pid is None:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except psutil.Error:
        return None


def parse_prom(text: str, names: tuple[str, ...]) -> dict[str, float | None]:
    out: dict[str, float | None] = dict.fromkeys(names)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in out:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


ENGINE_METRICS = ("llamacpp:requests_processing", "llamacpp:requests_deferred")
GATEWAY_METRICS = ("inferscale_queue_depth", "inferscale_inflight")


async def scrape(
    client: httpx.AsyncClient, url: str | None, names: tuple[str, ...]
) -> dict[str, float | None]:
    if not url:
        return dict.fromkeys(names)
    try:
        r = await client.get(f"{url}/metrics", timeout=0.8)
        return parse_prom(r.text, names) if r.status_code == 200 else dict.fromkeys(names)
    except httpx.HTTPError:
        return dict.fromkeys(names)


Probe = Callable[[], Any]


class MemorySampler:
    def __init__(
        self,
        engine_url: str | None,
        gateway_url: str | None,
        interval_s: float = 1.0,
        engine_pid: int | None = None,
        probes: dict[str, Probe] | None = None,
        scraper: Callable[
            [httpx.AsyncClient, str | None, tuple[str, ...]], Awaitable[dict[str, float | None]]
        ] = scrape,
    ) -> None:
        self.engine_url = engine_url
        self.gateway_url = gateway_url
        self.interval_s = interval_s
        self.engine_pid = engine_pid if engine_pid is not None else find_engine_pid()
        self.probes: dict[str, Probe] = {
            "swap_used_mb": swap_used_mb,
            "memory_pressure": memory_pressure_level,
            "thermal": thermal_state,
            "engine_rss_bytes": lambda: process_rss(self.engine_pid),
            **(probes or {}),
        }
        self.scraper = scraper
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._t0 = time.perf_counter()

    async def sample_once(self, client: httpx.AsyncClient) -> dict[str, Any]:
        s: dict[str, Any] = {"t": round(time.perf_counter() - self._t0, 3), "wall": time.time()}
        for name, probe in self.probes.items():
            s[name] = await asyncio.to_thread(probe)
        s.update(await self.scraper(client, self.engine_url, ENGINE_METRICS))
        s.update(await self.scraper(client, self.gateway_url, GATEWAY_METRICS))
        self.samples.append(s)
        return s

    async def _loop(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                started = time.perf_counter()
                await self.sample_once(client)
                await asyncio.sleep(max(0.0, self.interval_s - (time.perf_counter() - started)))

    async def __aenter__(self) -> MemorySampler:
        self._t0 = time.perf_counter()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        async with httpx.AsyncClient() as client:
            await self.sample_once(client)  # always end with a final sample


def swap_validity(
    samples: list[dict[str, Any]], max_growth_mb: float = DEFAULT_MAX_SWAP_GROWTH_MB
) -> dict[str, Any]:
    """valid=False if swap used grows by more than max_growth_mb over the first sample."""
    swaps = [s["swap_used_mb"] for s in samples if s.get("swap_used_mb") is not None]
    if not swaps:
        return {
            "valid": None,
            "reason": "swap not measured",
            "swap_start_mb": None,
            "swap_growth_mb": None,
        }
    growth = max(swaps) - swaps[0]
    valid = growth <= max_growth_mb
    return {
        "valid": valid,
        "reason": None if valid else f"swap grew {growth:.1f} MB (> {max_growth_mb:g} MB)",
        "swap_start_mb": swaps[0],
        "swap_max_mb": max(swaps),
        "swap_growth_mb": round(growth, 2),
        "threshold_mb": max_growth_mb,
    }
