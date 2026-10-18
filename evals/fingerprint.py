"""Environment fingerprint shared by run_eval and the benchmark harness."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import httpx

from evals.dataset import ROOT


def _cmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    sha = _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status = _cmd(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def host_state() -> dict[str, Any]:
    batt = _cmd(["pmset", "-g", "batt"]) or ""
    power = None
    if "AC Power" in batt:
        power = "AC"
    elif "Battery Power" in batt:
        power = "battery"
    mem = _cmd(["sysctl", "-n", "hw.memsize"])
    return {
        "os": platform.system(),
        "macos_version": platform.mac_ver()[0] or None,
        "chip": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_gb": round(int(mem) / 2**30) if mem and mem.isdigit() else None,
        "power_source": power,
        "battery": batt.splitlines()[1].strip() if len(batt.splitlines()) > 1 else None,
        "python": platform.python_version(),
    }


async def engine_props(engine_url: str) -> dict[str, Any] | None:
    if not engine_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{engine_url}/props")
            r.raise_for_status()
            props = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    return {k: props.get(k) for k in ("build_info", "model_path", "total_slots")}
