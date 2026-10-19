"""Run a benchmark experiment from a YAML config and write results/bench/<exp>/<utc_ts>.json.

The harness targets the InferScale gateway only. Preflight refuses to start (unless
--force-dirty-env) when on battery, when swap used is above the limit, or when
Chrome/Safari/Ollama are running. Configs with `preflight.enforce: false` (smoke) only warn.

Usage:
  uv run python -m bench.run_experiment configs/experiments/e1_baseline.yaml [--force-dirty-env]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field

from bench import analyze
from bench.harness import Prompt, Target, run_load
from bench.memory_sampler import MemorySampler, parse_prom, swap_used_mb, swap_validity
from bench.workload import bfcl_requests, pag_prompts
from evals.dataset import ROOT, sha256_file
from evals.fingerprint import engine_props, git_state, host_state
from inferscale.registry import load_registry

Messages = list[dict[str, str]]
ENGINE_COUNTERS = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:n_decode_total",
    "llamacpp:n_busy_slots_per_decode",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Workload(_Strict):
    dataset: Literal["pag", "bfcl"] = "pag"
    split_file: str = "splits/pag_v1.json"
    # bfcl: requests exported by tools/bfcl/export_requests.py (dev only; never the test split)
    categories: list[str] | None = None
    # bfcl: requests are stored grouped by category; a seeded shuffle makes any prefix a mix.
    shuffle_seed: int | None = None
    partition: Literal["dev", "pool", "test"] = "pool"
    prompt: str = "pag_v1"
    response_format: Literal["none", "json_schema"] = "json_schema"
    n_requests: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = 0.0


class Preflight(_Strict):
    enforce: bool = True
    require_ac: bool = True
    max_swap_mb_start: float = 64.0
    forbidden_processes: list[str] = Field(default_factory=list)


class Experiment(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = ""
    smoke: bool = False
    workload: Workload = Workload()
    mode: Literal["closed", "open"] = "closed"
    concurrency: list[int] = Field(default_factory=lambda: [1])
    rates: list[float] = Field(default_factory=list)
    warmup: int = Field(default=5, ge=0)
    repeats: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=120.0, ge=0)
    gateway_url: str = "http://127.0.0.1:8000"
    engine_url: str = "http://127.0.0.1:8081"  # /props and /metrics only; never a load target
    engine_log: str | None = None
    model_version: str | None = None
    max_swap_growth_mb: float = 32.0
    seed: int = 1234
    report: str = "reports/benchmark.md"
    preflight: Preflight = Preflight()


def load_experiment(path: Path) -> Experiment:
    return Experiment.model_validate(yaml.safe_load(path.read_text()))


# --- preflight ---------------------------------------------------------------


def power_source() -> str | None:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "AC Power" in out:
        return "AC"
    if "Battery Power" in out:
        return "battery"
    return None


def running_process_names() -> list[str]:
    return [str(p.info["name"]) for p in psutil.process_iter(["name"]) if p.info["name"]]


def preflight_reasons(
    exp: Experiment,
    power: Callable[[], str | None] = power_source,
    swap: Callable[[], float | None] = swap_used_mb,
    processes: Callable[[], list[str]] = running_process_names,
) -> list[str]:
    pf = exp.preflight
    reasons = []
    if pf.require_ac:
        src = power()
        if src != "AC":
            reasons.append(f"not on AC power (power source: {src or 'unknown'})")
    used = swap()
    if used is None:
        reasons.append("could not read swap usage")
    elif used > pf.max_swap_mb_start:
        reasons.append(
            f"swap used {used:.1f} MB > {pf.max_swap_mb_start:g} MB (restart the Mac first)"
        )
    pats = [re.compile(p) for p in pf.forbidden_processes]
    hits = sorted({n for n in processes() for p in pats if p.search(n)})
    if hits:
        shown = ", ".join(hits[:5]) + (f" (+{len(hits) - 5} more)" if len(hits) > 5 else "")
        reasons.append(f"forbidden processes running: {shown}")
    return reasons


async def check_health(exp: Experiment) -> list[str]:
    problems = []
    async with httpx.AsyncClient(timeout=3) as c:
        try:
            r = await c.get(f"{exp.gateway_url}/readyz")
            if r.status_code != 200:
                problems.append(f"gateway not ready: {r.status_code} {r.text[:200]}")
        except httpx.HTTPError as e:
            problems.append(f"gateway unreachable at {exp.gateway_url}: {e!r}")
    return problems


# --- run -----------------------------------------------------------------------


async def engine_counters(url: str) -> dict[str, float | None]:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            return parse_prom((await c.get(f"{url}/metrics")).text, ENGINE_COUNTERS)
    except httpx.HTTPError:
        return dict.fromkeys(ENGINE_COUNTERS)


def engine_delta(
    before: dict[str, float | None], after: dict[str, float | None], client_prompt_tokens: int
) -> dict[str, float | None]:
    """Engine-side work during one run, from llama.cpp's cumulative counters.

    `prompt_tokens_total` counts only tokens the engine evaluated, so prompt tokens reused from
    its prompt cache are the gap to what clients sent. The seconds counters sum per-request time
    (concurrent slots both accrue), so the rates are per-request rates, not aggregate throughput.
    """

    def d(name: str) -> float | None:
        a, b = before.get(f"llamacpp:{name}"), after.get(f"llamacpp:{name}")
        return None if a is None or b is None else b - a

    pt, ps, gt, gs = (
        d("prompt_tokens_total"),
        d("prompt_seconds_total"),
        d("tokens_predicted_total"),
        d("tokens_predicted_seconds_total"),
    )
    return {
        "prompt_tokens_evaluated": pt,
        "prompt_seconds": ps,
        "prefill_tok_s": pt / ps if pt and ps else None,
        "tokens_predicted": gt,
        "decode_seconds": gs,
        "decode_tok_s_per_request": gt / gs if gt and gs else None,
        "client_prompt_tokens": client_prompt_tokens,
        "prompt_cache_reuse_fraction": (
            1 - pt / client_prompt_tokens if pt is not None and client_prompt_tokens else None
        ),
        "busy_slots_per_decode": after.get("llamacpp:n_busy_slots_per_decode"),
    }


async def run_experiment(
    exp: Experiment,
    *,
    prompts: Sequence[Prompt],
    workload_fingerprint: dict[str, Any],
    preflight: dict[str, Any],
    response_format: dict[str, Any] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sampler_factory: Callable[[Experiment], MemorySampler] | None = None,
    before_run: Callable[[int], Awaitable[None]] | None = None,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> dict[str, Any]:
    registry = load_registry(ROOT / "configs" / "models" / "registry.yaml")
    model_version = exp.model_version or registry.active
    entry = registry.models.get(model_version)
    extra = {"response_format": response_format} if response_format else {}
    target = Target("gateway", exp.gateway_url, model=model_version, extra=extra)
    n = exp.workload.n_requests or len(prompts)
    loads: list[tuple[str, float]] = (
        [("concurrency", float(c)) for c in exp.concurrency]
        if exp.mode == "closed"
        else [("rate", r) for r in exp.rates]
    )
    started = datetime.now(UTC)
    fingerprint = {
        "git": git_state(),
        **workload_fingerprint,
        "model": {"version": model_version, **(entry.model_dump() if entry else {})},
        "engine": {
            "registry_version": entry.engine_version if entry else None,
            "props": await engine_props(exp.engine_url),
        },
        "host": host_state(),
    }
    runs: list[dict[str, Any]] = []
    plan = [(load, rep) for load in loads for rep in range(exp.repeats)]
    for i, ((kind, value), rep) in enumerate(plan):
        if before_run is not None:
            await before_run(i)
        log(f"[{i + 1}/{len(plan)}] {kind}={value:g} repeat={rep + 1}/{exp.repeats} n={n}")
        sampler = (
            sampler_factory(exp)
            if sampler_factory
            else MemorySampler(exp.engine_url, exp.gateway_url)
        )
        counters_before = await engine_counters(exp.engine_url)
        async with sampler:
            run = await run_load(
                target,
                prompts,
                n_requests=n,
                max_tokens=exp.workload.max_tokens,
                temperature=exp.workload.temperature,
                mode=exp.mode,
                concurrency=int(value) if kind == "concurrency" else 1,
                rate=value if kind == "rate" else None,
                warmup=exp.warmup,
                seed=exp.seed + rep,
                transport=transport,
            )
        run["repeat"] = rep
        # Counters cover warm-up too, so the client-side total does as well.
        run["engine"] = engine_delta(
            counters_before,
            await engine_counters(exp.engine_url),
            sum(r.get("prompt_tokens") or 0 for r in run["records"]),
        )
        run["samples"] = sampler.samples
        run["validity"] = swap_validity(sampler.samples, exp.max_swap_growth_mb)
        run["summary"] = analyze.run_metrics(run, n_boot=200, seed=exp.seed)
        s = run["summary"]
        log(
            f"    ok {s['n_ok']}/{s['n']}  ttft p50 {analyze._ms(s['ttft_s']['p50'])} ms  "
            f"e2e p50 {analyze._ms(s['e2e_s']['p50'])} ms  "
            f"decode p50 {analyze._s(s['decode_tok_s']['p50'], 1)} tok/s  "
            f"valid={run['validity']['valid']}"
        )
        runs.append(run)
        if i < len(plan) - 1 and exp.cooldown_s:
            log(f"    cool-down {exp.cooldown_s:g}s")
            await asyncio.sleep(exp.cooldown_s)
    return {
        "run_kind": "smoke" if exp.smoke else "full",
        "note": "SMOKE RUN: pipeline check only, not a result" if exp.smoke else None,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": exp.model_dump(),
        "preflight": preflight,
        "fingerprint": fingerprint,
        "valid": all(r["validity"]["valid"] is not False for r in runs),
        "runs": runs,
    }


def load_workload(
    w: Workload,
) -> tuple[Sequence[Prompt], dict[str, Any] | None, dict[str, Any]]:
    if w.dataset == "bfcl":
        if w.partition != "dev":
            raise SystemExit("bfcl benchmark workloads use the dev partition only")
        reqs = bfcl_requests("dev", w.categories)
        if w.shuffle_seed is not None:
            random.Random(w.shuffle_seed).shuffle(reqs)
        path = ROOT / "results" / "bfcl" / "raw" / "requests-dev.jsonl"
        fp = {
            "workload": {
                "dataset": "bfcl",
                "partition": "dev",
                "categories": w.categories,
                "shuffle_seed": w.shuffle_seed,
                "order_sha256": hashlib.sha256(
                    "\n".join(r["id"] for r in reqs).encode()
                ).hexdigest(),
                "n_requests_available": len(reqs),
                "requests_sha256": sha256_file(path),
                "split_file": "splits/bfcl_v4_single_turn_v1.json",
            }
        }
        return reqs, None, fp
    return pag_prompts(w.split_file, w.partition, w.prompt, w.response_format)


def write_result(exp: Experiment, result: dict[str, Any], out_root: Path) -> Path:
    out_dir = out_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(result["started_at"]).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{ts}.json"
    out.write_text(json.dumps(result) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a benchmark experiment against the gateway")
    ap.add_argument("config", type=Path)
    ap.add_argument("--force-dirty-env", action="store_true", help="run even if preflight fails")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results" / "bench")
    args = ap.parse_args(argv)
    exp = load_experiment(args.config)

    reasons = preflight_reasons(exp)
    blocking = bool(reasons) and exp.preflight.enforce and not args.force_dirty_env
    for r in reasons:
        print(f"preflight {'FAIL' if blocking else 'WARNING'}: {r}", file=sys.stderr)
    if blocking:
        print("refusing to start; fix the above or pass --force-dirty-env", file=sys.stderr)
        return 2
    preflight = {
        "warnings": reasons,
        "forced": bool(reasons) and exp.preflight.enforce,
        "enforced": exp.preflight.enforce,
    }

    problems = asyncio.run(check_health(exp))
    if problems:
        for p in problems:
            print(f"health: {p}", file=sys.stderr)
        return 3

    prompts, rf, wfp = load_workload(exp.workload)
    result = asyncio.run(
        run_experiment(
            exp, prompts=prompts, workload_fingerprint=wfp, preflight=preflight, response_format=rf
        )
    )
    out = write_result(exp, result, args.out_root)
    print(f"wrote {out}", file=sys.stderr)
    a = analyze.analyze(result)
    report = ROOT / exp.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(analyze.render_markdown(result, a))
    print(report.read_text())
    print(f"wrote {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
