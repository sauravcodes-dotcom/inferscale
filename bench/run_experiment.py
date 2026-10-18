"""Run a benchmark experiment from a YAML config and write results/bench/<exp>/<utc_ts>.json.

Preflight refuses to start (unless --force-dirty-env) when on battery, when swap used is above
the limit, when Chrome/Safari/Ollama are running, or when the dataset is not reviewed. Configs
with `preflight.enforce: false` (the smoke config) only print warnings.

Usage:
  uv run python -m bench.run_experiment configs/experiments/e1_baseline.yaml [--force-dirty-env]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field

from bench import analyze
from bench.harness import Target, run_load
from bench.memory_sampler import MemorySampler, swap_used_mb, swap_validity
from evals.dataset import ROOT, load_jsonl, sha256_file
from evals.fingerprint import engine_props, git_state, host_state
from evals.prompts import triage_v1 as prompt
from inferscale.registry import load_registry

TargetName = Literal["gateway", "engine"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Workload(_Strict):
    dataset: Literal["triage_v1"] = "triage_v1"
    split: Literal["dev", "test"] = "test"
    prompt: Literal["triage_v1"] = "triage_v1"
    n_requests: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = 0.0


class Preflight(_Strict):
    enforce: bool = True
    require_ac: bool = True
    max_swap_mb_start: float = 64.0
    require_reviewed_dataset: bool = True
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
    targets: list[TargetName] = Field(
        default_factory=lambda: list[TargetName](["gateway", "engine"])
    )
    gateway_url: str = "http://127.0.0.1:8000"
    engine_url: str = "http://127.0.0.1:8081"
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
    names = []
    for p in psutil.process_iter(["name"]):
        if p.info["name"]:
            names.append(str(p.info["name"]))
    return names


def preflight_reasons(
    exp: Experiment,
    dataset_items: list[dict[str, Any]],
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
    if pf.require_reviewed_dataset:
        statuses = sorted({it["review_status"] for it in dataset_items})
        if statuses != ["reviewed"]:
            reasons.append(
                f"dataset {exp.workload.dataset}/{exp.workload.split} not reviewed "
                f"(status: {statuses})"
            )
    return reasons


async def check_health(exp: Experiment) -> list[str]:
    problems = []
    async with httpx.AsyncClient(timeout=3) as c:
        if "gateway" in exp.targets:
            try:
                r = await c.get(f"{exp.gateway_url}/readyz")
                if r.status_code != 200:
                    problems.append(f"gateway not ready: {r.status_code} {r.text[:200]}")
            except httpx.HTTPError as e:
                problems.append(f"gateway unreachable at {exp.gateway_url}: {e!r}")
        try:
            r = await c.get(f"{exp.engine_url}/health")
            if r.status_code != 200 or r.json().get("status") != "ok":
                problems.append(f"engine not healthy: {r.status_code}")
        except (httpx.HTTPError, ValueError) as e:
            problems.append(f"engine unreachable at {exp.engine_url}: {e!r}")
    return problems


# --- run -----------------------------------------------------------------------


def target_for(exp: Experiment, name: str, model_version: str) -> Target:
    if name == "gateway":
        return Target("gateway", exp.gateway_url, model=model_version)
    return Target("engine", exp.engine_url)


async def run_experiment(
    exp: Experiment,
    *,
    dataset_items: list[dict[str, Any]],
    dataset_file: Path,
    preflight: dict[str, Any],
    transport_for: Callable[[str], httpx.AsyncBaseTransport | None] = lambda _t: None,
    sampler_factory: Callable[[Experiment], MemorySampler] | None = None,
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> dict[str, Any]:
    registry = load_registry(ROOT / "configs" / "models" / "registry.yaml")
    model_version = exp.model_version or registry.active
    entry = registry.models.get(model_version)
    prompts = [prompt.build_messages(it["input"]) for it in dataset_items]
    n = exp.workload.n_requests or len(prompts)
    loads: list[tuple[str, float]] = (
        [("concurrency", float(c)) for c in exp.concurrency]
        if exp.mode == "closed"
        else [("rate", r) for r in exp.rates]
    )
    started = datetime.now(UTC)
    fingerprint = {
        "git": git_state(),
        "dataset": {
            "name": exp.workload.dataset,
            "split_file": dataset_file.name,
            "sha256": sha256_file(dataset_file),
            "review_status": sorted({it["review_status"] for it in dataset_items}),
        },
        "prompt": prompt.fingerprint(),
        "model": {"version": model_version, **(entry.model_dump() if entry else {})},
        "engine": {
            "registry_version": entry.engine_version if entry else None,
            "props": await engine_props(exp.engine_url),
        },
        "host": host_state(),
    }
    runs: list[dict[str, Any]] = []
    plan = [(load, rep, t) for load in loads for rep in range(exp.repeats) for t in exp.targets]
    for i, ((kind, value), rep, tname) in enumerate(plan):
        target = target_for(exp, tname, model_version)
        log(
            f"[{i + 1}/{len(plan)}] target={tname} {kind}={value:g} "
            f"repeat={rep + 1}/{exp.repeats} n={n}"
        )
        sampler = (
            sampler_factory(exp)
            if sampler_factory
            else MemorySampler(exp.engine_url, exp.gateway_url)
        )
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
                transport=transport_for(tname),
            )
        run["repeat"] = rep
        run["samples"] = sampler.samples
        run["validity"] = swap_validity(sampler.samples, exp.max_swap_growth_mb)
        run["summary"] = analyze.run_metrics(run, n_boot=200, seed=exp.seed)
        s = run["summary"]
        log(
            f"    ok {s['n_ok']}/{s['n']}  ttft p50 {analyze._ms(s['ttft_s']['p50'])} ms  "
            f"decode p50 {analyze._s(s['decode_tok_s']['p50'], 1)} tok/s  "
            f"valid={run['validity']['valid']}"
        )
        runs.append(run)
        if i < len(plan) - 1 and exp.cooldown_s:
            log(f"    cool-down {exp.cooldown_s:g}s")
            await asyncio.sleep(exp.cooldown_s)
    valid = all(r["validity"]["valid"] is not False for r in runs)
    return {
        "run_kind": "smoke" if exp.smoke else "full",
        "note": "SMOKE RUN: pipeline check only, not a result" if exp.smoke else None,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": exp.model_dump(),
        "preflight": preflight,
        "fingerprint": fingerprint,
        "valid": valid,
        "runs": runs,
    }


def write_result(exp: Experiment, result: dict[str, Any], out_root: Path) -> Path:
    out_dir = out_root / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(result["started_at"]).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{ts}.json"
    out.write_text(json.dumps(result) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a benchmark experiment")
    ap.add_argument("config", type=Path)
    ap.add_argument(
        "--force-dirty-env", action="store_true", help="run even if preflight fails (recorded)"
    )
    ap.add_argument("--out-root", type=Path, default=ROOT / "results" / "bench")
    args = ap.parse_args(argv)
    exp = load_experiment(args.config)
    dataset_file = (
        ROOT / "evals" / "datasets" / exp.workload.dataset / f"{exp.workload.split}.jsonl"
    )
    items = load_jsonl(dataset_file)

    reasons = preflight_reasons(exp, items)
    blocking = bool(reasons) and exp.preflight.enforce and not args.force_dirty_env
    for r in reasons:
        print(f"preflight {'FAIL' if blocking else 'WARNING'}: {r}", file=sys.stderr)
    if blocking:
        print(
            "refusing to start; fix the above or pass --force-dirty-env (the run will be marked)",
            file=sys.stderr,
        )
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

    result = asyncio.run(
        run_experiment(exp, dataset_items=items, dataset_file=dataset_file, preflight=preflight)
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
