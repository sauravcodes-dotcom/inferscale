"""Engine-configuration sweep: restart the engine per variant and run one experiment on each.

An experiment (bench/run_experiment.py) measures a stack that is already running. A sweep
answers "how does an engine setting change latency, throughput and memory": for each variant it
starts llama-server with that variant's env overrides (ENGINE_CACHE_RAM, ENGINE_PARALLEL,
ENGINE_MODEL, ...) and a gateway whose admission limit equals the engine's slots, runs the
experiment, records the engine's own buffer sizes and counters, and stops both processes.

  uv run python -m bench.sweep configs/sweeps/m43_cache_ram.yaml [--force-dirty-env]
         [--only VARIANT ...]

Writes results/bench/<sweep>/<utc_ts>.json and the sweep's markdown report. Requires ports
8000/8081 free. Preflight is checked once, before the first variant, and again (as warnings
recorded in the result) before each later one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from bench.reliability import ENGINE, Procs, engine_healthy, gateway_ready, wait_until
from bench.run_experiment import (
    Experiment,
    _Strict,
    load_experiment,
    load_workload,
    preflight_reasons,
    run_experiment,
)
from evals.dataset import ROOT
from evals.fingerprint import engine_props, git_state
from inferscale.registry import load_registry


class Variant(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = ""
    engine_env: dict[str, str] = Field(default_factory=dict)
    gateway_env: dict[str, str] = Field(default_factory=dict)
    concurrency: list[int] | None = None  # overrides the experiment's levels for this variant
    # Registry version this variant serves (gateway active model + result fingerprint); the
    # engine's loaded file must be that version's file. None = the registry's active version.
    model_version: str | None = None

    @field_validator("engine_env")
    @classmethod
    def _expand_home(cls, env: dict[str, str]) -> dict[str, str]:
        return {k: os.path.expanduser(x) for k, x in env.items()}  # e.g. "~/models/..." paths


class Sweep(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = ""
    experiment: str  # base experiment config (workload, repeats, warm-up, preflight, ...)
    engine_profile: str = "bfcl"
    serving_profile: str = "configs/serving/mac-llamacpp-bfcl-gwv.yaml"
    engine_start_timeout_s: float = 240.0
    settle_s: float = Field(default=30.0, ge=0)  # idle time after the engine is up
    # Gateway env for every variant (a variant's gateway_env wins), e.g. a long queue timeout so
    # concurrency above the slot count measures queueing delay instead of shedding.
    gateway_env: dict[str, str] = Field(default_factory=dict)
    restart_per_run: bool = True  # cold engine (empty caches) for every level x repeat
    # KV cache bytes per context token, for the computed KV column (this llama.cpp build does
    # not log buffer sizes). Qwen3-8B, f16 cache: 36 layers x 8 KV heads x 128 x (K+V) x 2 B.
    kv_bytes_per_token: int | None = 147456
    report: str
    variants: list[Variant]


def load_sweep(path: Path) -> Sweep:
    return Sweep.model_validate(yaml.safe_load(path.read_text()))


def engine_env_value(profile: str, overrides: dict[str, str], key: str) -> str | None:
    """A setting as run_engine.sh resolves it: explicit override, else the profile file."""
    if key in overrides:
        return overrides[key]
    path = ROOT / "configs" / "engine" / f"{profile}.env"
    for line in path.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == key and not line.lstrip().startswith("#"):
            return v.strip()
    return None


def gateway_env(sweep: Sweep, v: Variant) -> dict[str, str]:
    """Admission must match the engine's slots, or the gateway over- or under-admits."""
    slots = engine_env_value(sweep.engine_profile, v.engine_env, "ENGINE_PARALLEL") or "2"
    env = {"INFERSCALE_ADMISSION__MAX_INFLIGHT": slots}
    if v.model_version:
        env["INFERSCALE_ACTIVE_MODEL"] = v.model_version
    return {**env, **sweep.gateway_env, **v.gateway_env}


def model_mismatch(v: Variant, props: dict[str, Any] | None) -> str | None:
    """Refuse to label a run with a registry version the engine is not actually serving."""
    registry = load_registry(ROOT / "configs" / "models" / "registry.yaml", v.model_version)
    want = registry.active_entry.file
    got = Path(str((props or {}).get("model_path") or "")).name
    return None if got == want else f"engine serves {got or 'unknown'!r}, registry expects {want!r}"


def peak(runs: list[dict[str, Any]], key: str) -> float | None:
    xs = [s[key] for r in runs for s in r.get("samples", []) if s.get(key) is not None]
    return max(xs) if xs else None


async def bring_up(sweep: Sweep, v: Variant, procs: Procs, stamp: str) -> tuple[Path, float | None]:
    """Start engine then gateway; returns (engine log, seconds to healthy or None if it failed)."""
    log_name = f"sweep-{sweep.name}-{stamp}-{v.name}-engine.log"
    log_path = procs.start_engine(sweep.engine_profile, v.engine_env, log_name)
    up = await wait_until(engine_healthy, sweep.engine_start_timeout_s, interval=1.0)
    if up is None:
        return log_path, None
    procs.start_gateway(
        gateway_env(sweep, v), sweep.serving_profile, f"sweep-{sweep.name}-{stamp}-gateway.log"
    )
    if await wait_until(gateway_ready, 30) is None:
        procs.stop_all()
        raise SystemExit(f"{v.name}: gateway did not become ready")
    problem = model_mismatch(v, await engine_props(ENGINE))
    if problem:
        procs.stop_all()
        raise SystemExit(f"{v.name}: {problem}")
    if sweep.settle_s:
        await asyncio.sleep(sweep.settle_s)
    return log_path, up


def kv_mib(sweep: Sweep, v: Variant) -> float | None:
    ctx = engine_env_value(sweep.engine_profile, v.engine_env, "ENGINE_CTX")
    if sweep.kv_bytes_per_token is None or ctx is None:
        return None
    return int(ctx) * sweep.kv_bytes_per_token / 2**20


async def run_variant(
    sweep: Sweep, exp: Experiment, v: Variant, procs: Procs, stamp: str
) -> dict[str, Any]:
    reasons = preflight_reasons(exp)
    log_path, up = await bring_up(sweep, v, procs, stamp)
    if up is None:
        exited = procs.engine is not None and procs.engine.poll() is not None
        procs.stop_all()
        tail = log_path.read_text(errors="replace").splitlines()[-5:]
        return {"variant": v.model_dump(), "skipped": True, "exited": exited, "log_tail": tail}
    restarts: list[float | None] = []

    async def cold_start(i: int) -> None:
        # Every run starts from a fresh engine: an empty prompt cache and KV, so runs replaying
        # the same requests are comparable and none benefits from an earlier run's cache.
        if i == 0 or not sweep.restart_per_run:
            return
        procs.stop_all()
        restarts.append((await bring_up(sweep, v, procs, stamp))[1])
        if restarts[-1] is None:
            raise SystemExit(f"{v.name}: engine did not restart")

    vexp = exp.model_copy(
        update={
            "concurrency": v.concurrency or exp.concurrency,
            "model_version": v.model_version or exp.model_version,
            "engine_log": str(log_path.relative_to(ROOT)),
        }
    )
    prompts, rf, wfp = load_workload(vexp.workload)
    try:
        result = await run_experiment(
            vexp,
            prompts=prompts,
            workload_fingerprint=wfp,
            preflight={"warnings": reasons},
            response_format=rf,
            before_run=cold_start,
        )
    finally:
        procs.stop_all()
    return {
        "variant": v.model_dump(),
        "skipped": False,
        "engine_up_s": round(up, 1),
        "engine_restarts_up_s": restarts,
        "engine_log": str(log_path.relative_to(ROOT)),
        "kv_cache_mib_computed": kv_mib(sweep, v),
        "gateway_env": gateway_env(sweep, v),
        "peak_engine_rss_bytes": peak(result["runs"], "engine_rss_bytes"),
        "result": result,
    }


def sweep_result(
    sweep: Sweep, exp: Experiment, started: datetime, variants: list[dict[str, Any]]
) -> dict[str, Any]:
    ran = [v for v in variants if not v["skipped"]]
    done = {v["variant"]["name"] for v in variants}
    wanted = [v.name for v in sweep.variants]
    return {
        "sweep": sweep.model_dump(),
        "experiment": exp.model_dump(),
        "run_kind": "smoke" if exp.smoke else "full",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "complete": all(n in done for n in wanted),
        "valid": bool(ran) and all(v["result"]["valid"] for v in ran),
        "variants": variants,
    }


async def run_sweep(
    sweep: Sweep,
    exp: Experiment,
    only: list[str] | None,
    out: Path,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs the variants, rewriting `out` after each one so an interrupted sweep keeps what it
    finished; `previous` (an interrupted result) supplies variants that are already done."""
    started = datetime.fromisoformat(previous["started_at"]) if previous else datetime.now(UTC)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    done: list[dict[str, Any]] = list(previous["variants"]) if previous else []
    procs = Procs()
    try:
        for v in sweep.variants:
            if (only and v.name not in only) or v.name in {d["variant"]["name"] for d in done}:
                continue
            print(f"== variant {v.name}: {v.engine_env}", file=sys.stderr)
            done.append(await run_variant(sweep, exp, v, procs, stamp))
            partial = json.dumps(sweep_result(sweep, exp, started, done), default=str) + "\n"
            await asyncio.to_thread(out.write_text, partial)  # keep finished variants
            if done[-1]["skipped"]:
                print(f"   skipped: {done[-1]['log_tail']}", file=sys.stderr)
            elif exp.cooldown_s:
                await asyncio.sleep(exp.cooldown_s)
    finally:
        procs.stop_all()
    return sweep_result(sweep, exp, started, done)


def _f(v: float | None, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def render_markdown(res: dict[str, Any]) -> str:
    sw = res["sweep"]
    lines = [f"# Sweep {sw['name']}", "", sw["description"].strip(), ""]
    if res["run_kind"] == "smoke":
        lines += ["> **SMOKE RUN: pipeline check only, not a result.**", ""]
    lines += [
        f"Started {res['started_at']}, git {(res['git'].get('sha') or '?')[:12]}"
        f"{' (dirty)' if res['git'].get('dirty') else ''}. Valid (no swap growth): "
        f"{res['valid']}.",
        "",
        "| variant | conc | ok/n | e2e p50 s [95% CI] | e2e p95 s | req/s | prefill tok/s "
        "| decode tok/s/req | prompt reuse | peak RSS GB | KV MiB (computed) | swap growth MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in res["variants"]:
        name = v["variant"]["name"]
        if v["skipped"]:
            lines.append(f"| {name} | skipped (engine did not start) |" + " |" * 10)
            continue
        kv = v.get("kv_cache_mib_computed")
        for run in v["result"]["runs"]:
            s, e = run["summary"], run.get("engine", {})
            rss = peak([run], "engine_rss_bytes")
            ci = s["e2e_s"].get("p50_ci95")
            ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            reuse = e.get("prompt_cache_reuse_fraction")
            lines.append(
                f"| {name} | {run['concurrency']} | {s['n_ok']}/{s['n']} "
                f"| {_f(s['e2e_s']['p50'], '.2f')} {ci_s} | {_f(s['e2e_s']['p95'], '.2f')} "
                f"| {_f(s['throughput_req_s'], '.3f')} | {_f(e.get('prefill_tok_s'), '.0f')} "
                f"| {_f(e.get('decode_tok_s_per_request'), '.1f')} "
                f"| {_f(reuse * 100 if reuse is not None else None, '.0f')}% "
                f"| {_f(rss / 1e9 if rss else None, '.2f')} | {_f(kv, '.0f')} "
                f"| {_f(run['validity'].get('swap_growth_mb'), '.0f')} |"
            )
    lines += [
        "",
        "e2e: request sent to last byte received, as the client sees it (in gateway_validated "
        "mode the gateway sends a tool call only after validating it, so time to first token "
        "equals e2e). prefill/decode: llama.cpp counters over the run (per-request rates). "
        "Prompt reuse: share of client prompt tokens the engine did not evaluate (prompt cache). "
        "Each run starts from a restarted engine, so reuse is within-run only. KV MiB: computed "
        "from total context (ENGINE_CTX) and the model's KV bytes per token.",
        "",
    ]
    for v in res["variants"]:
        if not v["skipped"] and v["result"]["runs"]:
            errs = {k: n for r in v["result"]["runs"] for k, n in r["summary"]["errors"].items()}
            if errs:
                lines.append(f"- {v['variant']['name']} errors: {errs}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Engine-configuration sweep through the gateway")
    ap.add_argument("config", type=Path)
    ap.add_argument("--force-dirty-env", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these variants")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results" / "bench")
    ap.add_argument("--resume", type=Path, help="continue an interrupted sweep result file")
    args = ap.parse_args(argv)
    sweep = load_sweep(args.config)
    exp = load_experiment(ROOT / sweep.experiment)
    reasons = preflight_reasons(exp)
    blocking = bool(reasons) and exp.preflight.enforce and not args.force_dirty_env
    for r in reasons:
        print(f"preflight {'FAIL' if blocking else 'WARNING'}: {r}", file=sys.stderr)
    if blocking:
        print("refusing to start; fix the above or pass --force-dirty-env", file=sys.stderr)
        return 2
    previous = json.loads(args.resume.read_text()) if args.resume else None
    if previous and previous["sweep"]["name"] != sweep.name:
        raise SystemExit(f"{args.resume} is not a result of sweep {sweep.name}")
    out_dir = args.out_root / sweep.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.resume or out_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    res = asyncio.run(run_sweep(sweep, exp, args.only, out, previous))
    res["preflight"] = {"warnings": reasons, "forced": bool(reasons) and exp.preflight.enforce}
    out.write_text(json.dumps(res, default=str) + "\n")
    report = ROOT / sweep.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(res))
    print(report.read_text())
    print(f"wrote {out} and {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
