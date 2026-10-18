"""Run BFCL single-turn categories through the InferScale gateway with the official scorer.

  uv run python -m evals.bfcl.run --partition dev [--limit-per-category 5]
  uv run python -m evals.bfcl.run --partition test --i-am-running-final-test

Generation and scoring run in the isolated BFCL environment (tools/bfcl). Outputs:
- raw BFCL result/score files (model outputs + dataset text): results/bfcl/raw/<run_id>/
  (git-ignored)
- metrics-only summary: results/bfcl/<model_version>/<run_id>.json (smoke: results/bfcl/smoke/)

The held-out partition is refused unless --i-am-running-final-test is passed and the current
run configuration matches the one frozen in configs/eval/bfcl_frozen.yaml (see evals.bfcl.freeze).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.analyze import percentile
from bench.memory_sampler import MemorySampler, parse_prom, swap_validity
from evals.bfcl import freeze
from evals.bfcl.metrics import SINGLE_TURN, ingest, summarize
from evals.dataset import ROOT, sha256_file
from evals.fingerprint import engine_props, git_state, host_state
from inferscale.registry import load_registry

SPLIT_FILE = ROOT / "splits" / "bfcl_v4_single_turn_v1.json"
ADAPTER = ROOT / "tools" / "bfcl" / "inferscale_bfcl.py"
BFCL_PYTHON = ROOT / "tools" / "bfcl" / ".venv" / "bin" / "python"
RESULTS = ROOT / "results" / "bfcl"
TTFT_METRIC = "inferscale_ttft_seconds"


def select_ids(partition: str, categories: list[str], limit: int | None) -> dict[str, list[str]]:
    split = json.loads(SPLIT_FILE.read_text())
    ids = split["partitions"][partition]["ids"]
    return {c: (ids[c][:limit] if limit else ids[c]) for c in categories if ids.get(c)}


def ids_sha256(ids: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def _bfcl(cmd: str, run_dir: Path, args: argparse.Namespace, ids_file: Path | None = None) -> None:
    argv = [
        str(BFCL_PYTHON),
        str(ADAPTER),
        cmd,
        "--run-dir",
        str(run_dir),
        "--model-version",
        args.model_version,
        "--gateway",
        args.gateway,
    ]
    if ids_file is not None:
        argv += ["--ids", str(ids_file)]
    if cmd == "generate":
        argv += ["--threads", str(args.threads), "--temperature", str(args.temperature)]
    log = run_dir / f"{cmd}.log"
    with log.open("w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"bfcl {cmd} failed (exit {proc.returncode}); see {log}")


def ttft_histogram(text: str) -> dict[str, float]:
    """Cumulative bucket counts of the gateway TTFT histogram, plus _count/_sum."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith(f"{TTFT_METRIC}_bucket"):
            le = line.split('le="')[1].split('"')[0]
            out[le] = float(line.split()[-1])
        elif line.startswith((f"{TTFT_METRIC}_count", f"{TTFT_METRIC}_sum")):
            out[line.split()[0].removeprefix(f"{TTFT_METRIC}_")] = float(line.split()[-1])
    return out


def ttft_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """TTFT over this run from gateway histogram deltas; quantiles = bucket upper bounds."""
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    n = d.get("count", 0.0)
    if not n:
        return {"n": 0}
    buckets = sorted(
        ((float("inf") if k == "+Inf" else float(k)), v)
        for k, v in d.items()
        if k not in ("count", "sum")
    )

    def q(p: float) -> float | None:
        for le, c in buckets:
            if c >= p * n:
                return None if le == float("inf") else le
        return None

    return {
        "n": int(n),
        "mean_s": d["sum"] / n,
        "p50_le_s": q(0.5),
        "p95_le_s": q(0.95),
        "note": "quantiles are histogram bucket upper bounds",
    }


async def _scrape(url: str) -> str:
    async with httpx.AsyncClient(timeout=3) as c:
        return (await c.get(f"{url}/metrics")).text


async def run(args: argparse.Namespace) -> Path:
    registry = load_registry(ROOT / "configs" / "models" / "registry.yaml")
    args.model_version = args.model_version or registry.active
    entry = registry.models.get(args.model_version)
    config = await freeze.current_config(args)
    if args.partition == "test":
        if not args.i_am_running_final_test:
            raise SystemExit("refusing: the held-out partition needs --i-am-running-final-test")
        refusal = freeze.test_refusal(config, freeze.load_frozen())
        if refusal:
            raise SystemExit(f"refusing: {refusal}")

    categories = args.categories or list(SINGLE_TURN)
    ids = select_ids(args.partition, categories, args.limit_per_category)
    kind = "smoke" if args.limit_per_category else "full"
    started = datetime.now(UTC)
    suffix = "-smoke" if kind == "smoke" else ""
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{args.partition}{suffix}"
    run_dir = RESULTS / "raw" / run_id
    run_dir.mkdir(parents=True)
    ids_file = run_dir / "ids.json"
    ids_file.write_text(json.dumps(ids, indent=1, sort_keys=True) + "\n")

    before = ttft_histogram(await _scrape(args.gateway))
    t0 = time.perf_counter()
    sampler = MemorySampler(args.engine_url, args.gateway, interval_s=5.0)
    async with sampler:
        await asyncio.to_thread(_bfcl, "generate", run_dir, args, ids_file)
    wall = time.perf_counter() - t0
    after = ttft_histogram(await _scrape(args.gateway))
    await asyncio.to_thread(_bfcl, "evaluate", run_dir, args)

    rows = ingest(run_dir, f"inferscale_{args.model_version}-FC")
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    engine_metrics = parse_prom(await _scrape(args.engine_url), ("llamacpp:prompt_tokens_total",))
    summary = {
        "task": "bfcl_v4_single_turn",
        "run_id": run_id,
        "run_kind": kind,
        "label": "SMOKE: pipeline check, not a result"
        if kind == "smoke"
        else ("held-out test (final)" if args.partition == "test" else "dev (not final)"),
        "partition": args.partition,
        "started_at": started.isoformat(timespec="seconds"),
        "quality": summarize(rows, n_boot=args.bootstrap),
        "serving": {
            "wall_s": round(wall, 1),
            "entries": len(rows),
            "e2e_latency_s": {
                "p50": percentile(lat, 50),
                "p95": percentile(lat, 95),
                "mean": statistics.fmean(lat) if lat else None,
            },
            "input_tokens_total": sum(r["input_tokens"] or 0 for r in rows),
            "output_tokens_total": sum(r["output_tokens"] or 0 for r in rows),
            "gateway_ttft": ttft_delta(before, after),
            "engine_prompt_tokens_total_after": engine_metrics.get("llamacpp:prompt_tokens_total"),
            "memory_validity": swap_validity(sampler.samples),
            "peak_engine_rss_bytes": max((s["engine_rss_bytes"] or 0) for s in sampler.samples)
            if sampler.samples
            else None,
        },
        "config": config,
        "fingerprint": {
            "git": git_state(),
            "split": {
                "file": str(SPLIT_FILE.relative_to(ROOT)),
                "sha256": sha256_file(SPLIT_FILE),
                "ids_sha256": ids_sha256(ids),
                "categories": sorted(ids),
            },
            "model": {"version": args.model_version, **(entry.model_dump() if entry else {})},
            "engine": await engine_props(args.engine_url),
            "host": host_state(),
        },
        "raw_dir": str(run_dir.relative_to(ROOT)),
    }
    out_dir = RESULTS / ("smoke" if kind == "smoke" else args.model_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_id}.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BFCL single-turn evaluation through InferScale")
    ap.add_argument("--partition", choices=("dev", "test"), required=True)
    ap.add_argument("--i-am-running-final-test", action="store_true")
    ap.add_argument("--categories", nargs="*", choices=SINGLE_TURN, default=None)
    ap.add_argument(
        "--limit-per-category", type=int, default=None, help="first N per category (smoke)"
    )
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--engine-url", default="http://127.0.0.1:8081")
    ap.add_argument(
        "--engine-profile", default="bfcl", help="configs/engine/<name>.env the engine runs"
    )
    ap.add_argument("--serving-profile", default="configs/serving/mac-llamacpp-bfcl.yaml")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--bootstrap", type=int, default=1000)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = asyncio.run(run(args))
    s = json.loads(out.read_text())
    m = s["quality"]["metrics"]
    keys = ("non_live_ast", "live", "irrelevance", "live_irrelevance", "live_relevance")
    print(
        json.dumps(
            {
                "label": s["label"],
                **{k: m[k] for k in keys if k in m},
                "error_classes": s["quality"]["error_classes"],
                "serving": s["serving"],
            },
            indent=1,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
