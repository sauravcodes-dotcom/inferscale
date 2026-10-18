"""Analyze a benchmark result file and write a markdown report.

Latency metrics (per measured, successful request):
- TTFT = first content delta - send; E2E = end - send;
- ITL = gaps between consecutive content deltas (pooled over requests);
- decode rate = (completion_tokens - 1) / (t_last_delta - t_first_delta), None if < 2 tokens.
System throughput = completion tokens of measured requests / measured window.
Missing values are printed as n/a, never estimated.

Usage: uv run python -m bench.analyze [RESULT.json] [--out reports/x.md]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.dataset import ROOT

Rec = dict[str, Any]
QUANTILES = (50, 90, 95, 99)


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default 'linear' method)."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def ttft(r: Rec) -> float | None:
    if r.get("first_token_t") is None or r.get("send_t") is None:
        return None
    return float(r["first_token_t"] - r["send_t"])


def e2e(r: Rec) -> float | None:
    if r.get("end_t") is None or r.get("send_t") is None:
        return None
    return float(r["end_t"] - r["send_t"])


def itls(r: Rec) -> list[float]:
    ts = r.get("delta_ts") or []
    return [b - a for a, b in itertools.pairwise(ts)]


def decode_rate(r: Rec) -> float | None:
    ts = r.get("delta_ts") or []
    n = r.get("completion_tokens")
    if n is None or n < 2 or len(ts) < 2:
        return None
    span = ts[-1] - ts[0]
    return (n - 1) / span if span > 0 else None


def measured(records: Sequence[Rec]) -> list[Rec]:
    return [r for r in records if r.get("phase") == "measure"]


def bootstrap(
    xs: Sequence[float], stat: Callable[[Sequence[float]], float | None], n_boot: int, seed: int
) -> list[float] | None:
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        v = stat([xs[rng.randrange(len(xs))] for _ in range(len(xs))])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [percentile(vals, 2.5) or 0.0, percentile(vals, 97.5) or 0.0]


def dist(xs: Sequence[float], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(xs), "mean": statistics.fmean(xs) if xs else None}
    for q in QUANTILES:
        out[f"p{q}"] = percentile(xs, q)
    out["p50_ci95"] = bootstrap(xs, lambda v: percentile(v, 50), n_boot, seed)
    out["p95_ci95"] = bootstrap(xs, lambda v: percentile(v, 95), n_boot, seed)
    return out


def run_metrics(run: dict[str, Any], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    recs = measured(run["records"])
    ok = [r for r in recs if r.get("ok")]
    window = run.get("measure_window_s") or [None, None]
    span = (window[1] - window[0]) if None not in window else None
    tokens = sum(r.get("completion_tokens") or 0 for r in ok)
    errors: dict[str, int] = {}
    for r in recs:
        if not r.get("ok"):
            key = r.get("error") or f"http_{r.get('status')}"
            errors[key] = errors.get(key, 0) + 1
    return {
        "n": len(recs),
        "n_ok": len(ok),
        "errors": errors,
        "ttft_s": dist([v for r in ok if (v := ttft(r)) is not None], n_boot, seed),
        "e2e_s": dist([v for r in ok if (v := e2e(r)) is not None], n_boot, seed),
        "itl_s": dist([g for r in ok for g in itls(r)], n_boot, seed),
        "decode_tok_s": dist([v for r in ok if (v := decode_rate(r)) is not None], n_boot, seed),
        "completion_tokens_total": tokens,
        "prompt_tokens_total": sum(r.get("prompt_tokens") or 0 for r in ok),
        "window_s": span,
        "throughput_tok_s": tokens / span if span else None,
        "throughput_req_s": len(ok) / span if span else None,
    }


BUFFER_RE = re.compile(r"(\w+(?:_Mapped)?) (model|KV|compute|output) buffer size =\s+([\d.]+) MiB")


def engine_buffers(log_path: Path | None) -> dict[str, float] | None:
    if log_path is None or not log_path.is_file():
        return None
    out: dict[str, float] = {}
    for dev, kind, mib in BUFFER_RE.findall(log_path.read_text(errors="replace")):
        key = f"{kind}:{dev}"
        out[key] = out.get(key, 0.0) + float(mib)
    return out or None


def run_key(run: dict[str, Any]) -> str:
    load = f"c={run['concurrency']}" if run.get("mode") == "closed" else f"rate={run.get('rate')}"
    return f"{run['target']}|{load}"


def analyze(result: dict[str, Any], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in result["runs"]:
        groups.setdefault(run_key(run), []).append(run)
    pooled = {}
    for key, runs in groups.items():
        merged = {**runs[0], "records": [r for run in runs for r in run["records"]]}
        merged["measure_window_s"] = None
        m = run_metrics(merged, n_boot, seed)
        windows = [run_metrics(run, 0)["throughput_tok_s"] for run in runs]
        m["throughput_tok_s"] = (
            statistics.fmean([w for w in windows if w is not None]) if any(windows) else None
        )
        m["valid"] = all(run.get("validity", {}).get("valid") is not False for run in runs)
        m["repeats"] = len(runs)
        pooled[key] = m
    samples = [s for run in result["runs"] for s in run.get("samples", [])]
    rss = [s["engine_rss_bytes"] for s in samples if s.get("engine_rss_bytes") is not None]
    pressure = [s["memory_pressure"] for s in samples if s.get("memory_pressure") is not None]
    thermal = [s["thermal"] for s in samples if s.get("thermal")]
    log = result.get("experiment", {}).get("engine_log")
    return {
        "pooled": pooled,
        "memory": {
            "peak_engine_rss_bytes": max(rss) if rss else None,
            "max_memory_pressure": max(pressure) if pressure else None,
            "thermal_warning_seen": any(
                t.get("thermal_warning") or t.get("performance_warning") for t in thermal
            )
            if thermal
            else None,
            "swap_growth_mb_max": max(
                (run.get("validity", {}).get("swap_growth_mb") or 0.0) for run in result["runs"]
            )
            if any(
                run.get("validity", {}).get("swap_growth_mb") is not None for run in result["runs"]
            )
            else None,
            "engine_buffers_mib": engine_buffers(ROOT / log if log else None),
        },
        "valid": all(m["valid"] for m in pooled.values()),
    }


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 1000:.0f}"


def _s(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci_ms(ci: list[float] | None) -> str:
    return "n/a" if not ci else f"[{ci[0] * 1000:.0f}, {ci[1] * 1000:.0f}]"


def render_markdown(result: dict[str, Any], a: dict[str, Any]) -> str:
    exp = result.get("experiment", {})
    fp = result.get("fingerprint", {})
    host, git, eng = fp.get("host", {}), fp.get("git", {}), fp.get("engine", {})
    smoke = result.get("run_kind") == "smoke"
    lines = [
        f"# {exp.get('name', 'benchmark')}"
        + (" (SMOKE: pipeline check, not a result)" if smoke else ""),
        "",
    ]
    if smoke:
        lines += [
            "> **SMOKE RUN.** Numbers below only prove the pipeline works. Do not cite them.",
            "",
        ]
    if not a["valid"]:
        lines += ["> **INVALID RUN:** swap grew beyond the threshold during at least one run.", ""]
    for w in result.get("preflight", {}).get("warnings", []):
        lines.append(f"> preflight warning: {w}")
    lines += [
        "",
        f"- Started: {result.get('started_at', 'n/a')}; workload: {exp.get('workload', {}).get('split', 'n/a')} split, "
        f"max_tokens {exp.get('workload', {}).get('max_tokens', 'n/a')}, temperature {exp.get('workload', {}).get('temperature', 'n/a')}, "
        f"warm-up {exp.get('warmup', 'n/a')}, repeats {exp.get('repeats', 'n/a')}, cool-down {exp.get('cooldown_s', 'n/a')} s",
        f"- Model: {fp.get('model', {}).get('version', 'n/a')} (sha256 {str(fp.get('model', {}).get('sha256', 'n/a'))[:12]}), "
        f"engine {((eng.get('props') or {}).get('build_info')) or 'n/a'}",
        f"- Host: {host.get('chip', 'n/a')}, {host.get('memory_gb', 'n/a')} GB, macOS {host.get('macos_version', 'n/a')}, power {host.get('power_source', 'n/a')}",
        f"- Git: {str(git.get('sha', 'n/a'))[:10]} (dirty={git.get('dirty', 'n/a')})",
        "",
        "| target | load | ok/n | TTFT p50 | p90 | p95 | p99 (ms) | TTFT p50 95% CI | E2E p50 | p95 (s) | ITL p50 | p95 (ms) | decode p50 (tok/s) | throughput (tok/s) | valid |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in sorted(a["pooled"].items(), key=lambda kv: (kv[0].split("|")[1], kv[0])):
        target, load = key.split("|")
        t, e, i, d = m["ttft_s"], m["e2e_s"], m["itl_s"], m["decode_tok_s"]
        lines.append(
            f"| {target} | {load} | {m['n_ok']}/{m['n']} | {_ms(t['p50'])} | {_ms(t['p90'])} | {_ms(t['p95'])} | {_ms(t['p99'])} | "
            f"{_ci_ms(t['p50_ci95'])} | {_s(e['p50'])} | {_s(e['p95'])} | {_ms(i['p50'])} | {_ms(i['p95'])} | "
            f"{_s(d['p50'], 1)} | {_s(m['throughput_tok_s'], 1)} | {'yes' if m['valid'] else 'NO'} |"
        )
        if m["errors"]:
            lines.append(f"|  | errors: {m['errors']} |" + " |" * 13)
    mem = a["memory"]
    rss = mem["peak_engine_rss_bytes"]
    bufs = mem["engine_buffers_mib"]
    lines += [
        "",
        "**Memory and thermal:**",
        "",
        f"- Peak llama-server RSS: {'n/a' if rss is None else f'{rss / 2**30:.2f} GiB'}",
        f"- Max swap growth during a run: {'n/a' if mem['swap_growth_mb_max'] is None else f'{mem["swap_growth_mb_max"]:.1f} MB'}",
        f"- Max memory pressure level: {mem['max_memory_pressure'] if mem['max_memory_pressure'] is not None else 'n/a'} (1 normal, 2 warn, 4 critical)",
        f"- Thermal/performance warning seen: {'n/a' if mem['thermal_warning_seen'] is None else mem['thermal_warning_seen']}",
        "- Engine buffers (MiB, from log): "
        + (
            "n/a (not in engine log; start the engine with -lv 4 to log them)"
            if not bufs
            else ", ".join(f"{k} {v:.1f}" for k, v in sorted(bufs.items()))
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def latest_result(exp: str) -> Path:
    files = sorted((ROOT / "results" / "bench" / exp).glob("*.json"))
    if not files:
        raise SystemExit(f"no results for {exp}")
    return files[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a benchmark result")
    ap.add_argument("result", nargs="?", type=Path)
    ap.add_argument(
        "--exp", default="e1_baseline", help="pick the latest result of this experiment"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    path = args.result or latest_result(args.exp)
    result = json.loads(path.read_text())
    a = analyze(result)
    md = render_markdown(result, a)
    out = args.out or ROOT / result.get("experiment", {}).get("report", f"reports/{args.exp}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
