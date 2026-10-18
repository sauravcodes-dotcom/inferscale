"""Run the PAG eval through the InferScale gateway (streaming, temperature 0).

Outputs:
- per-example JSONL with predictions and reference fields (contains dataset text, git-ignored):
  results/eval/raw/<model_version>/<ts>-<split>-<prompt>-<rf>[-smoke].jsonl
- summary JSON with metrics, environment fingerprint and the hash of the ID list (committed):
  results/eval/<model_version>/<ts>-<split>-<prompt>-<rf>.json  (smoke runs: results/eval/smoke/)

The test split is refused unless --i-am-running-final-test is passed AND the prompt (prompt
file + output schema, by sha256) is the prompt of the most recent full dev run.

Usage:
  uv run python -m evals.run_eval --split dev [--limit 10]
  uv run python -m evals.run_eval --split test --i-am-running-final-test
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench.analyze import decode_rate, percentile
from bench.harness import RequestRecord, Target, stream_one
from evals.dataset import ROOT, sha256_file
from evals.datasets.pag import SOURCE, load_records
from evals.fingerprint import engine_props, git_state, host_state
from evals.prompts import DEFAULT_PAG_PROMPT, PAG_PROMPTS, get_prompt
from evals.scorers import classify_output, load_lexicon, score_pag_item, summarize_pag
from inferscale.registry import load_registry

SPLIT_FILE = ROOT / "splits" / "pag_v1.json"
RESULTS = ROOT / "results" / "eval"
REGISTRY = ROOT / "configs" / "models" / "registry.yaml"


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def partition_ids(split_file: Path, partition: str) -> list[str]:
    ids: list[str] = json.loads(split_file.read_text())["partitions"][partition]["incident_ids"]
    return ids


def last_dev_prompt(model_version: str, results: Path = RESULTS) -> dict[str, str] | None:
    runs = sorted((results / model_version).glob("*-dev-*.json"))
    for path in reversed(runs):
        s = json.loads(path.read_text())
        if s.get("run_kind") == "full":
            fp: dict[str, str] = s["fingerprint"]["prompt"]
            return fp
    return None


def check_test_allowed(
    args: argparse.Namespace, model_version: str, results: Path = RESULTS
) -> str | None:
    """Return a refusal reason, or None if a test-split run may proceed."""
    if args.split != "test":
        return None
    if not args.i_am_running_final_test:
        return "the test split needs --i-am-running-final-test"
    last = last_dev_prompt(model_version, results)
    if last is None:
        return "no full dev run recorded for this model; run --split dev first"
    prompt = get_prompt(args.prompt_version)
    now = prompt.prompt_sha256()
    if last["prompt_sha256"] != now:
        return (
            f"prompt {prompt.PROMPT_VERSION} ({now[:12]}) is not the prompt of the most recent "
            f"full dev run ({last['prompt_version']}, {last['prompt_sha256'][:12]})"
        )
    return None


async def run(
    args: argparse.Namespace, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[Path, Path]:
    registry = load_registry(REGISTRY)
    model_version = args.model_version or registry.active
    entry = registry.models.get(model_version)
    refusal = check_test_allowed(args, model_version, Path(args.results_dir))
    if refusal:
        raise SystemExit(f"refusing to run: {refusal}")
    prompt = get_prompt(args.prompt_version)

    ids = partition_ids(Path(args.split_file), args.split)
    if args.limit:
        ids = ids[: args.limit]
    run_kind = "smoke" if args.limit else "full"
    records = (
        load_records(Path(args.data_file), verify_sha=False) if args.data_file else load_records()
    )
    lexicon = load_lexicon()
    rf = prompt.response_format(args.response_format)
    target = Target(
        "gateway", args.base_url, model=model_version, extra={"response_format": rf} if rf else {}
    )

    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    suffix = "-smoke" if run_kind == "smoke" else ""
    name = f"{stamp}-{args.split}-{prompt.PROMPT_VERSION}-{args.response_format}{suffix}"
    raw_path = Path(args.results_dir) / "raw" / model_version / f"{name}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, transport=transport
    ) as client:

        async def one(i: int, rid: str) -> tuple[int, dict[str, Any], dict[str, Any]]:
            rec = records[rid]
            body = target.body(prompt.build_messages(rec.model_input()), args.max_tokens, 0.0)
            async with sem:
                r = await stream_one(
                    client, body, RequestRecord(index=i, phase="measure"), t0, capture_content=True
                )
            parsed, reason = classify_output(
                r.content, prompt.SCHEMA, r.finish_reason, r.status, r.error
            )
            score = score_pag_item(rid, rec.reference(), parsed, reason, lexicon)
            ttft = (
                (r.first_token_t - r.send_t)
                if r.first_token_t is not None and r.send_t is not None
                else None
            )
            e2e = (r.end_t - r.send_t) if r.end_t is not None and r.send_t is not None else None
            dr = decode_rate(r.to_dict())
            row = {
                "incident_id": rid,
                "source_index": rec.source_index,
                "split": args.split,
                "model_version": model_version,
                "prompt_version": prompt.PROMPT_VERSION,
                "response_format": args.response_format,
                "raw_prediction": r.content,
                "prediction": parsed,
                "reference": rec.reference(),
                "valid": score["valid"],
                "invalid_reason": score["invalid_reason"],
                "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
                "e2e_ms": round(e2e * 1000, 1) if e2e is not None else None,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "decode_tok_s": round(dr, 2) if dr is not None else None,
                "finish_reason": r.finish_reason,
                "http_status": r.status,
                "error": r.error,
                "score": score,
            }
            if not args.quiet:
                mark = "ok " if score["valid"] else f"INVALID({score['invalid_reason']})"
                print(f"[{i + 1}/{len(ids)}] {rid} {mark} e2e={row['e2e_ms']}ms", file=sys.stderr)
            return i, row, score

        results = await asyncio.gather(*(one(i, rid) for i, rid in enumerate(ids)))
    for _, row, score in sorted(results, key=lambda x: x[0]):
        rows.append(row)
        scores.append(score)
    with raw_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = [r for r in rows if r["http_status"] == 200]

    def dist(key: str) -> dict[str, float | None]:
        xs = [float(r[key]) for r in ok if r[key] is not None]
        return {
            "p50": percentile(xs, 50),
            "p95": percentile(xs, 95),
            "mean": statistics.fmean(xs) if xs else None,
        }

    summary = {
        "task": "pag_v1",
        "run_kind": run_kind,
        "label": "SMOKE: pipeline check, not a result"
        if run_kind == "smoke"
        else f"{args.split} (not final)"
        if args.split == "dev"
        else "test (final)",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "split": args.split,
            "n": len(ids),
            "model_version": model_version,
            "base_url": args.base_url,
            "response_format": args.response_format,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "streaming": True,
        },
        "quality": summarize_pag(scores, n_boot=args.bootstrap, seed=1234),
        "serving": {
            "requests_ok": len(ok),
            "requests_failed": len(rows) - len(ok),
            "ttft_ms": dist("ttft_ms"),
            "e2e_ms": dist("e2e_ms"),
            "decode_tok_s": dist("decode_tok_s"),
            "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in ok),
            "completion_tokens_total": sum(r["completion_tokens"] or 0 for r in ok),
            "finish_reasons": {
                k: sum(1 for r in rows if r["finish_reason"] == k)
                for k in sorted({str(r["finish_reason"]) for r in rows})
            },
            "wall_s": round(time.perf_counter() - t0, 2),
        },
        "fingerprint": {
            "git": git_state(),
            "dataset": {
                "name": "pag",
                "source_commit": SOURCE["commit"],
                "source_sha256": SOURCE["sha256"],
                "split_file": str(Path(args.split_file).relative_to(ROOT))
                if Path(args.split_file).is_relative_to(ROOT)
                else str(args.split_file),
                "split_sha256": sha256_file(Path(args.split_file)),
                "partition": args.split,
                "ids_sha256": ids_sha256(ids),
            },
            "prompt": prompt.fingerprint(),
            "model": {"version": model_version, **(entry.model_dump() if entry else {})},
            "engine": {
                "registry_version": entry.engine_version if entry else None,
                "props": await engine_props(args.engine_url),
            },
            "host": host_state(),
        },
        "raw_jsonl": str(raw_path.relative_to(ROOT))
        if raw_path.is_relative_to(ROOT)
        else str(raw_path),
    }
    out_dir = Path(args.results_dir) / ("smoke" if run_kind == "smoke" else model_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return out, raw_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the PAG eval through the gateway")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--i-am-running-final-test", action="store_true")
    ap.add_argument("--model-version", default=None, help="default: registry active model")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--engine-url",
        default="http://127.0.0.1:8081",
        help="for the /props fingerprint ('' to skip)",
    )
    ap.add_argument("--prompt-version", choices=sorted(PAG_PROMPTS), default=DEFAULT_PAG_PROMPT)
    ap.add_argument("--response-format", choices=("json_schema", "none"), default="json_schema")
    ap.add_argument("--limit", type=int, default=None, help="first N items; marks the run as smoke")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--split-file", default=str(SPLIT_FILE))
    ap.add_argument("--data-file", default=None, help="override the PAG file (tests)")
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out, raw = asyncio.run(run(args))
    s = json.loads(out.read_text())
    q = s["quality"]
    keys = (
        "valid",
        "type_correct",
        "team_correct",
        "severity_correct",
        "corrective_f1",
        "preventive_f1",
        "grounded",
        "format_ok",
    )
    print(
        json.dumps(
            {
                "label": s["label"],
                **{k: q[k] for k in keys},
                "severity_majority_baseline": q["severity_majority_baseline"],
                "invalid_breakdown": q["invalid_breakdown"],
                "serving": s["serving"],
            },
            indent=1,
        )
    )
    print(f"wrote {out}\nwrote {raw} (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
