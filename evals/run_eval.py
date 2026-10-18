"""Run the triage_v1 eval through the InferScale gateway.

Sequential, temperature 0. Writes results/eval/<model_version>/<utc_ts>[-smoke].json with
per-item rows, a summary with bootstrap CIs, and an environment fingerprint.

Examples:
  uv run python -m evals.run_eval --split dev --limit 5            # smoke (not a result)
  uv run python -m evals.run_eval --split test --response-format json_schema
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.dataset import ROOT, TRIAGE_V1, load_jsonl, load_schema, sha256_file
from evals.fingerprint import engine_props, git_state, host_state
from evals.prompts import triage_v1 as prompt
from evals.scorers import score_item, summarize
from inferscale.registry import load_registry

REGISTRY_PATH = ROOT / "configs" / "models" / "registry.yaml"


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(p / 100 * len(s)) - 1))]


async def run(args: argparse.Namespace, transport: httpx.AsyncBaseTransport | None = None) -> Path:
    dataset_dir: Path = args.dataset_dir
    split_path = dataset_dir / f"{args.split}.jsonl"
    items = load_jsonl(split_path)
    n_total = len(items)
    if args.limit:
        items = items[: args.limit]
    run_kind = "smoke" if len(items) < n_total else "full"
    statuses = sorted({it["review_status"] for it in items})
    if run_kind == "full" and statuses != ["reviewed"] and not args.allow_draft:
        raise SystemExit(
            f"{args.split} has unreviewed items ({statuses}); pass --allow-draft to run anyway"
        )

    schema = load_schema(dataset_dir)
    registry = load_registry(REGISTRY_PATH)
    model_version = args.model_version or registry.active
    entry = registry.models.get(model_version)
    rf = prompt.response_format(args.response_format, schema)

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=args.timeout, transport=transport
    ) as client:
        for i, it in enumerate(items, 1):
            body: dict[str, Any] = {
                "model": model_version,
                "messages": prompt.build_messages(it["input"]),
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            if rf is not None:
                body["response_format"] = rf
            t0 = time.perf_counter()
            output: str | None = None
            error: str | None = None
            status: int | None = None
            usage: dict[str, Any] = {}
            try:
                r = await client.post("/v1/chat/completions", json=body)
                status = r.status_code
                if r.status_code == 200:
                    data = r.json()
                    output = data["choices"][0]["message"]["content"]
                    usage = data.get("usage") or {}
                else:
                    error = r.text[:500]
            except httpx.HTTPError as e:
                error = repr(e)
            latency = time.perf_counter() - t0
            score = score_item(it["id"], it["label"], output, schema)
            rows.append(
                {
                    "id": it["id"],
                    "gold": it["label"],
                    "output": output,
                    "http_status": status,
                    "error": error,
                    "latency_s": round(latency, 4),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "score": score.to_dict(),
                }
            )
            if not args.quiet:
                mark = "ok " if score.all_fields_correct else ("bad" if score.json_valid else "inv")
                print(f"[{i}/{len(items)}] {it['id']} {mark} {latency:.2f}s", file=sys.stderr)

    latencies = [r["latency_s"] for r in rows if r["http_status"] == 200]
    summary = summarize([r["score"] for r in rows], n_boot=args.bootstrap, seed=args.seed)
    summary["requests"] = {
        "ok": sum(1 for r in rows if r["http_status"] == 200),
        "failed": sum(1 for r in rows if r["http_status"] != 200),
        "latency_s_p50": statistics.median(latencies) if latencies else None,
        "latency_s_p95": percentile(latencies, 95),
        "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens_total": sum(r["completion_tokens"] or 0 for r in rows),
    }

    manifest_path = dataset_dir / "manifest.json"
    now = datetime.now(UTC)
    result = {
        "run_kind": run_kind,
        "note": "SMOKE RUN - pipeline check only, not a result" if run_kind == "smoke" else None,
        "started_at": now.isoformat(timespec="seconds"),
        "config": {
            "split": args.split,
            "n_items": len(items),
            "model_version": model_version,
            "base_url": args.base_url,
            "response_format": args.response_format,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "concurrency": 1,
        },
        "fingerprint": {
            "git": git_state(),
            "dataset": {
                "name": "triage_v1",
                "split_file": split_path.name,
                "sha256": sha256_file(split_path),
                "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
                "review_status": statuses,
            },
            "prompt": prompt.fingerprint(),
            "model": {"version": model_version, **(entry.model_dump() if entry else {})},
            "engine": {
                "registry_version": entry.engine_version if entry else None,
                "props": await engine_props(args.engine_url),
            },
            "host": host_state(),
        },
        "summary": summary,
        "items": rows,
    }
    out_dir = Path(args.out_dir) / model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-smoke" if run_kind == "smoke" else ""
    out = (
        out_dir
        / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{args.split}-{args.response_format}{suffix}.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the triage_v1 eval through the gateway")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--model-version", default=None, help="default: registry active model")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="InferScale gateway")
    ap.add_argument(
        "--engine-url",
        default="http://127.0.0.1:8081",
        help="engine, for /props fingerprint ('' to skip)",
    )
    ap.add_argument("--response-format", choices=("none", "json_schema"), default="none")
    ap.add_argument(
        "--limit", type=int, default=None, help="first N items (marks the run as smoke)"
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dataset-dir", type=Path, default=TRIAGE_V1)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "eval"))
    ap.add_argument(
        "--allow-draft", action="store_true", help="allow a full run on unreviewed data"
    )
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = asyncio.run(run(args))
    summary = json.loads(out.read_text())["summary"]
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n",
                    "json_valid",
                    "schema_valid",
                    "all_fields_correct",
                    "category_macro_f1",
                    "requests",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
