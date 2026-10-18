"""Per-entry ingestion of BFCL result/score files and leaderboard-style aggregates.

Aggregates follow bfcl_eval.eval_checker.eval_runner_helper (bfcl-eval 2026.3.23):
- non_live_ast = unweighted mean of [simple, multiple, parallel, parallel_multiple], where
  simple = unweighted mean of [simple_python, simple_java, simple_javascript];
- live = count-weighted accuracy over [live_simple, live_multiple, live_parallel,
  live_parallel_multiple];
- irrelevance (non-live), live_irrelevance and live_relevance are reported on their own.
Confidence intervals: percentile bootstrap that resamples entries within each category
(stratified), recomputing the aggregate each time (B=1000, seed 1234).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.scorers import mcnemar_exact

NON_LIVE_SIMPLE = ("simple_python", "simple_java", "simple_javascript")
NON_LIVE_AST = ("multiple", "parallel", "parallel_multiple")
LIVE_AST = ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple")
RELEVANCE = ("irrelevance", "live_irrelevance", "live_relevance")
SINGLE_TURN = NON_LIVE_SIMPLE + NON_LIVE_AST + LIVE_AST + RELEVANCE

# Published reference (BFCL leaderboard CSV, https://gorilla.cs.berkeley.edu/data_overall.csv,
# fetched 2026-10-18). Likely the hosted model with thinking enabled (QwenAPIHandler sends
# enable_thinking=True); full precision. Values in percent.
PUBLISHED_QWEN3_8B = {
    "FC": {
        "non_live_ast": 87.58,
        "simple_python": 72.83,
        "multiple": 96.50,
        "parallel": 92.00,
        "parallel_multiple": 89.00,
        "live": 80.53,
        "live_simple": 84.50,
        "live_multiple": 79.68,
        "live_parallel": 75.00,
        "live_parallel_multiple": 79.17,
    },
    "Prompt": {"non_live_ast": 88.56, "live": 80.09},
}

ERROR_CLASSES = {
    "ast_decoder": "undecodable_output",
    "irrelevance_error": "called_when_irrelevant",
    "relevance_error": "no_call_when_relevant",
    "wrong_func_name": "wrong_function",
    "wrong_count": "wrong_call_count",
    "cannot_find_match": "wrong_call_set",
    "missing_required": "wrong_arguments",
    "missing_optional": "wrong_arguments",
    "unexpected_param": "wrong_arguments",
    "type_error": "wrong_arguments",
    "value_error": "wrong_arguments",
}

Row = dict[str, Any]


def classify_error(error_type: str | None, inference_failed: bool) -> str | None:
    if inference_failed:
        return "serving_error"
    if not error_type:
        return None
    head, _, tail = error_type.partition(":")
    for key in (tail, head):
        if key in ERROR_CLASSES:
            return ERROR_CLASSES[key]
    for key, cls in ERROR_CLASSES.items():
        if key in error_type:
            return cls
    return "other"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def ingest(run_dir: Path, model_dir_name: str) -> list[Row]:
    """One row per generated entry: correctness from the official score files."""
    rows: list[Row] = []
    result_root = run_dir / "result" / model_dir_name
    score_root = run_dir / "score" / model_dir_name
    for result_file in sorted(result_root.rglob("BFCL_v4_*_result.json")):
        category = result_file.name.removeprefix("BFCL_v4_").removesuffix("_result.json")
        score_file = score_root / result_file.parent.name / f"BFCL_v4_{category}_score.json"
        failures: dict[str, dict[str, Any]] = {}
        if score_file.exists():
            for entry in _jsonl(score_file)[1:]:
                failures[entry["id"]] = entry
        for r in _jsonl(result_file):
            res = r.get("result")
            inference_failed = isinstance(res, str) and res.startswith("Error during inference")
            fail = failures.get(r["id"])
            correct = fail is None and not inference_failed
            rows.append(
                {
                    "id": r["id"],
                    "category": category,
                    "correct": correct,
                    "error_type": None if correct else (fail or {}).get("error_type"),
                    "error_class": None
                    if correct
                    else classify_error((fail or {}).get("error_type"), inference_failed),
                    "made_call": isinstance(res, list) and len(res) > 0,
                    "input_tokens": r.get("input_token_count"),
                    "output_tokens": r.get("output_token_count"),
                    "latency_s": r.get("latency"),
                }
            )
    return rows


def by_category(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        out[r["category"]].append(r)
    return out


def _acc(rows: Sequence[Row]) -> float | None:
    return sum(1 for r in rows if r["correct"]) / len(rows) if rows else None


def aggregates(cats: Mapping[str, Sequence[Row]]) -> dict[str, float | None]:
    acc = {c: _acc(cats.get(c, [])) for c in SINGLE_TURN}
    out: dict[str, float | None] = dict(acc)
    simple = [v for c in NON_LIVE_SIMPLE if (v := acc[c]) is not None]
    out["non_live_simple"] = sum(simple) / len(simple) if simple else None
    parts = [out["non_live_simple"], *(acc[c] for c in NON_LIVE_AST)]
    # BFCL counts a missing category as 0 in this mean; we report None instead of a biased value.
    out["non_live_ast"] = (
        sum(p for p in parts if p is not None) / 4 if all(p is not None for p in parts) else None
    )
    live_rows = [r for c in LIVE_AST for r in cats.get(c, [])]
    out["live"] = _acc(live_rows)
    return out


def stratified_bootstrap(
    rows: Sequence[Row],
    stat: Callable[[Mapping[str, Sequence[Row]]], dict[str, float | None]],
    n_boot: int = 1000,
    seed: int = 1234,
) -> dict[str, list[float] | None]:
    cats = by_category(rows)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        resampled = {c: [rs[rng.randrange(len(rs))] for _ in rs] for c, rs in cats.items()}
        for k, v in stat(resampled).items():
            if v is not None:
                samples[k].append(v)
    out: dict[str, list[float] | None] = {}
    for k in stat(cats):
        vals = sorted(samples.get(k, []))
        out[k] = (
            [vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]]
            if vals
            else None
        )
    return out


def summarize(rows: Sequence[Row], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    cats = by_category(rows)
    point = aggregates(cats)
    ci = stratified_bootstrap(rows, aggregates, n_boot, seed)
    metrics = {
        k: {
            "value": None if v is None else round(v, 6),
            "ci95": None if ci.get(k) is None else [round(x, 6) for x in ci[k] or []],
            "n": len(cats.get(k, [])) if k in SINGLE_TURN else None,
        }
        for k, v in point.items()
    }
    errors = Counter(r["error_class"] for r in rows if not r["correct"])
    by_cat_errors = {
        c: dict(Counter(r["error_class"] for r in rs if not r["correct"])) for c, rs in cats.items()
    }
    return {
        "n": len(rows),
        "metrics": metrics,
        "error_classes": dict(errors.most_common()),
        "error_classes_by_category": by_cat_errors,
        "serving_errors": errors.get("serving_error", 0),
        "bootstrap": {"n_boot": n_boot, "seed": seed, "stratified_by": "category"},
    }


def paired(rows_a: Sequence[Row], rows_b: Sequence[Row]) -> dict[str, Any]:
    """Exact McNemar on entries present in both runs (B vs A), overall and per category."""
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    common = sorted(set(a) & set(b))
    out: dict[str, Any] = {"n_common": len(common)}

    def test(ids: list[str]) -> dict[str, Any]:
        a_only = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
        b_only = sum(1 for i in ids if b[i]["correct"] and not a[i]["correct"])
        return {
            "n": len(ids),
            "a_acc": _acc([a[i] for i in ids]),
            "b_acc": _acc([b[i] for i in ids]),
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "p_value": mcnemar_exact(a_only, b_only),
        }

    out["overall"] = test(common)
    cats = sorted({a[i]["category"] for i in common})
    out["by_category"] = {c: test([i for i in common if a[i]["category"] == c]) for c in cats}
    return out
