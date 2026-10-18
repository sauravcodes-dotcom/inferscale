"""Scoring for triage_v1: validity, per-field accuracy, category macro-F1, bootstrap CIs and
paired McNemar tests.

Definitions:
- json_valid: the whole response (after trimming whitespace) parses as a JSON object.
- schema_valid: json_valid and the object validates against schema.json.
- <field>_correct (severity, component, category, needs_page): json_valid and the field equals
  the gold value exactly (other schema violations don't block field credit).
- all_fields_correct: schema_valid and all four categorical fields correct.
- category_macro_f1: unweighted mean F1 over the gold category classes. Invalid or
  out-of-vocabulary predictions count as misses for the gold class and false positives for none.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import jsonschema

FIELDS = ("severity", "component", "category", "needs_page")
RATE_METRICS = (
    "json_valid",
    "schema_valid",
    "severity_correct",
    "component_correct",
    "category_correct",
    "needs_page_correct",
    "all_fields_correct",
)


@dataclass
class ItemScore:
    id: str
    json_valid: bool
    schema_valid: bool
    severity_correct: bool
    component_correct: bool
    category_correct: bool
    needs_page_correct: bool
    all_fields_correct: bool
    gold_category: str
    pred_category: str | None
    schema_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_output(text: str | None) -> dict[str, Any] | None:
    if text is None:
        return None
    try:
        obj = json.loads(text.strip())
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def score_item(
    item_id: str, gold: dict[str, Any], output: str | None, schema: dict[str, Any]
) -> ItemScore:
    obj = parse_output(output)
    errors: list[str] = []
    if obj is not None:
        errors = sorted(e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(obj))
    schema_valid = obj is not None and not errors
    correct = {f: obj is not None and f in obj and obj[f] == gold[f] for f in FIELDS}
    # bool is a subclass of int in Python: never let 1 == True count for needs_page
    if obj is not None and not isinstance(obj.get("needs_page"), bool):
        correct["needs_page"] = False
    cat_enum = schema["properties"]["category"]["enum"]
    pred_cat = obj.get("category") if obj is not None else None
    return ItemScore(
        id=item_id,
        json_valid=obj is not None,
        schema_valid=schema_valid,
        severity_correct=correct["severity"],
        component_correct=correct["component"],
        category_correct=correct["category"],
        needs_page_correct=correct["needs_page"],
        all_fields_correct=schema_valid and all(correct.values()),
        gold_category=gold["category"],
        pred_category=pred_cat if pred_cat in cat_enum else None,
        schema_errors=errors,
    )


def macro_f1(
    golds: Sequence[str], preds: Sequence[str | None], labels: Sequence[str] | None = None
) -> float:
    classes = sorted(set(golds)) if labels is None else list(labels)
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds, strict=True) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


Row = dict[str, Any]


def _rate(name: str) -> Callable[[Sequence[Row]], float]:
    return lambda rows: sum(1 for r in rows if r[name]) / len(rows) if rows else 0.0


def _macro(rows: Sequence[Row]) -> float:
    return macro_f1([r["gold_category"] for r in rows], [r["pred_category"] for r in rows])


METRICS: dict[str, Callable[[Sequence[Row]], float]] = {
    **{m: _rate(m) for m in RATE_METRICS},
    "category_macro_f1": _macro,
}


def bootstrap_ci(
    rows: Sequence[Row],
    metric: Callable[[Sequence[Row]], float],
    n_boot: int = 1000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI over items (resampled with replacement, fixed seed)."""
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(rows)
    stats = sorted(metric([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[math.floor(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, math.ceil((1 - alpha / 2) * n_boot) - 1)]
    return (lo, hi)


def summarize(rows: Sequence[Row], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(rows),
        "bootstrap": {"n_boot": n_boot, "seed": seed, "ci": 0.95},
    }
    for name, fn in METRICS.items():
        lo, hi = bootstrap_ci(rows, fn, n_boot, seed)
        out[name] = {"value": round(fn(rows), 6), "ci95": [round(lo, 6), round(hi, 6)]}
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b vs c discordant pairs at p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def paired_mcnemar(
    rows_a: Sequence[Row], rows_b: Sequence[Row], metric: str = "all_fields_correct"
) -> dict[str, Any]:
    """Compare two runs on the same items. b = A right & B wrong, c = A wrong & B right."""
    a = {r["id"]: bool(r[metric]) for r in rows_a}
    bmap = {r["id"]: bool(r[metric]) for r in rows_b}
    if set(a) != set(bmap):
        raise ValueError(f"runs cover different items ({len(set(a) ^ set(bmap))} differ)")
    b = sum(1 for i in a if a[i] and not bmap[i])
    c = sum(1 for i in a if not a[i] and bmap[i])
    return {
        "metric": metric,
        "n": len(a),
        "a_rate": sum(a.values()) / len(a) if a else 0.0,
        "b_rate": sum(bmap.values()) / len(bmap) if bmap else 0.0,
        "a_only_correct": b,
        "b_only_correct": c,
        "p_value": mcnemar_exact(b, c),
        "test": "exact binomial McNemar, two-sided",
    }


def compare_result_files(
    path_a: str, path_b: str, metric: str = "all_fields_correct"
) -> dict[str, Any]:
    with open(path_a) as fa, open(path_b) as fb:
        ra, rb = json.load(fa), json.load(fb)
    return paired_mcnemar(
        [r["score"] for r in ra["items"]], [r["score"] for r in rb["items"]], metric
    )
