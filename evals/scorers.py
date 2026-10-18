"""Generic scoring utilities: macro-F1, bootstrap confidence intervals, paired McNemar tests."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from typing import Any


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
