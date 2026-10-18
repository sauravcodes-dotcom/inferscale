"""Scoring utilities.

Generic: macro-F1, bootstrap confidence intervals, paired McNemar tests.

PAG task (see score_pag_item / summarize_pag):
- valid structured output: response parsed as JSON and validated against pag_output_v1.json.
  Invalid reasons, first match wins: http_error, timeout, length (finish_reason == "length"),
  parse (not a JSON object), schema (JSON object failing the schema).
- root-cause type / team: exact match and macro-F1 over gold classes ("root-cause extraction
  accuracy": the type is stated verbatim in every Summary, the team in every Description).
- severity: accuracy next to the majority-class baseline of the scored set. Severity is
  independent of every input field in PAG, so it is diagnostic only and excluded from gates.
- ROUGE-L (in-repo): lowercase, tokens = [a-z0-9]+ runs, LCS-based F1 between two strings.
  Per list (corrective, preventive): coverage = mean over reference actions of the best
  ROUGE-L against predicted actions in the same list; precision = the reverse (mean over
  predicted actions of the best match among reference actions); set F1 = harmonic mean.
- root-cause grounding: an item is grounded if any predicted action contains a keyword stem
  from the committed lexicon (evals/lexicons/pag_root_cause_v1.json) for the GOLD root-cause
  type. This is the keyword lexicon (connectivity, provision, hardware, ...), not full
  Root_Cause_Type substring matching; the stricter substring variant is reported separately.
- format fidelity: valid output with no duplicate actions (after case/space normalization),
  within or across the two lists.
Invalid outputs score 0 on every content metric and count in every denominator.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jsonschema


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


# --- PAG task ------------------------------------------------------------------

LEXICON_PATH = Path(__file__).resolve().parent / "lexicons" / "pag_root_cause_v1.json"
INVALID_REASONS = ("http_error", "timeout", "length", "parse", "schema")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 (beta = 1) on lowercase word tokens."""
    c, r = tokenize(candidate), tokenize(reference)
    lcs = lcs_len(c, r)
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def set_scores(preds: Sequence[str], refs: Sequence[str]) -> dict[str, float]:
    """Coverage (reference-side best match), precision (prediction-side), and their F1."""
    if not preds or not refs:
        return {"coverage": 0.0, "precision": 0.0, "f1": 0.0}
    sim = [[rouge_l(p, r) for r in refs] for p in preds]
    coverage = sum(max(sim[i][j] for i in range(len(preds))) for j in range(len(refs))) / len(refs)
    precision = sum(max(row) for row in sim) / len(preds)
    f1 = 2 * coverage * precision / (coverage + precision) if coverage + precision else 0.0
    return {"coverage": coverage, "precision": precision, "f1": f1}


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, list[str]]:
    data: dict[str, list[str]] = json.loads(path.read_text())["keywords"]
    return data


def classify_output(
    text: str | None,
    schema: dict[str, Any],
    finish_reason: str | None = None,
    http_status: int | None = 200,
    error: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed object if valid, invalid reason or None)."""
    if error == "timeout" or http_status == 504:
        return None, "timeout"
    if http_status != 200 or error:
        return None, "http_error"
    if finish_reason == "length":
        return None, "length"
    try:
        obj = json.loads((text or "").strip())
    except ValueError:
        return None, "parse"
    if not isinstance(obj, dict):
        return None, "parse"
    if any(True for _ in jsonschema.Draft202012Validator(schema).iter_errors(obj)):
        return None, "schema"
    return obj, None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def grounded(actions: Sequence[str], gold_type: str, lexicon: dict[str, list[str]]) -> bool:
    text = " ".join(actions).lower()
    return any(k in text for k in lexicon[gold_type])


def grounded_substring(actions: Sequence[str], gold_type: str) -> bool:
    return gold_type.lower() in " ".join(actions).lower()


def score_pag_item(
    item_id: str,
    reference: dict[str, Any],
    parsed: dict[str, Any] | None,
    invalid_reason: str | None,
    lexicon: dict[str, list[str]],
) -> Row:
    valid = parsed is not None and invalid_reason is None
    row: Row = {
        "id": item_id,
        "valid": valid,
        "invalid_reason": None if valid else (invalid_reason or "parse"),
        "gold_severity": reference["severity"],
        "gold_type": reference["root_cause_type"],
        "gold_team": reference["root_cause_team"],
        "pred_severity": parsed.get("severity") if valid and parsed else None,
        "pred_type": parsed.get("root_cause_type") if valid and parsed else None,
        "pred_team": parsed.get("root_cause_team") if valid and parsed else None,
    }
    row["severity_correct"] = row["pred_severity"] == row["gold_severity"]
    row["type_correct"] = row["pred_type"] == row["gold_type"]
    row["team_correct"] = row["pred_team"] == row["gold_team"]
    for kind in ("corrective", "preventive"):
        preds = list(parsed[f"{kind}_actions"]) if valid and parsed else []
        sc = set_scores(preds, reference[f"{kind}_actions"])
        for k, v in sc.items():
            row[f"{kind}_{k}"] = v
    actions = (
        (list(parsed["corrective_actions"]) + list(parsed["preventive_actions"]))
        if valid and parsed
        else []
    )
    row["grounded"] = valid and grounded(actions, reference["root_cause_type"], lexicon)
    row["grounded_substring"] = valid and grounded_substring(actions, reference["root_cause_type"])
    normed = [_norm(a) for a in actions]
    row["format_ok"] = valid and len(normed) == len(set(normed))
    return row


PAG_RATES = (
    "valid",
    "type_correct",
    "team_correct",
    "severity_correct",
    "grounded",
    "grounded_substring",
    "format_ok",
)
PAG_MEANS = tuple(
    f"{k}_{m}" for k in ("corrective", "preventive") for m in ("coverage", "precision", "f1")
)


def _mean_of(name: str) -> Callable[[Sequence[Row]], float]:
    return lambda rows: sum(float(r[name]) for r in rows) / len(rows) if rows else 0.0


def _macro_of(gold: str, pred: str) -> Callable[[Sequence[Row]], float]:
    return lambda rows: macro_f1([r[gold] for r in rows], [r[pred] for r in rows])


def summarize_pag(rows: Sequence[Row], n_boot: int = 1000, seed: int = 1234) -> dict[str, Any]:
    metrics: dict[str, Callable[[Sequence[Row]], float]] = {
        **{m: _mean_of(m) for m in PAG_RATES + PAG_MEANS},
        "type_macro_f1": _macro_of("gold_type", "pred_type"),
        "team_macro_f1": _macro_of("gold_team", "pred_team"),
    }
    out: dict[str, Any] = {
        "n": len(rows),
        "bootstrap": {"n_boot": n_boot, "seed": seed, "ci": 0.95},
    }
    for name, fn in metrics.items():
        lo, hi = bootstrap_ci(rows, fn, n_boot, seed)
        out[name] = {"value": round(fn(rows), 6), "ci95": [round(lo, 6), round(hi, 6)]}
    sev = Counter(r["gold_severity"] for r in rows)
    out["severity_majority_baseline"] = round(max(sev.values()) / len(rows), 6) if rows else None
    out["invalid_breakdown"] = {
        k: sum(1 for r in rows if r["invalid_reason"] == k) for k in INVALID_REASONS
    }
    return out
