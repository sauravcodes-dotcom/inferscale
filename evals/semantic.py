"""Optional: semantic similarity (model-based) for CAPM action lists.

Uses sentence-transformers/all-MiniLM-L6-v2 at a pinned revision (license: apache-2.0 per its
Hugging Face model card). Install with `uv sync --group semantic`. When the library is missing,
the step is skipped and reported as such; nothing else depends on it.

Metrics per list, labelled "semantic similarity (model-based)": greedy cosine coverage (mean
over reference actions of the best cosine to any predicted action), precision (the reverse)
and their F1.

Usage: uv run python -m evals.semantic RESULTS.jsonl [--out summary.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE = "apache-2.0"
LABEL = "semantic similarity (model-based)"

Vec = Sequence[float]


def available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def cosine(a: Vec, b: Vec) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def greedy_cosine(pred_vecs: Sequence[Vec], ref_vecs: Sequence[Vec]) -> dict[str, float]:
    if not pred_vecs or not ref_vecs:
        return {"coverage": 0.0, "precision": 0.0, "f1": 0.0}
    sim = [[cosine(p, r) for r in ref_vecs] for p in pred_vecs]
    cov = sum(max(sim[i][j] for i in range(len(pred_vecs))) for j in range(len(ref_vecs))) / len(
        ref_vecs
    )
    prec = sum(max(row) for row in sim) / len(pred_vecs)
    f1 = 2 * cov * prec / (cov + prec) if cov + prec > 0 else 0.0
    return {"coverage": cov, "precision": prec, "f1": f1}


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: per-example records with 'prediction' (parsed or None) and 'reference'."""
    if not available():
        return {
            "label": LABEL,
            "skipped": "sentence-transformers not installed (uv sync --group semantic)",
        }
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION)
    totals: dict[str, float] = {}
    for row in rows:
        pred = row.get("prediction") or {}
        for kind in ("corrective", "preventive"):
            preds = list(pred.get(f"{kind}_actions") or [])
            refs = list(row["reference"][f"{kind}_actions"])
            pv = model.encode(preds).tolist() if preds else []
            rv = model.encode(refs).tolist()
            for k, v in greedy_cosine(pv, rv).items():
                totals[f"{kind}_{k}"] = totals.get(f"{kind}_{k}", 0.0) + v
    n = len(rows)
    return {
        "label": LABEL,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "n": n,
        **{k: v / n for k, v in totals.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Score {LABEL} for a run's per-example JSONL")
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    res = score_rows(rows)
    text = json.dumps(res, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    if "skipped" in res:
        print(f"{LABEL}: skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
