"""Paired comparison of two BFCL runs on the same entries (B vs A).

Per category: exact McNemar on discordant entries. Aggregates (non-live AST, live, irrelevance
categories): B - A with a paired, category-stratified bootstrap CI (the same resampled entries
are used for both runs). Also the change in error classes.

  uv run python -m evals.bfcl.compare results/bfcl/<model>/<A>.json results/bfcl/<model>/<B>.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.bfcl.metrics import Row, aggregates, cluster_key, paired
from evals.dataset import ROOT

AGGREGATES = ("non_live_ast", "live", "irrelevance", "live_irrelevance")


def load_rows(summary_path: Path) -> tuple[dict[str, Any], list[Row]]:
    s = json.loads(summary_path.read_text())
    rows_file = ROOT / s["raw_dir"] / "rows.jsonl"
    return s, [json.loads(x) for x in rows_file.read_text().splitlines() if x.strip()]


def paired_bootstrap(
    a: list[Row], b: list[Row], n_boot: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    am = {r["id"]: r for r in a}
    bm = {r["id"]: r for r in b}
    ids = sorted(set(am) & set(bm))
    by_cat: dict[str, list[str]] = defaultdict(list)
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for i in ids:
        by_cat[am[i]["category"]].append(i)
        groups[am[i]["category"]][cluster_key(i)].append(i)
    clusters = {c: list(g.values()) for c, g in groups.items()}

    def aggs(sample: dict[str, list[str]], m: dict[str, Row]) -> dict[str, float | None]:
        return aggregates({c: [m[i] for i in v] for c, v in sample.items()})

    point_a, point_b = aggs(by_cat, am), aggs(by_cat, bm)
    rng = random.Random(seed)
    diffs: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        # Same resampled clusters for both runs (paired), live source groups as units.
        sample = {
            c: [i for _ in cl for i in cl[rng.randrange(len(cl))]] for c, cl in clusters.items()
        }
        ra, rb = aggs(sample, am), aggs(sample, bm)
        for k in AGGREGATES:
            if ra.get(k) is not None and rb.get(k) is not None:
                diffs[k].append(rb[k] - ra[k])  # type: ignore[operator]
    out: dict[str, Any] = {}
    for k in AGGREGATES:
        if point_a.get(k) is None or point_b.get(k) is None:
            continue
        d = sorted(diffs[k])
        out[k] = {
            "a": point_a[k],
            "b": point_b[k],
            "diff": point_b[k] - point_a[k],  # type: ignore[operator]
            "ci95": [d[int(0.025 * len(d))], d[min(len(d) - 1, int(0.975 * len(d)))]]
            if d
            else None,
        }
    return out


def compare(a: list[Row], b: list[Row], n_boot: int = 1000) -> dict[str, Any]:
    return {
        "aggregates": paired_bootstrap(a, b, n_boot),
        "mcnemar": paired(a, b),
        "error_classes": {
            "a": dict(Counter(r["error_class"] for r in a if not r["correct"])),
            "b": dict(Counter(r["error_class"] for r in b if not r["correct"])),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired comparison of two BFCL runs")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    sa, ra = load_rows(args.a)
    sb, rb = load_rows(args.b)
    res = {
        "a": {
            "run_id": sa["run_id"],
            "config_sha256": sa["config"]["config_sha256"],
            "serving_profile": sa["config"]["serving_profile"],
        },
        "b": {
            "run_id": sb["run_id"],
            "config_sha256": sb["config"]["config_sha256"],
            "serving_profile": sb["config"]["serving_profile"],
        },
        **compare(ra, rb, args.n_boot),
    }
    out = args.b.parent / f"compare-{sa['run_id']}-vs-{sb['run_id']}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
