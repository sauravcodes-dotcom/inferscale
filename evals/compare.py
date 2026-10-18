"""Paired comparison of two run_eval runs on the same items (B vs A).

- Continuous per-item metrics (ROUGE-L set coverage / precision / F1): mean paired difference
  B - A with a percentile bootstrap 95% CI over items (seed 1234).
- Binary per-item metrics (valid, type, team, severity, grounding, format): rates, B - A, and the
  exact two-sided McNemar test on discordant pairs.
- Action statistics: count distribution per list and mean action length (characters) for A, B
  and the references.

Reads the git-ignored per-example JSONL named in each summary; writes a metrics-only JSON.

Usage: uv run python -m evals.compare SUMMARY_A.json SUMMARY_B.json [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from bench.analyze import percentile
from evals.dataset import ROOT
from evals.scorers import paired_mcnemar

CONTINUOUS = tuple(
    f"{k}_{m}" for k in ("corrective", "preventive") for m in ("f1", "coverage", "precision")
)
BINARY = (
    "valid",
    "type_correct",
    "team_correct",
    "severity_correct",
    "grounded",
    "grounded_substring",
    "format_ok",
)


def load_rows(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    s = json.loads(summary_path.read_text())
    raw = Path(s["raw_jsonl"])
    raw = raw if raw.is_absolute() else ROOT / raw
    rows = [json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
    return s, rows


def paired_diff_ci(
    a: list[float], b: list[float], n_boot: int = 1000, seed: int = 1234
) -> dict[str, Any]:
    d = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "mean_a": statistics.fmean(a),
        "mean_b": statistics.fmean(b),
        "diff": statistics.fmean(d),
        "ci95": [percentile(boots, 2.5), percentile(boots, 97.5)],
        "b_better_items": sum(1 for x in d if x > 0),
        "a_better_items": sum(1 for x in d if x < 0),
    }


def action_stats(lists: list[list[str]]) -> dict[str, Any]:
    lengths = [len(a) for lst in lists for a in lst]
    return {
        "count_distribution": {
            str(k): v for k, v in sorted(Counter(len(lst) for lst in lists).items())
        },
        "mean_count": statistics.fmean(len(lst) for lst in lists) if lists else None,
        "mean_chars": statistics.fmean(lengths) if lengths else None,
        "median_chars": statistics.median(lengths) if lengths else None,
    }


def compare(
    sa: dict[str, Any],
    ra: list[dict[str, Any]],
    sb: dict[str, Any],
    rb: list[dict[str, Any]],
    n_boot: int = 1000,
) -> dict[str, Any]:
    ia = {r["incident_id"]: r for r in ra}
    ib = {r["incident_id"]: r for r in rb}
    if set(ia) != set(ib):
        raise ValueError(f"runs cover different items ({len(set(ia) ^ set(ib))} differ)")
    ids = sorted(ia)
    cont = {
        m: paired_diff_ci(
            [ia[i]["score"][m] for i in ids], [ib[i]["score"][m] for i in ids], n_boot
        )
        for m in CONTINUOUS
    }
    binary = {}
    for m in BINARY:
        res = paired_mcnemar(
            [{"id": i, m: ia[i]["score"][m]} for i in ids],
            [{"id": i, m: ib[i]["score"][m]} for i in ids],
            metric=m,
        )
        res["diff"] = res["b_rate"] - res["a_rate"]
        binary[m] = res
    actions: dict[str, Any] = {}
    for kind in ("corrective", "preventive"):
        key = f"{kind}_actions"
        actions[kind] = {
            "a": action_stats([r["prediction"][key] for r in ra if r["prediction"]]),
            "b": action_stats([r["prediction"][key] for r in rb if r["prediction"]]),
            "reference": action_stats([r["reference"][key] for r in ra]),
        }

    def label(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt_version": s["fingerprint"]["prompt"]["prompt_version"],
            "prompt_sha256": s["fingerprint"]["prompt"]["prompt_sha256"],
            "started_at": s["started_at"],
            "run_kind": s["run_kind"],
            "split": s["config"]["split"],
            "git": s["fingerprint"]["git"],
        }

    return {
        "a": label(sa),
        "b": label(sb),
        "n": len(ids),
        "direction": "B - A",
        "continuous": cont,
        "binary": binary,
        "actions": actions,
        "bootstrap": {"n_boot": n_boot, "seed": 1234},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired comparison of two eval runs")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    sa, ra = load_rows(args.a)
    sb, rb = load_rows(args.b)
    res = compare(sa, ra, sb, rb, args.n_boot)
    out = (
        args.out
        or args.b.parent
        / f"compare-{res['b']['prompt_version']}-vs-{res['a']['prompt_version']}-{args.b.stem}.json"
    )
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
