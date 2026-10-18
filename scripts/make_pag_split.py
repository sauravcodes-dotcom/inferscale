"""Build the deterministic PAG split splits/pag_v1.json (dev / pool / test).

Records are grouped by lowercased Summary (records sharing a summary always land in the same
partition) and stratified by Root_Cause_Type. Per type, partition quotas are allocated from the
target sizes by largest remainder; groups are shuffled with the seed and each is assigned to the
partition with the largest remaining quota for that type. Group sizes can make the final sizes
differ from the targets by a few records.

Usage: uv run python scripts/make_pag_split.py [--check]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import ROOT
from evals.datasets.pag import SOURCE, PagRecord, excluded, load_raw, load_records

SPLIT_NAME = "pag_v1"
SEED = 20261018
TARGETS = {"dev": 100, "pool": 250, "test": 500}
OUT = ROOT / "splits" / f"{SPLIT_NAME}.json"


def _quotas(n_type: dict[str, int], targets: dict[str, int]) -> dict[str, dict[str, int]]:
    total = sum(n_type.values())
    quotas: dict[str, dict[str, int]] = {}
    for t, n in sorted(n_type.items()):
        raw = {p: n * size / total for p, size in targets.items()}
        q = {p: int(v) for p, v in raw.items()}
        for p in sorted(raw, key=lambda p: (raw[p] - q[p], p), reverse=True)[: n - sum(q.values())]:
            q[p] += 1
        quotas[t] = q
    return quotas


def split_records(
    records: dict[str, PagRecord], targets: dict[str, int] = TARGETS, seed: int = SEED
) -> dict[str, list[PagRecord]]:
    groups: dict[str, list[PagRecord]] = defaultdict(list)
    for r in records.values():
        groups[r.summary.strip().lower()].append(r)
    by_type: dict[str, list[list[PagRecord]]] = defaultdict(list)
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda r: r.source_index)
        types = {m.root_cause_type for m in members}
        if len(types) != 1:
            raise ValueError(f"summary group {key!r} spans root-cause types {sorted(types)}")
        by_type[members[0].root_cause_type].append(members)
    n_type = {t: sum(len(g) for g in gs) for t, gs in by_type.items()}
    quotas = _quotas(n_type, targets)
    rng = random.Random(seed)
    parts: dict[str, list[PagRecord]] = {p: [] for p in targets}
    order = list(targets)
    for t in sorted(by_type):
        gs = list(by_type[t])
        rng.shuffle(gs)
        filled = dict.fromkeys(targets, 0)
        for g in gs:
            # Largest remaining quota wins; ties go to the earlier partition in `order`.
            p = max(order, key=lambda p: (quotas[t][p] - filled[p], -order.index(p)))
            parts[p].extend(g)
            filled[p] += len(g)
    return {p: sorted(rs, key=lambda r: r.source_index) for p, rs in parts.items()}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def action_overlap(a: list[PagRecord], b: list[PagRecord]) -> dict[str, Any]:
    """Exact (whitespace/case-normalized) action-string overlap of partition a against b."""
    out: dict[str, Any] = {}
    for kind in ("corrective_actions", "preventive_actions"):
        other = {_norm(x) for r in b for x in getattr(r, kind)}
        mine = [_norm(x) for r in a for x in getattr(r, kind)]
        hits = sum(1 for x in mine if x in other)
        recs_hit = sum(1 for r in a if any(_norm(x) in other for x in getattr(r, kind)))
        out[kind] = {
            "actions": len(mine),
            "exact_match_in_other": hits,
            "action_rate": round(hits / len(mine), 4) if mine else None,
            "records_with_any_match": recs_hit,
            "record_rate": round(recs_hit / len(a), 4) if a else None,
        }
    return out


def build() -> dict[str, Any]:
    raw = load_raw()
    records = load_records(raw=raw)
    parts = split_records(records)
    partitions = {}
    for p, rs in parts.items():
        partitions[p] = {
            "n": len(rs),
            "target": TARGETS[p],
            "root_cause_type_counts": dict(sorted(Counter(r.root_cause_type for r in rs).items())),
            "root_cause_team_counts": dict(sorted(Counter(r.root_cause_team for r in rs).items())),
            "severity_counts": dict(sorted(Counter(r.severity for r in rs).items())),
            "summary_groups": len({r.summary.strip().lower() for r in rs}),
            "incident_ids": [r.incident_id for r in rs],
            "source_indices": [r.source_index for r in rs],
        }
    return {
        "name": SPLIT_NAME,
        "source": {k: SOURCE[k] for k in ("repo", "commit", "file", "sha256")}
        | {"n_records": len(raw)},
        "seed": SEED,
        "grouping": "lowercased, stripped Summary (a group never spans partitions)",
        "stratify": "Root_Cause_Type (largest-remainder quotas per type)",
        "universe": {"n": len(records), "source_indices": "100-949"},
        "excluded": excluded(raw),
        "partitions": partitions,
        "residual_overlap": {
            "note": "exact action-string matches after lowercasing and whitespace normalization",
            "test_vs_pool": action_overlap(parts["test"], parts["pool"]),
            "test_vs_dev": action_overlap(parts["test"], parts["dev"]),
            "dev_vs_pool": action_overlap(parts["dev"], parts["pool"]),
        },
    }


def render(split: dict[str, Any]) -> str:
    return json.dumps(split, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build splits/pag_v1.json")
    ap.add_argument("--check", action="store_true", help="verify the committed file is up to date")
    args = ap.parse_args()
    text = render(build())
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is missing or out of date", file=sys.stderr)
            return 1
        print(f"ok: {OUT} up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    s = json.loads(text)
    for p, v in s["partitions"].items():
        print(f"{p}: n={v['n']} (target {v['target']}), summary groups {v['summary_groups']}")
    print(json.dumps(s["residual_overlap"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
