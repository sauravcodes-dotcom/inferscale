"""Leakage-aware dev / held-out split of the BFCL v4 single-turn categories.

BFCL has no official dev/test split. Duplicates and near-duplicates exist across entries:
identical user questions (sometimes across non-live and live categories), and live entries
that come from the same source conversation (the middle number of "live_multiple_217-93-0")
and share tool definitions. Entries connected by either relation form one component; whole
components are assigned to a partition, so no question text or live source group is shared
between dev and held-out. Components are shuffled with a fixed seed and assigned greedily so
each category gets about DEV_FRACTION of its entries in dev (at least one when possible).

Reads the data files shipped in the pinned bfcl-eval package (tools/bfcl/.venv) and records
their sha256. Output: splits/bfcl_v4_single_turn_v1.json (IDs only).

Usage: uv run python scripts/make_bfcl_split.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.bfcl.data import BFCL_VERSION, DATA_DIR  # noqa: E402

OUT = ROOT / "splits" / "bfcl_v4_single_turn_v1.json"
SEED = 20261018
DEV_FRACTION = 0.10
CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def question_key(entry: dict[str, Any]) -> str:
    return " || ".join(
        _norm(m["content"]) for turn in entry["question"] for m in turn if m.get("role") == "user"
    )


def live_group(entry_id: str) -> str | None:
    """'live_multiple_217-93-0' -> '93' (source conversation shared across live categories)."""
    if not entry_id.startswith("live_"):
        return None
    tail = entry_id.rsplit("_", 1)[-1]
    parts = tail.split("-")
    return parts[1] if len(parts) == 3 else None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load_entries(data_dir: Path = DATA_DIR) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    entries: dict[str, dict[str, Any]] = {}
    shas: dict[str, str] = {}
    for cat in CATEGORIES:
        for sub in ("", "possible_answer/"):
            p = data_dir / f"{sub}BFCL_v4_{cat}.json"
            if p.exists():
                shas[f"{sub}BFCL_v4_{cat}.json"] = hashlib.sha256(p.read_bytes()).hexdigest()
        for line in (data_dir / f"BFCL_v4_{cat}.json").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                entries[e["id"]] = {"category": cat, **e}
    return entries, shas


def components(entries: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    uf = UnionFind()
    first_by_key: dict[tuple[str, str], str] = {}
    for eid, e in sorted(entries.items()):
        uf.find(eid)
        keys = [("q", question_key(e))]
        g = live_group(eid)
        if g is not None:
            keys.append(("live", g))
        for k in keys:
            if k in first_by_key:
                uf.union(first_by_key[k], eid)
            else:
                first_by_key[k] = eid
    comps: dict[str, list[str]] = defaultdict(list)
    for eid in sorted(entries):
        comps[uf.find(eid)].append(eid)
    return comps


def split(
    entries: dict[str, dict[str, Any]], seed: int = SEED, frac: float = DEV_FRACTION
) -> dict[str, list[str]]:
    comps = components(entries)
    n_cat = Counter(e["category"] for e in entries.values())
    target = {c: max(1, math.floor(n * frac)) if n >= 5 else 0 for c, n in n_cat.items()}
    dev_count: Counter[str] = Counter()
    roots = sorted(comps)
    random.Random(seed).shuffle(roots)
    dev: list[str] = []
    test: list[str] = []
    for root in roots:
        members = comps[root]
        cats = Counter(entries[m]["category"] for m in members)
        fits = all(dev_count[c] + k <= target[c] for c, k in cats.items())
        helps = any(dev_count[c] < target[c] for c in cats)
        if fits and helps:
            dev.extend(members)
            dev_count.update(cats)
        else:
            test.extend(members)
    return {"dev": sorted(dev), "test": sorted(test)}


def by_category(ids: list[str], entries: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for i in ids:
        out[entries[i]["category"]].append(i)
    return out


def build() -> dict[str, Any]:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"{DATA_DIR} missing; run `make bfcl-setup`")
    entries, shas = load_entries()
    comps = components(entries)
    parts = split(entries)
    dev_comp = {r for r, ms in comps.items() if ms[0] in set(parts["dev"])}
    leaks = [
        r
        for r, ms in comps.items()
        if (set(ms) & set(parts["dev"])) and (set(ms) & set(parts["test"]))
    ]
    assert not leaks
    sizes = Counter(len(ms) for ms in comps.values())
    return {
        "name": "bfcl_v4_single_turn_v1",
        "source": {
            "package": "bfcl-eval",
            "version": BFCL_VERSION,
            "license": "Apache-2.0",
            "data_sha256": dict(sorted(shas.items())),
        },
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "grouping": "components connected by identical normalized user questions"
        " or shared live source group",
        "stats": {
            "entries": len(entries),
            "components": len(comps),
            "multi_entry_components": sum(v for k, v in sizes.items() if k > 1),
            "largest_component": max(len(ms) for ms in comps.values()),
            "dev_components": len(dev_comp),
        },
        "partitions": {
            p: {
                "n": len(ids),
                "category_counts": {c: len(v) for c, v in by_category(ids, entries).items()},
                "ids": by_category(ids, entries),
            }
            for p, ids in parts.items()
        },
    }


def render(split_doc: dict[str, Any]) -> str:
    return json.dumps(split_doc, indent=1, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the BFCL single-turn dev/held-out split")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(build())
    if args.check:
        ok = OUT.exists() and OUT.read_text() == text
        print(f"{'ok' if ok else 'OUT OF DATE'}: {OUT}")
        return 0 if ok else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    doc = json.loads(text)
    print(json.dumps(doc["stats"]))
    for p, v in doc["partitions"].items():
        print(p, v["n"], v["category_counts"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
