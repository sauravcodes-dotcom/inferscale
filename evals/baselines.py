"""Model-free baselines for the PAG task, scored with the same scorers as model runs.

- generic: one fixed, hand-written generic CAPM (4 corrective + 4 preventive actions) for every
  incident; severity, type and team are the pool-partition majority classes.
- retrieval: root-cause type and team are extracted from the input text (the type name appears
  verbatim in every Summary, the team name in every Description); the output copies the CAPM,
  severity and root-cause description of the pool record with the same type and team whose
  Summary is nearest by token Jaccard (fallback: same type, then the whole pool).

Neither baseline reads gold labels of the scored record. Writes results/baselines/pag_v1.json
(metrics only, no dataset text).

Usage: uv run python -m evals.baselines [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from evals.dataset import ROOT, sha256_file
from evals.datasets.pag import SOURCE, PagRecord, load_records
from evals.prompts.pag_v1 import SCHEMA, TEAMS, TYPES
from evals.scorers import classify_output, load_lexicon, score_pag_item, summarize_pag, tokenize

SPLIT_FILE = ROOT / "splits" / "pag_v1.json"
OUT = ROOT / "results" / "baselines" / "pag_v1.json"

GENERIC_CORRECTIVE = [
    "Roll back the most recent change to the affected "
    "service and confirm that error rates return to normal levels.",
    "Restart or fail over the unhealthy components and "
    "verify recovery with health checks and synthetic probes.",
    "Communicate the incident status and expected resolution "
    "time to affected customers and internal stakeholders.",
    "Monitor the affected service closely for recurrence and "
    "keep the incident open until metrics are stable.",
]
GENERIC_PREVENTIVE = [
    "Conduct a blameless post-incident review to "
    "identify contributing factors and document lessons learned.",
    "Add automated alerting and dashboards for the "
    "signals that would have detected this incident earlier.",
    "Strengthen change management with staged rollouts, "
    "automated testing and a documented rollback plan.",
    "Run regular resilience drills and update on-call runbooks "
    "with the mitigation steps used in this incident.",
]

Output = dict[str, Any]


def extract(options: Sequence[str], text: str) -> str | None:
    t = text.lower()
    hits = [o for o in options if o.lower() in t]
    return max(hits, key=len) if hits else None


def jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def generic_baseline(pool: list[PagRecord]) -> Callable[[PagRecord], Output]:
    sev = Counter(r.severity for r in pool).most_common(1)[0][0]
    typ = Counter(r.root_cause_type for r in pool).most_common(1)[0][0]
    team = Counter(r.root_cause_team for r in pool).most_common(1)[0][0]

    def predict(_rec: PagRecord) -> Output:
        return {
            "severity": sev,
            "root_cause_type": typ,
            "root_cause_team": team,
            "root_cause_summary": "The incident was caused by a fault in the affected service.",
            "corrective_actions": list(GENERIC_CORRECTIVE),
            "preventive_actions": list(GENERIC_PREVENTIVE),
        }

    return predict


def retrieval_baseline(pool: list[PagRecord]) -> Callable[[PagRecord], Output]:
    def predict(rec: PagRecord) -> Output:
        typ = extract(TYPES, rec.summary)
        team = extract(TEAMS, rec.description)
        candidates = [p for p in pool if p.root_cause_type == typ and p.root_cause_team == team]
        if not candidates:
            candidates = [p for p in pool if p.root_cause_type == typ] or pool
        best = max(candidates, key=lambda p: (jaccard(rec.summary, p.summary), -p.source_index))
        return {
            "severity": best.severity,
            "root_cause_type": typ or best.root_cause_type,
            "root_cause_team": team or best.root_cause_team,
            "root_cause_summary": best.root_cause_description[:300],
            "corrective_actions": list(best.corrective_actions)[:6],
            "preventive_actions": list(best.preventive_actions)[:6],
        }

    return predict


def score_baseline(
    predict: Callable[[PagRecord], Output], records: list[PagRecord], n_boot: int
) -> dict[str, Any]:
    lexicon = load_lexicon()
    rows = []
    for rec in records:
        out = predict(rec)
        parsed, reason = classify_output(json.dumps(out), SCHEMA)
        rows.append(score_pag_item(rec.incident_id, rec.reference(), parsed, reason, lexicon))
    return summarize_pag(rows, n_boot=n_boot)


def run(n_boot: int = 1000) -> dict[str, Any]:
    split = json.loads(SPLIT_FILE.read_text())
    records = load_records()
    part = {p: [records[i] for i in v["incident_ids"]] for p, v in split["partitions"].items()}
    baselines = {
        "generic": generic_baseline(part["pool"]),
        "retrieval": retrieval_baseline(part["pool"]),
    }
    results: dict[str, Any] = {}
    for name, predict in baselines.items():
        results[name] = {}
        for p in ("dev", "test"):
            results[name][p] = score_baseline(predict, part[p], n_boot)
            ids = "\n".join(split["partitions"][p]["incident_ids"]).encode()
            results[name][p]["ids_sha256"] = hashlib.sha256(ids).hexdigest()
    return {
        "task": "pag_v1",
        "source_sha256": SOURCE["sha256"],
        "split_sha256": sha256_file(SPLIT_FILE),
        "schema": "pag_output_v1.json",
        "lexicon": "pag_root_cause_v1",
        "retrieval_pool": "pool",
        "baselines": results,
    }


METRICS_TABLE = (
    ("valid", "valid"),
    ("type_correct", "type acc"),
    ("team_correct", "team acc"),
    ("severity_correct", "sev acc"),
    ("corrective_coverage", "corr cov"),
    ("corrective_precision", "corr prec"),
    ("corrective_f1", "corr F1"),
    ("preventive_coverage", "prev cov"),
    ("preventive_precision", "prev prec"),
    ("preventive_f1", "prev F1"),
    ("grounded", "grounded"),
    ("format_ok", "format"),
)


def table(res: dict[str, Any]) -> str:
    head = (
        "| baseline | split | " + " | ".join(lbl for _, lbl in METRICS_TABLE) + " | sev majority |"
    )
    lines = [head, "|" + "---|" * (len(METRICS_TABLE) + 3)]
    for name, by_split in res["baselines"].items():
        for split_name, s in by_split.items():
            cells = [f"{s[k]['value']:.3f}" for k, _ in METRICS_TABLE]
            lines.append(
                f"| {name} | {split_name} | "
                + " | ".join(cells)
                + f" | {s['severity_majority_baseline']:.3f} |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the PAG baselines on dev and test")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    res = run(args.n_boot)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(table(res))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
