"""Merge evals/datasets/triage_v1/review.csv back into dev.jsonl / test.jsonl.

Every row needs reviewer_ok:
  yes/y/true/1/ok -> the draft label is accepted;
  no/n/false/0    -> corrected_* columns override the label (at least one correction or a note
                     is required).
needs_page is re-derived from severity unless corrected_needs_page is given, which must agree
with the rule (SEV1/SEV2 <=> true). The script fails without writing anything if any row is
unreviewed or invalid. On success it sets review_status "reviewed" and recomputes the manifest.

Usage: uv run python scripts/apply_review.py [--dataset-dir DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import SPLITS, TRIAGE_V1, load_jsonl, load_schema, write_jsonl, write_manifest

YES = {"yes", "y", "true", "1", "ok"}
NO = {"no", "n", "false", "0"}
CORRECTABLE = ("severity", "component", "category", "needs_page")


class ReviewError(Exception):
    pass


def _bool(v: str, what: str) -> bool:
    s = v.strip().lower()
    if s in YES:
        return True
    if s in NO:
        return False
    raise ReviewError(f"{what}: expected yes/no, got {v!r}")


def apply(dataset_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    schema = load_schema(dataset_dir)
    validator = jsonschema.Draft202012Validator(schema)
    items = {s: load_jsonl(dataset_dir / f"{s}.jsonl") for s in SPLITS}
    by_id = {it["id"]: (s, it) for s, its in items.items() for it in its}

    with (dataset_dir / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    unreviewed: list[str] = []
    seen: set[str] = set()
    corrected = 0
    for row in rows:
        rid = row["id"]
        try:
            if rid not in by_id:
                raise ReviewError("id not in dataset")
            if rid in seen:
                raise ReviewError("duplicate row")
            seen.add(rid)
            split, item = by_id[rid]
            if row["split"] != split:
                raise ReviewError(f"split {row['split']!r} != dataset split {split!r}")
            if not row["reviewer_ok"].strip():
                unreviewed.append(rid)
                continue
            ok = _bool(row["reviewer_ok"], "reviewer_ok")
            label = dict(item["label"])
            changed = []
            if not ok:
                for fld in CORRECTABLE:
                    val = row[f"corrected_{fld}"].strip()
                    if not val:
                        continue
                    new: Any = _bool(val, f"corrected_{fld}") if fld == "needs_page" else val
                    if new != label[fld]:
                        label[fld] = new
                        changed.append(fld)
                if not changed and not row["notes"].strip():
                    raise ReviewError("reviewer_ok=no needs a correction or a note")
            derived = label["severity"] in ("SEV1", "SEV2")
            if row["corrected_needs_page"].strip():
                if label["needs_page"] != derived:
                    raise ReviewError(
                        f"needs_page={label['needs_page']} contradicts {label['severity']}"
                    )
            elif label["needs_page"] != derived:
                label["needs_page"] = derived
                changed.append("needs_page")
            errs = sorted(e.message for e in validator.iter_errors(label))
            if errs:
                raise ReviewError("label invalid: " + "; ".join(errs))
            corrected += bool(changed)
            item["label"] = label
            item["review_status"] = "reviewed"
            item["review"] = {
                "ok": ok,
                "changed": sorted(set(changed)),
                "notes": row["notes"].strip() or None,
            }
        except ReviewError as e:
            errors.append(f"{rid}: {e}")

    missing = sorted(set(by_id) - seen)
    if missing:
        errors.append(f"{len(missing)} dataset items have no review row: {', '.join(missing[:10])}")
    if unreviewed:
        errors.append(
            f"{len(unreviewed)} rows unreviewed (reviewer_ok empty): {', '.join(unreviewed[:10])}"
            + (" ..." if len(unreviewed) > 10 else "")
        )
    if errors:
        raise ReviewError("\n".join(errors))

    summary = {"reviewed": len(seen), "corrected": corrected}
    if dry_run:
        return summary
    for s, its in items.items():
        write_jsonl(dataset_dir / f"{s}.jsonl", its)
    old = json.loads((dataset_dir / "manifest.json").read_text())
    old.pop("files", None)
    old.pop("counts", None)
    old["review"] = {**summary, "applied_at": datetime.now(UTC).isoformat(timespec="seconds")}
    write_manifest(dataset_dir, old)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge review.csv into the triage_v1 dataset")
    ap.add_argument("--dataset-dir", type=Path, default=TRIAGE_V1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        summary = apply(args.dataset_dir, args.dry_run)
    except ReviewError as e:
        print(f"review not applied:\n{e}", file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
