from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.dataset import TRIAGE_V1, load_jsonl, sha256_file, write_jsonl, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import apply_review

COLS = [
    "id",
    "split",
    "alert_title",
    "alert_body",
    "log_excerpt",
    "severity",
    "component",
    "category",
    "needs_page",
    "summary",
    "reviewer_ok",
    "corrected_severity",
    "corrected_component",
    "corrected_category",
    "corrected_needs_page",
    "notes",
]
LABEL = {
    "severity": "SEV3",
    "component": "database",
    "category": "data",
    "needs_page": False,
    "summary": "s",
}


def it(i: int) -> dict[str, Any]:
    return {
        "id": f"tri-{i:04d}",
        "input": {"alert_title": "t", "alert_body": "b", "log_excerpt": "l"},
        "label": dict(LABEL),
        "review_status": "draft",
        "meta": {"ambiguous": False},
    }


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    shutil.copy(TRIAGE_V1 / "schema.json", d / "schema.json")
    write_jsonl(d / "dev.jsonl", [it(1)])
    write_jsonl(d / "test.jsonl", [it(2), it(3)])
    write_manifest(d, {"split_seed": 1})
    return d


def write_review(d: Path, rows: dict[str, dict[str, str]]) -> None:
    with (d / "review.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for rid, split in (("tri-0001", "dev"), ("tri-0002", "test"), ("tri-0003", "test")):
            base = dict.fromkeys(COLS, "")
            base.update(id=rid, split=split)
            base.update(rows.get(rid, {}))
            w.writerow(base)


def test_unreviewed_rows_fail_and_write_nothing(ds: Path) -> None:
    before = sha256_file(ds / "test.jsonl")
    write_review(ds, {"tri-0001": {"reviewer_ok": "yes"}, "tri-0002": {"reviewer_ok": "yes"}})
    with pytest.raises(apply_review.ReviewError, match="1 rows unreviewed"):
        apply_review.apply(ds)
    assert sha256_file(ds / "test.jsonl") == before


def test_apply_accepts_and_corrects(ds: Path) -> None:
    write_review(
        ds,
        {
            "tri-0001": {"reviewer_ok": "yes"},
            "tri-0002": {
                "reviewer_ok": "no",
                "corrected_severity": "SEV2",
                "notes": "customer-facing",
            },
            "tri-0003": {
                "reviewer_ok": "N",
                "corrected_category": "resource_exhaustion",
                "corrected_component": "storage",
            },
        },
    )
    assert apply_review.apply(ds) == {"reviewed": 3, "corrected": 2}
    test = {x["id"]: x for x in load_jsonl(ds / "test.jsonl")}
    assert test["tri-0002"]["label"]["severity"] == "SEV2"
    assert test["tri-0002"]["label"]["needs_page"] is True  # re-derived from severity
    assert test["tri-0002"]["review"] == {
        "ok": False,
        "changed": ["needs_page", "severity"],
        "notes": "customer-facing",
    }
    assert test["tri-0003"]["label"]["category"] == "resource_exhaustion"
    assert all(x["review_status"] == "reviewed" for x in test.values())
    m = json.loads((ds / "manifest.json").read_text())
    assert m["files"]["test.jsonl"]["sha256"] == sha256_file(ds / "test.jsonl")
    assert m["counts"]["test"]["review_status"] == {"reviewed": 2}
    assert m["review"]["corrected"] == 2 and m["split_seed"] == 1


@pytest.mark.parametrize(
    ("row", "msg"),
    [
        ({"reviewer_ok": "no"}, "needs a correction or a note"),
        ({"reviewer_ok": "maybe"}, "expected yes/no"),
        ({"reviewer_ok": "no", "corrected_severity": "SEV9"}, "label invalid"),
        (
            {"reviewer_ok": "no", "corrected_severity": "SEV1", "corrected_needs_page": "false"},
            "contradicts",
        ),
        ({"reviewer_ok": "yes", "split": "dev"}, "split"),
    ],
)
def test_invalid_rows(ds: Path, row: dict[str, str], msg: str) -> None:
    write_review(
        ds,
        {"tri-0001": {"reviewer_ok": "yes"}, "tri-0002": row, "tri-0003": {"reviewer_ok": "yes"}},
    )
    with pytest.raises(apply_review.ReviewError, match=msg):
        apply_review.apply(ds)


def test_shipped_review_csv_matches_dataset() -> None:
    with (TRIAGE_V1 / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == COLS
    ids = {x["id"] for s in ("dev", "test") for x in load_jsonl(TRIAGE_V1 / f"{s}.jsonl")}
    assert {r["id"] for r in rows} == ids and len(rows) == 300
    assert all(r["reviewer_ok"] == "" for r in rows)
