"""PAG loader, normalization, exclusion and split tests.

Synthetic fixtures cover the mechanics; tests marked `needs_pag` check the real pinned file
and are skipped when data/pag/ has not been fetched (the data is not committed).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evals.datasets import pag
from evals.datasets.pag import PagDataError, load_records, normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import make_pag_split

HAVE_PAG = pag.PAG_FILE.is_file()
needs_pag = pytest.mark.skipif(not HAVE_PAG, reason="data/pag not fetched (scripts/fetch_pag.py)")
TYPES = ["Config Errors", "Hardware Issues", "Monitoring Gaps"]


def raw_record(i: int, **over: Any) -> dict[str, Any]:
    t = TYPES[i % len(TYPES)]
    r: dict[str, Any] = {
        "Incident_ID": f"INC-{i:04d}",
        "Summary": f"Svc {i % 7} experienced {t.lower()}.",
        "Description": f"Something about {t.lower()}.",
        "Severity": "Sev-1" if i % 2 else "Sev-2",
        "Detection_Method": "Customer Report",
        "Impact_Type": "Latency",
        "Impacted_Customer_Count": 100 + i,
        "Region": "EU-West",
        "Root_Cause_Team": "Logging",
        "Root_Cause_Type": t,
        "Root_Cause_Description": "desc",
        "Impact_Start_Time": "2025-01-01T00:00:00Z",
        "Detected_Time": "2025-01-01T00:05:00Z",
        "Mitigation_Time": "2025-01-01T01:00:00Z",
        "Repeat_Outage": False,
        "CAPM_Ticket_Link": "ABCD1234",
        "CAPM": {
            "Corrective_Actions": [f"fix {i} a", f"fix {i} b"],
            "Preventive_Actions": [f"prevent {i}"],
        },
    }
    r.update(over)
    return r


def test_normalize_lowercase_capm_keys() -> None:
    r = raw_record(1, CAPM={"corrective_actions": ["a1"], "preventive_actions": ["p1", " p2 "]})
    rec = normalize(r, 1)
    assert rec.corrective_actions == ("a1",)
    assert rec.preventive_actions == ("p1", "p2")


def test_normalize_flattens_dict_actions() -> None:
    capm = {
        "Corrective_Actions": [
            {"Action": "do X", "Responsible_Team": "Network Operations"},
            "do Y",
        ],
        "Preventive_Actions": [{"Action": "prevent Z", "Responsible_Team": "SRE"}],
    }
    rec = normalize(raw_record(2, CAPM=capm), 2)
    assert rec.corrective_actions == ("do X", "do Y")
    assert rec.preventive_actions == ("prevent Z",)


@pytest.mark.parametrize(
    "capm",
    [
        {"Corrective_Actions": ["a"]},  # no preventive key
        {"Corrective_Actions": [], "Preventive_Actions": ["p"]},  # empty list
        {"Corrective_Actions": [{"Team": "x"}], "Preventive_Actions": ["p"]},  # dict without Action
        {"Corrective_Actions": ["  "], "Preventive_Actions": ["p"]},  # blank action
    ],
)
def test_normalize_rejects_bad_capm(capm: dict[str, Any]) -> None:
    with pytest.raises(PagDataError):
        normalize(raw_record(3, CAPM=capm), 3)


def test_model_input_never_leaks_labels() -> None:
    inp = normalize(raw_record(4), 4).model_input()
    assert set(inp) == set(pag.INPUT_FIELDS)
    for leaked in (
        "Severity",
        "Root_Cause_Type",
        "Root_Cause_Team",
        "Root_Cause_Description",
        "CAPM",
        "CAPM_Ticket_Link",
        "Incident_ID",
    ):
        assert leaked not in inp


def test_exclusion_and_unique_ids() -> None:
    raw = [raw_record(i) for i in range(110)]
    for i in range(10):  # indices 0-9 duplicate the IDs of 100-109, like the real file
        raw[i]["Incident_ID"] = raw[100 + i]["Incident_ID"]
    recs = load_records(raw=raw)
    assert len(recs) == 10
    assert sorted(r.source_index for r in recs.values()) == list(range(100, 110))
    raw[105]["Incident_ID"] = raw[104]["Incident_ID"]
    with pytest.raises(PagDataError, match="duplicate Incident_ID"):
        load_records(raw=raw)


def synthetic_universe(n: int = 270) -> dict[str, pag.PagRecord]:
    raw = [raw_record(i) for i in range(100 + n)]
    return load_records(raw=raw)


TARGETS = {"dev": 30, "pool": 80, "test": 160}


def test_split_is_deterministic_and_complete() -> None:
    recs = synthetic_universe()
    a = make_pag_split.split_records(recs, TARGETS, seed=7)
    b = make_pag_split.split_records(recs, TARGETS, seed=7)
    assert {p: [r.incident_id for r in v] for p, v in a.items()} == {
        p: [r.incident_id for r in v] for p, v in b.items()
    }
    ids = [r.incident_id for v in a.values() for r in v]
    assert len(ids) == len(set(ids)) == len(recs)
    c = make_pag_split.split_records(recs, TARGETS, seed=8)
    assert [r.incident_id for r in c["dev"]] != [r.incident_id for r in a["dev"]]


def test_split_keeps_summary_groups_together_and_stratifies() -> None:
    recs = synthetic_universe()
    parts = make_pag_split.split_records(recs, TARGETS, seed=7)
    where = {r.summary.lower(): p for p, v in parts.items() for r in v}
    for p, v in parts.items():
        for r in v:
            assert where[r.summary.lower()] == p
    biggest_group = max(Counter(r.summary.lower() for r in recs.values()).values())
    for p, v in parts.items():
        assert abs(len(v) - TARGETS[p]) <= biggest_group * len(TYPES)
        counts = Counter(r.root_cause_type for r in v)
        for t in TYPES:  # each type gets roughly its proportional share
            expected = TARGETS[p] / len(TYPES)
            assert abs(counts[t] - expected) <= biggest_group + 1


def test_split_rejects_group_spanning_types() -> None:
    raw = [raw_record(i) for i in range(103)]
    raw[101]["Summary"] = raw[100]["Summary"]  # same summary, different type
    with pytest.raises(ValueError, match="spans root-cause types"):
        make_pag_split.split_records(load_records(raw=raw), {"dev": 1, "pool": 1, "test": 1})


def test_action_overlap_counts() -> None:
    recs = list(synthetic_universe(6).values())
    a, b = recs[:3], recs[3:]
    b[0] = replace(
        b[0], corrective_actions=("FIX " + a[0].corrective_actions[0][4:] + "  ", "other")
    )
    ov = make_pag_split.action_overlap(a, b)
    assert ov["corrective_actions"]["exact_match_in_other"] == 1  # case/whitespace-normalized
    assert ov["corrective_actions"]["records_with_any_match"] == 1
    assert ov["preventive_actions"]["exact_match_in_other"] == 0


# --- real pinned file --------------------------------------------------------


@needs_pag
def test_real_file_facts() -> None:
    raw = pag.load_raw()
    assert len(raw) == pag.N_RAW == 950
    assert sum(1 for r in raw if "Corrective_Actions" in r["CAPM"]) == 949
    assert "corrective_actions" in raw[526]["CAPM"]
    dict_recs = sorted(
        {i for i, r in enumerate(raw) for v in r["CAPM"].values() for a in v if isinstance(a, dict)}
    )
    assert dict_recs == [68, 677, 867]
    for i in range(100):
        assert raw[i]["Incident_ID"] == raw[799 + i]["Incident_ID"]
        assert json.dumps(raw[i]["CAPM"], sort_keys=True) == json.dumps(
            raw[799 + i]["CAPM"], sort_keys=True
        )
    recs = load_records(raw=raw)
    assert len(recs) == 850
    by_index = {r.source_index: r for r in recs.values()}
    assert all(
        isinstance(a, str)
        for i in (677, 867)
        for a in by_index[i].corrective_actions + by_index[i].preventive_actions
    )
    assert by_index[526].corrective_actions


@needs_pag
def test_committed_split_matches_build() -> None:
    split = json.loads(make_pag_split.OUT.read_text())
    assert make_pag_split.render(make_pag_split.build()) == make_pag_split.OUT.read_text()
    parts = split["partitions"]
    ids = [i for p in parts.values() for i in p["incident_ids"]]
    assert len(ids) == len(set(ids)) == 850
    assert split["source"]["sha256"] == pag.SOURCE["sha256"]
    assert len(split["excluded"]) == 100 and split["excluded"][0]["source_index"] == 0
    assert all(min(p["source_indices"]) >= 100 for p in parts.values())
    recs = load_records()
    groups: dict[str, set[str]] = {}
    for name, p in parts.items():
        for i in p["incident_ids"]:
            groups.setdefault(recs[i].summary.strip().lower(), set()).add(name)
    assert all(len(v) == 1 for v in groups.values())
