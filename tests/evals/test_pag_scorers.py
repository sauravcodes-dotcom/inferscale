"""PAG scorer, prompt and baseline tests with hand-computed values."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from evals import baselines, semantic
from evals.datasets.pag import normalize
from evals.prompts import pag_v1
from evals.scorers import (
    classify_output,
    grounded,
    lcs_len,
    load_lexicon,
    rouge_l,
    score_pag_item,
    set_scores,
    summarize_pag,
    tokenize,
)
from tests.evals.test_pag import raw_record

SCHEMA = pag_v1.SCHEMA
LEX = load_lexicon()


def test_tokenize() -> None:
    assert tokenize("Fix the DNS-resolver, NOW!  v2") == [
        "fix",
        "the",
        "dns",
        "resolver",
        "now",
        "v2",
    ]


def test_lcs() -> None:
    assert lcs_len(list("ABCBDAB"), list("BDCABA")) == 4  # textbook example
    assert lcs_len([], ["a"]) == 0


@pytest.mark.parametrize(
    ("cand", "ref", "expected"),
    [
        # tokens 6/6, LCS = the cat on the mat = 5 -> P = R = 5/6
        ("the cat sat on the mat", "the cat is on the mat", 5 / 6),
        # LCS = restart the = 2; P = 2/3, R = 2/4 -> F = 2*(1/3)/(7/6) = 4/7
        ("restart the db", "Restart the database server", 4 / 7),
        ("Fix, DNS!", "fix dns", 1.0),
        ("a b c", "x y z", 0.0),
        ("", "anything", 0.0),
        # order matters for LCS: "b a" vs "a b" -> LCS 1, P = R = 1/2
        ("b a", "a b", 0.5),
    ],
)
def test_rouge_l(cand: str, ref: str, expected: float) -> None:
    assert rouge_l(cand, ref) == pytest.approx(expected)


def test_set_scores_symmetric_case() -> None:
    # sim = [[6/7, 0], [0, 1]] -> coverage = precision = (6/7 + 1) / 2 = 13/14
    s = set_scores(["a b c", "x y"], ["a b c d", "x y"])
    assert s["coverage"] == pytest.approx(13 / 14)
    assert s["precision"] == pytest.approx(13 / 14)
    assert s["f1"] == pytest.approx(13 / 14)


def test_set_scores_asymmetric_case() -> None:
    # one prediction matching one of two references: coverage 1/2, precision 1, F1 2/3
    s = set_scores(["a b"], ["a b", "c d"])
    assert (s["coverage"], s["precision"]) == (pytest.approx(0.5), pytest.approx(1.0))
    assert s["f1"] == pytest.approx(2 / 3)
    assert set_scores([], ["a"]) == {"coverage": 0.0, "precision": 0.0, "f1": 0.0}


def valid_output(**over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "severity": "Sev-1",
        "root_cause_type": "Config Errors",
        "root_cause_team": "Logging",
        "root_cause_summary": "A bad configuration push.",
        "corrective_actions": [
            "Roll back the faulty configuration push now.",
            "Restart the logging pipeline workers.",
            "Validate the configuration on all nodes.",
        ],
        "preventive_actions": [
            "Add schema validation for config changes in CI.",
            "Stage configuration rollouts by region.",
            "Alert on configuration drift between nodes.",
        ],
    }
    out.update(over)
    return out


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"http_status": 500}, "http_error"),
        ({"http_status": 502, "error": "engine_error"}, "http_error"),
        ({"http_status": 504}, "timeout"),
        ({"error": "timeout", "http_status": None}, "timeout"),
        ({"finish_reason": "length"}, "length"),
    ],
)
def test_classify_transport_reasons(kw: dict[str, Any], reason: str) -> None:
    assert classify_output(json.dumps(valid_output()), SCHEMA, **kw) == (None, reason)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("not json", "parse"),
        ("[1, 2]", "parse"),
        ("```json\n{}\n```", "parse"),
        (json.dumps(valid_output(severity="Sev-3")), "schema"),
        (json.dumps(valid_output(corrective_actions=["too short", "x" * 30, "y" * 30])), "schema"),
        (json.dumps(valid_output(preventive_actions=["z" * 30, "w" * 30])), "schema"),  # < 3 items
        (json.dumps(valid_output(extra=1)), "schema"),
        (json.dumps(valid_output(root_cause_summary="s" * 301)), "schema"),
    ],
)
def test_classify_content_reasons(text: str, reason: str) -> None:
    assert classify_output(text, SCHEMA) == (None, reason)


def test_classify_valid_and_600_char_cap() -> None:
    parsed, reason = classify_output(
        json.dumps(valid_output(corrective_actions=["a" * 600] * 3)), SCHEMA
    )
    assert reason is None and parsed is not None
    assert (
        classify_output(json.dumps(valid_output(corrective_actions=["a" * 601] * 3)), SCHEMA)[1]
        == "schema"
    )


REF = {
    "severity": "Sev-2",
    "root_cause_type": "Config Errors",
    "root_cause_team": "Logging",
    "corrective_actions": [
        "Roll back the faulty configuration push now.",
        "Restart the logging pipeline workers.",
        "Validate the configuration on all nodes.",
    ],
    "preventive_actions": [
        "Add schema validation for config changes in CI.",
        "Stage configuration rollouts by region.",
        "Alert on configuration drift between nodes.",
    ],
}


def test_score_item_and_summary_hand_computed() -> None:
    good = score_pag_item("a", REF, valid_output(), None, LEX)  # actions copied from the reference
    assert good["valid"] and good["type_correct"] and good["team_correct"]
    assert not good["severity_correct"]  # Sev-1 vs Sev-2
    assert good["corrective_f1"] == pytest.approx(1.0) and good[
        "preventive_coverage"
    ] == pytest.approx(1.0)
    assert good["grounded"] and good["format_ok"]
    bad = score_pag_item("b", REF, None, "length", LEX)
    assert not bad["valid"] and bad["invalid_reason"] == "length"
    assert bad["corrective_f1"] == 0.0 and not bad["grounded"] and not bad["format_ok"]
    s = summarize_pag([good, bad], n_boot=100)
    assert s["valid"]["value"] == 0.5
    assert s["corrective_f1"]["value"] == pytest.approx(0.5)  # invalid counts as 0
    assert s["type_correct"]["value"] == 0.5
    assert s["type_macro_f1"]["value"] == pytest.approx(2 / 3)  # one class: P=1, R=1/2
    assert s["severity_correct"]["value"] == 0.0
    assert s["severity_majority_baseline"] == 1.0  # both gold Sev-2
    assert s["invalid_breakdown"] == {
        "http_error": 0,
        "timeout": 0,
        "length": 1,
        "parse": 0,
        "schema": 0,
    }


def test_format_fidelity_flags_duplicates() -> None:
    dup = valid_output(
        preventive_actions=["Roll back the faulty configuration push now.", "b" * 25, "c" * 25]
    )
    row = score_pag_item("a", REF, dup, None, LEX)
    assert row["valid"] and not row["format_ok"]


def test_grounding_keyword_vs_substring() -> None:
    acts = ["Re-provision the failed nodes with extra capacity."]
    assert grounded(acts, "Provisioning Failures", LEX)
    row = score_pag_item(
        "a",
        {**REF, "root_cause_type": "Provisioning Failures"},
        valid_output(
            corrective_actions=acts * 3, preventive_actions=["d" * 25, "e" * 25, "f" * 25]
        ),
        None,
        LEX,
    )
    assert (
        row["grounded"] and not row["grounded_substring"]
    )  # 'provisioning failures' never appears
    assert not grounded(["Restart the service."], "Hardware Issues", LEX)


def test_lexicon_covers_schema_types() -> None:
    assert set(LEX) == set(pag_v1.TYPES)
    assert all(LEX[t] for t in LEX)


# --- prompt -------------------------------------------------------------------------


def test_prompt_inputs_and_format() -> None:
    rec = normalize(raw_record(5), 5)
    msgs = pag_v1.build_messages(rec.model_input())
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert rec.summary in user and "Root Cause" not in user and "Severity" not in user
    assert "CAPM" not in user and rec.capm_ticket_link not in user
    for t in pag_v1.TYPES + pag_v1.TEAMS:
        assert json.dumps(t) in msgs[0]["content"]
    with pytest.raises(ValueError, match="not allowed"):
        pag_v1.build_messages({**rec.model_input(), "Severity": "Sev-1"})
    rf = pag_v1.response_format("json_schema")
    assert rf is not None and rf["json_schema"]["schema"]["additionalProperties"] is False
    assert "$schema" not in rf["json_schema"]["schema"]
    assert pag_v1.response_format("none") is None
    assert pag_v1.fingerprint()["prompt_version"] == "pag_v1.p1"
    assert len(pag_v1.prompt_sha256()) == 64


def test_schema_fits_gateway_limit() -> None:
    rf = pag_v1.response_format("json_schema")
    assert rf is not None
    assert len(json.dumps(rf["json_schema"]["schema"])) < 16_384


# --- baselines ----------------------------------------------------------------------


def test_extract_prefers_longest_match() -> None:
    assert baselines.extract(
        ["Network Connectivity Issues", "Config Errors"], "x network connectivity issues y"
    ) == ("Network Connectivity Issues")
    assert baselines.extract(["Logging"], "nothing here") is None


def test_retrieval_uses_text_not_gold_labels() -> None:
    pool = [normalize(raw_record(i), i) for i in range(100, 106)]
    target = normalize(raw_record(200), 200)
    # Corrupt the gold label: retrieval must still predict the type stated in the Summary.
    stated = baselines.extract(pag_v1.TYPES, target.summary)
    wrong = "Hardware Issues" if stated != "Hardware Issues" else "Config Errors"
    target = replace(target, root_cause_type=wrong)
    out = baselines.retrieval_baseline(pool)(target)
    assert out["root_cause_type"] == stated
    assert out["corrective_actions"] in [list(p.corrective_actions) for p in pool]


def test_generic_baseline_is_schema_valid() -> None:
    pool = [normalize(raw_record(i), i) for i in range(100, 103)]
    out = baselines.generic_baseline(pool)(pool[0])
    assert classify_output(json.dumps(out), SCHEMA)[1] is None
    assert len(out["corrective_actions"]) == 4 and len(out["preventive_actions"]) == 4


# --- semantic (optional) ------------------------------------------------------------


def test_greedy_cosine_hand_computed() -> None:
    # pred = [x-axis], refs = [x-axis, y-axis] -> sims [[1, 0]]: coverage 0.5, precision 1
    s = semantic.greedy_cosine([[1.0, 0.0]], [[2.0, 0.0], [0.0, 3.0]])
    assert s["coverage"] == pytest.approx(0.5) and s["precision"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(2 / 3)
    assert semantic.cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(2**-0.5)


def test_semantic_skips_without_library() -> None:
    if semantic.available():
        pytest.skip("sentence-transformers installed")
    res = semantic.score_rows([])
    assert "skipped" in res and res["label"] == "semantic similarity (model-based)"
