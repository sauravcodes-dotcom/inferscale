"""Scorer tests with hand-computed expected values."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evals.dataset import load_schema
from evals.scorers import (
    bootstrap_ci,
    macro_f1,
    mcnemar_exact,
    paired_mcnemar,
    score_item,
    summarize,
)

SCHEMA = load_schema()
GOLD: dict[str, Any] = {
    "severity": "SEV2",
    "component": "database",
    "category": "network",
    "needs_page": True,
    "summary": "x",
}


def out(**over: Any) -> str:
    return json.dumps({**GOLD, "summary": "model summary", **over})


def test_exact_match() -> None:
    s = score_item("a", GOLD, out(), SCHEMA)
    assert s.json_valid and s.schema_valid and s.all_fields_correct
    assert s.pred_category == "network"


def test_summary_not_exact_matched() -> None:
    assert score_item("a", GOLD, out(summary="different words"), SCHEMA).all_fields_correct


def test_one_field_wrong() -> None:
    s = score_item("a", GOLD, out(severity="SEV3"), SCHEMA)
    assert s.schema_valid and not s.severity_correct and s.category_correct
    assert not s.all_fields_correct


@pytest.mark.parametrize(
    "text",
    ["```json\n" + out() + "\n```", "Sure! " + out(), "[1, 2]", "", None, '{"severity": "SEV2"'],
)
def test_not_json_object(text: str | None) -> None:
    s = score_item("a", GOLD, text, SCHEMA)
    assert not s.json_valid and not s.schema_valid
    assert not any(
        [s.severity_correct, s.component_correct, s.category_correct, s.needs_page_correct]
    )
    assert s.pred_category is None


def test_whitespace_is_trimmed() -> None:
    assert score_item("a", GOLD, "\n  " + out() + "  \n", SCHEMA).all_fields_correct


def test_schema_invalid_keeps_field_credit() -> None:
    s = score_item("a", GOLD, out(confidence=0.9), SCHEMA)  # extra property
    assert s.json_valid and not s.schema_valid
    assert s.severity_correct and s.category_correct
    assert not s.all_fields_correct
    assert any("confidence" in e for e in s.schema_errors)


def test_needs_page_must_be_boolean() -> None:
    s = score_item("a", GOLD, out(needs_page=1), SCHEMA)
    assert not s.needs_page_correct and not s.schema_valid


def test_out_of_vocab_category() -> None:
    s = score_item("a", GOLD, out(category="networking"), SCHEMA)
    assert s.pred_category is None and not s.category_correct


def test_macro_f1_hand_computed() -> None:
    # a: P=1, R=1/2 -> F1=2/3 ; b: P=2/3, R=1 -> F1=4/5 ; c: P=0, R=0 -> 0
    golds = ["a", "a", "b", "b", "c"]
    preds = ["a", "b", "b", "b", None]
    assert macro_f1(golds, preds) == pytest.approx((2 / 3 + 4 / 5 + 0) / 3)
    assert macro_f1(golds, preds) == pytest.approx(22 / 45)


def test_macro_f1_perfect_and_label_set() -> None:
    assert macro_f1(["x", "y"], ["x", "y"]) == 1.0
    # A predicted class absent from gold doesn't add a class unless labels are given.
    assert macro_f1(["x", "x"], ["x", "z"]) == pytest.approx(2 / 3)
    assert macro_f1(["x", "x"], ["x", "z"], labels=["x", "z"]) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("b", "c", "p"),
    [
        (0, 0, 1.0),
        (5, 1, 2 * (1 + 6) / 64),  # n=6, k=1
        (1, 5, 2 * (1 + 6) / 64),
        (10, 0, 2 / 1024),
        (4, 2, 2 * (1 + 6 + 15) / 64),  # n=6, k=2 -> 0.6875
        (3, 3, 1.0),  # capped at 1
    ],
)
def test_mcnemar_exact(b: int, c: int, p: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(p)


def rows(flags: list[bool], metric: str = "all_fields_correct") -> list[dict[str, Any]]:
    return [{"id": f"i{i}", metric: f} for i, f in enumerate(flags)]


def test_paired_mcnemar_counts() -> None:
    a = rows([True, True, True, False, False, True, True, True])
    b = rows([True, False, False, False, True, False, False, True])
    res = paired_mcnemar(a, b)
    assert res["a_only_correct"] == 4  # i1, i2, i5, i6
    assert res["b_only_correct"] == 1  # i4
    assert res["p_value"] == pytest.approx(2 * (1 + 5) / 32)
    assert res["n"] == 8


def test_paired_mcnemar_rejects_mismatched_items() -> None:
    with pytest.raises(ValueError):
        paired_mcnemar(rows([True]), [{"id": "other", "all_fields_correct": True}])


def test_bootstrap_ci() -> None:
    def rate(rs: Any) -> float:
        return sum(r["m"] for r in rs) / len(rs)

    all_ones = [{"m": 1}] * 20
    assert bootstrap_ci(all_ones, rate) == (1.0, 1.0)
    half = [{"m": i % 2} for i in range(200)]
    lo, hi = bootstrap_ci(half, rate, n_boot=500, seed=7)
    assert lo < 0.5 < hi
    assert 0.38 < lo and hi < 0.62
    assert bootstrap_ci(half, rate, n_boot=500, seed=7) == (lo, hi)  # deterministic


def test_summarize_shapes() -> None:
    scores = [
        score_item(str(i), GOLD, out() if i % 2 else out(category="data"), SCHEMA).to_dict()
        for i in range(10)
    ]
    s = summarize(scores, n_boot=200, seed=1)
    assert s["n"] == 10
    assert s["json_valid"]["value"] == 1.0
    assert s["category_correct"]["value"] == 0.5
    assert s["all_fields_correct"]["value"] == 0.5
    # gold is all "network": F1(network) = 2*1*0.5/1.5 = 2/3
    assert s["category_macro_f1"]["value"] == pytest.approx(2 / 3, abs=1e-6)
    lo, hi = s["all_fields_correct"]["ci95"]
    assert lo <= 0.5 <= hi
