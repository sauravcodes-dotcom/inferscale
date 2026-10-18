"""Generic scorer tests with hand-computed expected values."""

from __future__ import annotations

from typing import Any

import pytest

from evals.scorers import bootstrap_ci, macro_f1, mcnemar_exact, paired_mcnemar


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
