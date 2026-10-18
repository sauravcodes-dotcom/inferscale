from __future__ import annotations

from typing import Any

import pytest

from evals.compare import action_stats, compare, paired_diff_ci


def test_paired_diff_ci_hand_computed() -> None:
    res = paired_diff_ci([0.1, 0.2, 0.3, 0.4], [0.2, 0.2, 0.5, 0.4], n_boot=500)
    assert res["diff"] == pytest.approx(0.075)  # (0.1 + 0 + 0.2 + 0) / 4
    assert res["b_better_items"] == 2 and res["a_better_items"] == 0
    lo, hi = res["ci95"]
    assert 0.0 <= lo <= 0.075 <= hi <= 0.2


def test_action_stats() -> None:
    s = action_stats([["aaaa", "bb"], ["cccccc", "d", "ee"]])
    assert s["count_distribution"] == {"2": 1, "3": 1}
    assert s["mean_count"] == 2.5 and s["mean_chars"] == pytest.approx(15 / 5)


def row(i: str, team: bool, f1: float, n_act: int = 3) -> dict[str, Any]:
    score = {
        k: 0.0
        for k in (
            "corrective_f1",
            "corrective_coverage",
            "corrective_precision",
            "preventive_f1",
            "preventive_coverage",
            "preventive_precision",
        )
    }
    score["corrective_f1"] = f1
    score.update(
        valid=True,
        type_correct=True,
        team_correct=team,
        severity_correct=False,
        grounded=True,
        grounded_substring=False,
        format_ok=True,
    )
    acts = ["x" * 10] * n_act
    return {
        "incident_id": i,
        "score": score,
        "prediction": {"corrective_actions": acts, "preventive_actions": acts},
        "reference": {"corrective_actions": ["r" * 20] * 4, "preventive_actions": ["r" * 20] * 4},
    }


def summary(v: str) -> dict[str, Any]:
    return {
        "fingerprint": {"prompt": {"prompt_version": v, "prompt_sha256": v * 4}, "git": {}},
        "started_at": "t",
        "run_kind": "full",
        "config": {"split": "dev"},
    }


def test_compare_counts_discordant_pairs() -> None:
    a = [row("1", True, 0.1), row("2", False, 0.2), row("3", False, 0.3), row("4", True, 0.4)]
    b = [
        row("1", True, 0.3, 4),
        row("2", True, 0.2, 4),
        row("3", True, 0.3, 4),
        row("4", False, 0.6, 4),
    ]
    res = compare(summary("p1"), a, summary("p2"), b, n_boot=200)
    team = res["binary"]["team_correct"]
    assert (team["a_only_correct"], team["b_only_correct"]) == (1, 2)
    assert team["p_value"] == pytest.approx(1.0)  # n=3, k=1: 2*(1+3)/8 = 1
    assert team["diff"] == pytest.approx(0.25)
    assert res["continuous"]["corrective_f1"]["diff"] == pytest.approx(0.1)
    assert res["actions"]["corrective"]["b"]["count_distribution"] == {"4": 4}
    assert res["actions"]["corrective"]["reference"]["mean_chars"] == 20
    with pytest.raises(ValueError):
        compare(summary("p1"), a, summary("p2"), b[:3])
