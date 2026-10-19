"""BFCL integration: aggregates (BFCL formulas), ingestion, error taxonomy, split, freeze."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.bfcl import freeze, metrics
from evals.bfcl.metrics import aggregates, classify_error, ingest, paired, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import make_bfcl_split


def rows(category: str, flags: list[bool]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{category}_{i}",
            "category": category,
            "correct": f,
            "error_class": None if f else "x",
        }
        for i, f in enumerate(flags)
    ]


def test_aggregates_follow_bfcl_formulas() -> None:
    cats = {
        "simple_python": rows("simple_python", [True, True, True, False]),  # 0.75
        "simple_java": rows("simple_java", [True, False]),  # 0.5
        "simple_javascript": rows("simple_javascript", [True]),  # 1.0
        "multiple": rows("multiple", [True, True]),  # 1.0
        "parallel": rows("parallel", [True, False]),  # 0.5
        "parallel_multiple": rows("parallel_multiple", [False, False]),  # 0.0
        "live_simple": rows("live_simple", [True, True, True]),  # 3/3
        "live_multiple": rows("live_multiple", [False]),  # 0/1
    }
    a = aggregates(cats)
    # simple = unweighted mean(0.75, 0.5, 1.0) = 0.75; non-live AST = mean(0.75, 1, 0.5, 0) = 0.5625
    assert a["non_live_simple"] == pytest.approx(0.75)
    assert a["non_live_ast"] == pytest.approx(0.5625)
    # live = count-weighted: 3 correct / 4 entries
    assert a["live"] == pytest.approx(0.75)
    assert a["irrelevance"] is None


def test_non_live_ast_is_none_when_a_part_is_missing() -> None:
    a = aggregates(
        {"simple_python": rows("simple_python", [True]), "multiple": rows("multiple", [True])}
    )
    assert a["non_live_ast"] is None  # BFCL would silently count the missing parts as 0


def test_summary_ci_brackets_point_estimate() -> None:
    rs = rows("simple_python", [True] * 30 + [False] * 10) + rows("multiple", [True] * 20)
    s = summarize(rs, n_boot=300)
    v = s["metrics"]["simple_python"]
    assert v["value"] == pytest.approx(0.75) and v["n"] == 40
    lo, hi = v["ci95"]
    assert lo < 0.75 < hi
    assert s["metrics"]["multiple"]["ci95"] == [1.0, 1.0]


@pytest.mark.parametrize(
    ("error_type", "failed", "cls"),
    [
        ("simple_function_checker:wrong_func_name", False, "wrong_function"),
        ("simple_function_checker:missing_required", False, "wrong_arguments"),
        ("type_error:nested", False, "wrong_arguments"),
        ("value_error:string", False, "wrong_arguments"),
        ("ast_decoder:decoder_failed", False, "undecodable_output"),
        ("irrelevance_error:decoder_success", False, "called_when_irrelevant"),
        ("relevance_error:decoder_failed", False, "no_call_when_relevant"),
        ("parallel_function_checker_no_order:wrong_count", False, "wrong_call_count"),
        ("parallel_function_checker_no_order:cannot_find_match", False, "wrong_call_set"),
        ("something:new", False, "other"),
        (None, True, "serving_error"),
    ],
)
def test_error_taxonomy(error_type: str | None, failed: bool, cls: str) -> None:
    assert classify_error(error_type, failed) == cls


def test_ingest_reads_official_files(tmp_path: Path) -> None:
    model = "inferscale_m-FC"
    (tmp_path / "result" / model / "non_live").mkdir(parents=True)
    (tmp_path / "score" / model / "non_live").mkdir(parents=True)
    results = [
        {
            "id": "simple_python_0",
            "result": [{"f": '{"a": 1}'}],
            "input_token_count": 100,
            "output_token_count": 9,
            "latency": 1.5,
        },
        {
            "id": "simple_python_1",
            "result": [{"g": "{}"}],
            "input_token_count": 90,
            "output_token_count": 7,
            "latency": 1.0,
        },
        {"id": "simple_python_2", "result": "Error during inference: 502"},
    ]
    (tmp_path / "result" / model / "non_live" / "BFCL_v4_simple_python_result.json").write_text(
        "\n".join(json.dumps(r) for r in results)
    )
    score = [
        {"accuracy": 1 / 3, "correct_count": 1, "total_count": 3},
        {"id": "simple_python_1", "error_type": "simple_function_checker:wrong_func_name"},
        {"id": "simple_python_2", "error_type": "ast_decoder:decoder_failed"},
    ]
    (tmp_path / "score" / model / "non_live" / "BFCL_v4_simple_python_score.json").write_text(
        "\n".join(json.dumps(r) for r in score)
    )
    got = {r["id"]: r for r in ingest(tmp_path, model)}
    assert got["simple_python_0"]["correct"] and got["simple_python_0"]["made_call"]
    assert got["simple_python_1"]["error_class"] == "wrong_function"
    assert (
        got["simple_python_2"]["error_class"] == "serving_error"
        and not got["simple_python_2"]["made_call"]
    )
    assert got["simple_python_0"]["input_tokens"] == 100


def test_paired_mcnemar() -> None:
    a = rows("simple_python", [True, True, False, False, True])
    b = rows("simple_python", [True, False, True, True, True])
    res = paired(a, b)["overall"]
    assert (res["a_only_correct"], res["b_only_correct"]) == (1, 2)
    assert res["p_value"] == pytest.approx(1.0)


# --- split ------------------------------------------------------------------------


def entry(eid: str, q: str) -> dict[str, Any]:
    cat = eid.rsplit("_", 1)[0]
    return {
        "id": eid,
        "category": cat,
        "question": [[{"role": "user", "content": q}]],
        "function": [],
    }


def test_live_group_parsing() -> None:
    assert make_bfcl_split.live_group("live_multiple_217-93-0") == "93"
    assert make_bfcl_split.live_group("simple_python_3") is None


def test_split_never_separates_connected_entries() -> None:
    es = {}
    for i in range(60):
        es[f"simple_python_{i}"] = entry(f"simple_python_{i}", f"question {i}")
    es["irrelevance_0"] = entry("irrelevance_0", "question 7")  # same text as simple_python_7
    for i in range(30):
        es[f"live_simple_{i}-{i % 5}-0"] = entry(
            f"live_simple_{i}-{i % 5}-0", f"live {i}"
        )  # 5 source groups
    parts = make_bfcl_split.split(es, seed=1, frac=0.2)
    dev, test = set(parts["dev"]), set(parts["test"])
    assert dev.isdisjoint(test) and len(dev | test) == len(es)
    assert ("simple_python_7" in dev) == ("irrelevance_0" in dev)
    for g in range(5):
        members = {k for k in es if k.startswith("live_simple_") and k.split("-")[1] == str(g)}
        assert members <= dev or members <= test
    assert 8 <= sum(1 for k in dev if k.startswith("simple_python")) <= 12  # ~20% of 60
    assert make_bfcl_split.split(es, seed=1, frac=0.2) == parts  # deterministic


def test_committed_split_is_up_to_date() -> None:
    if not make_bfcl_split.DATA_DIR.is_dir():
        pytest.skip("BFCL environment not installed (make bfcl-setup)")
    assert make_bfcl_split.render(make_bfcl_split.build()) == make_bfcl_split.OUT.read_text()


# --- freeze guard -----------------------------------------------------------------


def dev_summary(tmp_path: Path, config_sha: str, **over: Any) -> Path:
    split = json.loads(
        (
            Path(metrics.__file__).resolve().parents[2] / "splits" / "bfcl_v4_single_turn_v1.json"
        ).read_text()
    )
    cats = sorted(c for c, v in split["partitions"]["dev"]["ids"].items() if v)
    s = {
        "partition": "dev",
        "run_kind": "full",
        "config": {"config_sha256": config_sha},
        "fingerprint": {"split": {"categories": cats}},
    }
    s.update(over)
    p = tmp_path / f"dev-{len(list(tmp_path.iterdir()))}.json"
    p.write_text(json.dumps(s))
    return p


def test_freeze_guard(tmp_path: Path) -> None:
    good = dev_summary(tmp_path, "abc")
    frozen = {
        "config_sha256": "abc",
        "dev_result_file": str(good),
        "config": {"temperature": 0.001},
    }
    assert freeze.test_refusal({"config_sha256": "abc"}, frozen) is None
    drift = freeze.test_refusal({"config_sha256": "zzz", "temperature": 0.7}, frozen)
    assert drift is not None and "temperature" in drift
    assert "no BFCL configuration is frozen" in (
        freeze.test_refusal({"config_sha256": "abc"}, None) or ""
    )
    smoke = dev_summary(tmp_path, "abc", run_kind="smoke")
    assert "not a full dev run" in (
        freeze.test_refusal({"config_sha256": "abc"}, {**frozen, "dev_result_file": str(smoke)})
        or ""
    )
    partial = dev_summary(tmp_path, "abc", fingerprint={"split": {"categories": ["simple_python"]}})
    assert "every dev category" in (
        freeze.test_refusal({"config_sha256": "abc"}, {**frozen, "dev_result_file": str(partial)})
        or ""
    )
    other = dev_summary(tmp_path, "different")
    assert "different run configuration" in (
        freeze.test_refusal({"config_sha256": "abc"}, {**frozen, "dev_result_file": str(other)})
        or ""
    )
    assert "does not exist" in (
        freeze.test_refusal(
            {"config_sha256": "abc"}, {**frozen, "dev_result_file": str(tmp_path / "x.json")}
        )
        or ""
    )


# --- baselines --------------------------------------------------------------------

from evals.bfcl import baselines  # noqa: E402


def test_bm25_prefers_matching_function() -> None:
    entry = {
        "question": [
            [{"role": "user", "content": "What is the weather forecast in Paris tomorrow?"}]
        ],
        "function": [
            {
                "name": "stock.get_price",
                "description": "Latest stock price",
                "parameters": {"properties": {"ticker": {}}},
            },
            {
                "name": "weather.forecast",
                "description": "Weather forecast for a city",
                "parameters": {"properties": {"city": {}}},
            },
        ],
    }
    assert baselines.bm25_choice(entry) == "weather.forecast"
    assert baselines.as_fc_name("weather.forecast") == "weather_forecast"
    assert baselines.bm25_choice({"question": [[]], "function": []}) is None


def test_tokens_split_camel_and_snake() -> None:
    assert baselines.tokens("getUserName math.factorial") == [
        "get",
        "user",
        "name",
        "math",
        "factorial",
    ]


def test_selection_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    truth = {"m_0": [{"weather.forecast": {"city": ["Paris"]}}], "m_1": [{"a.b": {}}]}
    monkeypatch.setattr(
        baselines, "load_possible_answers", lambda cat: truth if cat == "multiple" else {}
    )
    res = baselines.selection_accuracy(
        {"multiple": ["m_0", "m_1"]}, {"m_0": "weather_forecast", "m_1": "x"}
    )
    assert res["multiple"] == {"n": 2, "accuracy": 0.5}


def test_paired_bootstrap_compare() -> None:
    from evals.bfcl.compare import compare

    a = rows("simple_python", [True, False, False, True]) + rows("live_simple", [False, False])
    b = rows("simple_python", [True, True, False, True]) + rows("live_simple", [True, False])
    res = compare(a, b, n_boot=200)
    assert res["aggregates"]["live"]["diff"] == pytest.approx(0.5)
    lo, hi = res["aggregates"]["live"]["ci95"]
    assert lo <= 0.5 <= hi
    assert res["mcnemar"]["by_category"]["simple_python"]["b_only_correct"] == 1
    assert res["error_classes"]["a"] == {"x": 4} and res["error_classes"]["b"] == {"x": 2}
