"""Sweep config resolution, engine counter deltas and report rendering (no processes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.run_experiment import Workload, engine_delta, load_workload
from bench.sweep import (
    Sweep,
    Variant,
    engine_env_value,
    gateway_env,
    kv_mib,
    load_sweep,
    model_mismatch,
    render_markdown,
)
from evals.dataset import ROOT


def sweep(*variants: Variant) -> Sweep:
    return Sweep(name="s", experiment="x.yaml", report="r.md", variants=list(variants))


def test_engine_env_override_wins_over_profile() -> None:
    assert engine_env_value("bfcl", {}, "ENGINE_PARALLEL") == "2"
    assert engine_env_value("bfcl", {"ENGINE_PARALLEL": "4"}, "ENGINE_PARALLEL") == "4"
    assert engine_env_value("bfcl", {}, "NOT_SET") is None


def test_gateway_admission_matches_engine_slots() -> None:
    v = Variant(name="slots4", engine_env={"ENGINE_PARALLEL": "4"})
    assert gateway_env(sweep(v), v)["INFERSCALE_ADMISSION__MAX_INFLIGHT"] == "4"
    v2 = Variant(name="d", gateway_env={"INFERSCALE_ADMISSION__MAX_QUEUE": "16"})
    env = gateway_env(sweep(v2), v2)
    assert env == {
        "INFERSCALE_ADMISSION__MAX_INFLIGHT": "2",
        "INFERSCALE_ADMISSION__MAX_QUEUE": "16",
    }


def test_home_is_expanded_in_engine_env() -> None:
    v = Variant(name="q", engine_env={"ENGINE_MODEL": "~/m.gguf"})
    assert v.engine_env["ENGINE_MODEL"] == str(Path.home() / "m.gguf")


def test_engine_delta_rates_and_cache_reuse() -> None:
    before = {
        "llamacpp:prompt_tokens_total": 100.0,
        "llamacpp:prompt_seconds_total": 1.0,
        "llamacpp:tokens_predicted_total": 10.0,
        "llamacpp:tokens_predicted_seconds_total": 0.5,
    }
    after = {
        "llamacpp:prompt_tokens_total": 400.0,
        "llamacpp:prompt_seconds_total": 2.0,
        "llamacpp:tokens_predicted_total": 60.0,
        "llamacpp:tokens_predicted_seconds_total": 3.0,
    }
    d = engine_delta(before, after, client_prompt_tokens=1200)
    assert d["prefill_tok_s"] == 300.0 and d["decode_tok_s_per_request"] == 20.0
    assert d["prompt_cache_reuse_fraction"] == pytest.approx(0.75)  # 300 of 1200 evaluated
    missing = engine_delta({}, after, 1200)
    assert missing["prefill_tok_s"] is None and missing["prompt_cache_reuse_fraction"] is None


def test_bfcl_workload_is_dev_only() -> None:
    with pytest.raises(SystemExit, match="dev partition only"):
        load_workload(Workload(dataset="bfcl", partition="test"))


def test_render_markdown_with_skipped_variant() -> None:
    dist = {"p50": 2.0, "p95": 3.0, "p50_ci95": [1.9, 2.1]}
    run = {
        "concurrency": 1,
        "summary": {"n": 4, "n_ok": 4, "errors": {}, "e2e_s": dist, "throughput_req_s": 0.5},
        "engine": {
            "prefill_tok_s": 300.0,
            "decode_tok_s_per_request": 20.0,
            "prompt_cache_reuse_fraction": 0.25,
        },
        "samples": [{"engine_rss_bytes": 6e9}, {"engine_rss_bytes": 7e9}],
        "validity": {"swap_growth_mb": 0.0},
    }
    res = {
        "sweep": {"name": "s", "description": "d"},
        "run_kind": "smoke",
        "started_at": "2026-10-19T00:00:00+00:00",
        "git": {"sha": "abc", "dirty": False},
        "valid": True,
        "variants": [
            {
                "variant": {"name": "a"},
                "skipped": False,
                "kv_cache_mib_computed": 2304.0,
                "result": {"runs": [run]},
            },
            {"variant": {"name": "b"}, "skipped": True},
        ],
    }
    md = render_markdown(res)
    assert "SMOKE RUN" in md
    assert (
        "| a | 1 | 4/4 | 2.00 [1.90, 2.10] | 3.00 | 0.500 | 300 | 20.0 | 25% | 7.00 | 2304 | 0 |"
        in md
    )
    assert "| b | skipped" in md


def test_committed_sweeps_load() -> None:
    for p in sorted((ROOT / "configs" / "sweeps").glob("*.yaml")):
        s = load_sweep(p)
        assert (ROOT / s.experiment).is_file(), p


def test_kv_computed_from_context() -> None:
    v = Variant(name="slots4", engine_env={"ENGINE_CTX": "32768"})
    assert kv_mib(sweep(v), v) == 4608.0  # 32768 tokens x 144 KiB
    assert kv_mib(sweep(Variant(name="d")), Variant(name="d")) == 2304.0  # bfcl profile 16384


def test_bfcl_shuffle_is_seeded_and_mixes_categories() -> None:
    if not (ROOT / "results" / "bfcl" / "raw" / "requests-dev.jsonl").is_file():
        pytest.skip("BFCL dev requests not exported")
    w = Workload(dataset="bfcl", partition="dev", shuffle_seed=7)
    a, _, fa = load_workload(w)
    b, _, fb = load_workload(w)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert fa["workload"]["order_sha256"] == fb["workload"]["order_sha256"]
    assert len({r["category"] for r in a[:40]}) > 3  # the unshuffled file starts with one category


def test_model_version_reaches_gateway_and_is_checked_against_engine() -> None:
    v = Variant(name="q8", model_version="qwen3-8b-q8-v1")
    assert gateway_env(sweep(v), v)["INFERSCALE_ACTIVE_MODEL"] == "qwen3-8b-q8-v1"
    assert model_mismatch(v, {"model_path": "/m/Qwen3-8B-Q8_0.gguf"}) is None
    assert "registry expects 'Qwen3-8B-Q8_0.gguf'" in (
        model_mismatch(v, {"model_path": "/m/Qwen3-8B-Q4_K_M.gguf"}) or ""
    )
    assert model_mismatch(Variant(name="d"), None) is not None  # unknown engine model: refuse
