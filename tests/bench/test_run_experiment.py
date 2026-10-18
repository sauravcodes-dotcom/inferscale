from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from bench import analyze
from bench.memory_sampler import MemorySampler
from bench.run_experiment import (
    Experiment,
    load_experiment,
    preflight_reasons,
    run_experiment,
    write_result,
)
from evals.dataset import ROOT, TRIAGE_V1, load_jsonl
from inferscale.backends.fake import FakeBackend
from tests.bench.conftest import LiveFactory

E1 = ROOT / "configs" / "experiments" / "e1_baseline.yaml"
SMOKE = ROOT / "configs" / "experiments" / "bench_smoke.yaml"
REVIEWED = [{"review_status": "reviewed"}]


def test_configs_load() -> None:
    e1 = load_experiment(E1)
    assert (e1.workload.split, e1.workload.max_tokens, e1.workload.temperature) == ("test", 256, 0)
    assert (e1.concurrency, e1.warmup, e1.repeats, e1.cooldown_s) == ([1], 5, 3, 120)
    assert e1.targets == ["gateway", "engine"] and e1.preflight.enforce
    smoke = load_experiment(SMOKE)
    assert smoke.smoke and not smoke.preflight.enforce and smoke.workload.n_requests == 10
    assert smoke.concurrency == [1, 2]


def test_preflight_clean() -> None:
    exp = load_experiment(E1)
    assert (
        preflight_reasons(
            exp, REVIEWED, power=lambda: "AC", swap=lambda: 0.0, processes=lambda: ["launchd"]
        )
        == []
    )


def test_preflight_reasons() -> None:
    exp = load_experiment(E1)
    procs = [
        "Google Chrome",
        "Google Chrome Helper (Renderer)",
        "Safari",
        "ollama",
        "chrome_crashpad_handler",
        "SafariBookmarksSyncAgent",
        "Claude",
    ]
    reasons = preflight_reasons(
        exp,
        [{"review_status": "draft"}],
        power=lambda: "battery",
        swap=lambda: 504.2,
        processes=lambda: procs,
    )
    text = "\n".join(reasons)
    assert "not on AC power (power source: battery)" in text
    assert "swap used 504.2 MB > 64 MB" in text
    assert "Google Chrome" in text and "Safari" in text and "ollama" in text
    assert (
        "crashpad" not in text and "SafariBookmarksSyncAgent" not in text and "Claude" not in text
    )
    assert "not reviewed" in text
    assert len(reasons) == 4


class StubSampler(MemorySampler):
    def __init__(self, swaps: list[float]) -> None:
        it = iter(swaps + [swaps[-1]] * 1000)

        async def scraper(
            _c: httpx.AsyncClient, _u: str | None, names: tuple[str, ...]
        ) -> dict[str, float | None]:
            return dict.fromkeys(names)

        super().__init__(
            None,
            None,
            interval_s=0.02,
            engine_pid=-1,
            scraper=scraper,
            probes={
                "swap_used_mb": lambda: next(it),
                "memory_pressure": lambda: 1,
                "thermal": lambda: {"thermal_warning": False, "performance_warning": False},
                "engine_rss_bytes": lambda: 5 * 2**30,
            },
        )


async def test_run_experiment_end_to_end(live_gateway: LiveFactory, tmp_path: Path) -> None:
    url = await live_gateway(
        FakeBackend(first_token_delay_s=0.02, token_delay_s=0.005), admission={"max_inflight": 4}
    )
    exp = Experiment.model_validate(
        {
            "name": "t_exp",
            "smoke": True,
            "workload": {"split": "dev", "n_requests": 4, "max_tokens": 16},
            "concurrency": [1, 2],
            "warmup": 1,
            "repeats": 2,
            "cooldown_s": 0,
            "targets": ["gateway", "engine"],
            # Both targets hit the fake gateway: the 'engine' target just omits the model field.
            "gateway_url": url,
            "engine_url": url,
            "report": str(tmp_path / "r.md"),
        }
    )
    swaps = iter([[10.0, 12.0], [10.0, 60.0]] + [[10.0, 11.0]] * 10)
    items = load_jsonl(TRIAGE_V1 / "dev.jsonl")
    result = await run_experiment(
        exp,
        dataset_items=items,
        dataset_file=TRIAGE_V1 / "dev.jsonl",
        preflight={"warnings": ["test"]},
        sampler_factory=lambda _e: StubSampler(next(swaps)),
        log=lambda _m: None,
    )
    assert result["run_kind"] == "smoke"
    assert len(result["runs"]) == 2 * 2 * 2  # loads x repeats x targets
    assert [r["target"] for r in result["runs"][:2]] == ["gateway", "engine"]
    assert result["runs"][1]["validity"]["valid"] is False  # swap +50 MB in the second run
    assert result["valid"] is False
    assert all(r["summary"]["n_ok"] == 4 for r in result["runs"])
    assert result["fingerprint"]["dataset"]["review_status"] == ["draft"]
    assert result["fingerprint"]["prompt"]["prompt_version"].startswith("triage_v1.")
    out = write_result(exp, result, tmp_path / "bench")
    loaded = json.loads(out.read_text())
    a = analyze.analyze(loaded)
    assert set(a["pooled"]) == {"gateway|c=1", "engine|c=1", "gateway|c=2", "engine|c=2"}
    assert a["pooled"]["gateway|c=1"]["ttft_s"]["n"] == 8  # 2 repeats x 4 requests
    assert a["gateway_overhead_ttft"]["c=1"]["median_s"] is not None
    assert a["memory"]["peak_engine_rss_bytes"] == 5 * 2**30
    assert a["memory"]["swap_growth_mb_max"] == 50.0
    md = analyze.render_markdown(loaded, a)
    assert "SMOKE" in md and "INVALID RUN" in md and "| gateway | c=1 |" in md
    assert "Engine buffers (MiB, from log): n/a" in md


def test_render_marks_missing_as_na() -> None:
    result: dict[str, Any] = {
        "experiment": {"name": "x"},
        "runs": [
            {
                "target": "gateway",
                "mode": "closed",
                "concurrency": 1,
                "measure_window_s": [0.0, 1.0],
                "records": [
                    {"phase": "measure", "ok": False, "error": "engine_unavailable", "status": 503}
                ],
                "samples": [],
                "validity": {"valid": None},
            }
        ],
    }
    md = analyze.render_markdown(result, analyze.analyze(result))
    assert "| gateway | c=1 | 0/1 | n/a | n/a |" in md
    assert "Peak llama-server RSS: n/a" in md
    assert "engine_unavailable" in md
