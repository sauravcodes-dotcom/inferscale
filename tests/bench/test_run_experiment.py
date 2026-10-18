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
from evals.dataset import ROOT
from inferscale.backends.fake import FakeBackend
from tests.bench.conftest import LiveFactory

E1 = ROOT / "configs" / "experiments" / "e1_baseline.yaml"
SMOKE = ROOT / "configs" / "experiments" / "bench_smoke.yaml"
PROMPTS = [[{"role": "user", "content": f"incident {i}"}] for i in range(5)]


def test_configs_load() -> None:
    e1 = load_experiment(E1)
    w = e1.workload
    assert (w.dataset, w.partition, w.prompt, w.response_format) == (
        "pag",
        "pool",
        "pag_v1",
        "json_schema",
    )
    assert (w.max_tokens, w.temperature) == (1024, 0)
    assert (e1.concurrency, e1.warmup, e1.repeats, e1.cooldown_s) == ([1], 5, 3, 120)
    assert e1.preflight.enforce
    smoke = load_experiment(SMOKE)
    assert smoke.smoke and not smoke.preflight.enforce and smoke.workload.n_requests == 10
    assert smoke.concurrency == [1, 2]


def test_preflight_clean() -> None:
    exp = load_experiment(E1)
    reasons = preflight_reasons(
        exp, power=lambda: "AC", swap=lambda: 0.0, processes=lambda: ["launchd"]
    )
    assert reasons == []


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
        exp, power=lambda: "battery", swap=lambda: 504.2, processes=lambda: procs
    )
    text = "\n".join(reasons)
    assert "not on AC power (power source: battery)" in text
    assert "swap used 504.2 MB > 64 MB" in text
    assert "Google Chrome" in text and "Safari" in text and "ollama" in text
    for absent in ("crashpad", "SafariBookmarksSyncAgent", "Claude"):
        assert absent not in text
    assert len(reasons) == 3


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
    backend = FakeBackend(first_token_delay_s=0.02, token_delay_s=0.005)
    url = await live_gateway(backend, admission={"max_inflight": 4})
    exp = Experiment.model_validate(
        {
            "name": "t_exp",
            "smoke": True,
            "workload": {"n_requests": 4, "max_tokens": 16},
            "concurrency": [1, 2],
            "warmup": 1,
            "repeats": 2,
            "cooldown_s": 0,
            "gateway_url": url,
            "engine_url": "",
            "report": str(tmp_path / "r.md"),
        }
    )
    swaps = iter([[10.0, 12.0], [10.0, 60.0], [10.0, 11.0], [10.0, 11.0]])
    rf = {"type": "json_object"}
    result = await run_experiment(
        exp,
        prompts=PROMPTS,
        workload_fingerprint={"dataset": {"name": "fake"}, "prompt": {"prompt_version": "t.p1"}},
        preflight={"warnings": ["test"]},
        response_format=rf,
        sampler_factory=lambda _e: StubSampler(next(swaps)),
        log=lambda _m: None,
    )
    assert result["run_kind"] == "smoke"
    assert len(result["runs"]) == 2 * 2  # loads x repeats, gateway only
    assert {r["target"] for r in result["runs"]} == {"gateway"}
    assert backend.requests[0]["response_format"] == rf  # forwarded through the gateway
    assert result["runs"][1]["validity"]["valid"] is False  # swap +50 MB in the second run
    assert result["valid"] is False
    assert all(r["summary"]["n_ok"] == 4 for r in result["runs"])
    assert result["fingerprint"]["prompt"]["prompt_version"] == "t.p1"
    out = write_result(exp, result, tmp_path / "bench")
    loaded = json.loads(out.read_text())
    a = analyze.analyze(loaded)
    assert set(a["pooled"]) == {"gateway|c=1", "gateway|c=2"}
    assert a["pooled"]["gateway|c=1"]["ttft_s"]["n"] == 8  # 2 repeats x 4 requests
    assert a["memory"]["peak_engine_rss_bytes"] == 5 * 2**30
    assert a["memory"]["swap_growth_mb_max"] == 50.0
    md = analyze.render_markdown(loaded, a)
    assert "SMOKE" in md and "INVALID RUN" in md and "| gateway | c=1 |" in md
    assert "Engine buffers (MiB, from log): n/a" in md


def test_render_marks_missing_as_na() -> None:
    run = {
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
    result: dict[str, Any] = {"experiment": {"name": "x"}, "runs": [run]}
    md = analyze.render_markdown(result, analyze.analyze(result))
    assert "| gateway | c=1 | 0/1 | n/a | n/a |" in md
    assert "Peak llama-server RSS: n/a" in md
    assert "engine_unavailable" in md
