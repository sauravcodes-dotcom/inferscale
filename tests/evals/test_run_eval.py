"""run_eval end to end against the in-process gateway + FakeBackend (canned JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals import run_eval
from evals.prompts import pag_v1
from inferscale.backends.fake import FakeBackend
from tests.conftest import AppFactory
from tests.evals.test_pag import raw_record

LIMITS = {"max_tokens_cap": 1024}
CANNED = {
    "severity": "Sev-1",
    "root_cause_type": "Config Errors",
    "root_cause_team": "Logging",
    "root_cause_summary": "A configuration error.",
    "corrective_actions": [
        "fix 101 a and more words here",
        "fix 101 b and more words here",
        "a" * 25,
    ],
    "preventive_actions": ["prevent 101 with many more words", "b" * 25, "c" * 25],
}


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    raw = [raw_record(i) for i in range(106)]
    data_file = tmp_path / "pag.json"
    data_file.write_text(json.dumps(raw))
    dev_ids = [raw[i]["Incident_ID"] for i in (100, 101, 102)]
    test_ids = [raw[i]["Incident_ID"] for i in (103, 104, 105)]
    split = {"partitions": {"dev": {"incident_ids": dev_ids}, "test": {"incident_ids": test_ids}}}
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split))
    return data_file, split_file, dev_ids


def args(data: tuple[Path, Path, list[str]], results: Path, *extra: str) -> Any:
    data_file, split_file, _ = data
    return run_eval.parse_args(
        [
            "--data-file",
            str(data_file),
            "--split-file",
            str(split_file),
            "--results-dir",
            str(results),
            "--base-url",
            "http://gw",
            "--engine-url",
            "",
            "--bootstrap",
            "100",
            "--quiet",
            *extra,
        ]
    )


async def test_dev_run_writes_raw_and_summary(
    make_app: AppFactory, data: Any, tmp_path: Path
) -> None:
    backend = FakeBackend(tokens=(json.dumps(CANNED),))
    app, _ = make_app(backend, limits=LIMITS)
    out, raw = await run_eval.run(
        args(data, tmp_path / "res", "--split", "dev"), transport=httpx.ASGITransport(app=app)
    )
    rows = [json.loads(x) for x in raw.read_text().splitlines()]
    assert [r["incident_id"] for r in rows] == data[2]
    r0 = rows[0]
    for key in (
        "incident_id",
        "source_index",
        "split",
        "model_version",
        "prompt_version",
        "response_format",
        "raw_prediction",
        "prediction",
        "reference",
        "valid",
        "invalid_reason",
        "ttft_ms",
        "e2e_ms",
        "prompt_tokens",
        "completion_tokens",
        "decode_tok_s",
        "finish_reason",
        "http_status",
        "error",
    ):
        assert key in r0
    assert r0["valid"] and r0["finish_reason"] == "stop" and r0["http_status"] == 200
    assert r0["prediction"] == CANNED and r0["prompt_version"] == "pag_v1.p1"
    sent = backend.requests[0]
    assert sent["temperature"] == 0 and sent["max_tokens"] == 1024
    assert sent["response_format"]["json_schema"]["name"] == "pag_output_v1"
    assert "Severity" not in sent["messages"][1]["content"]
    s = json.loads(out.read_text())
    assert s["run_kind"] == "full" and s["label"] == "dev (not final)"
    assert out.parent.name == "qwen3-8b-q4km-v1"
    q = s["quality"]
    assert q["valid"]["value"] == 1.0
    assert q["type_correct"]["value"] == pytest.approx(1 / 3)  # records cycle through 3 types
    assert q["severity_majority_baseline"] == pytest.approx(2 / 3)
    assert s["fingerprint"]["dataset"]["ids_sha256"] == run_eval.ids_sha256(data[2])
    assert s["fingerprint"]["prompt"]["prompt_sha256"] == pag_v1.prompt_sha256()
    assert "raw_prediction" not in json.dumps(s)  # summary carries metrics only
    assert s["serving"]["requests_ok"] == 3 and s["serving"]["finish_reasons"] == {"stop": 3}


async def test_invalid_outputs_are_classified(
    make_app: AppFactory, data: Any, tmp_path: Path
) -> None:
    app, _ = make_app(FakeBackend(tokens=("{not json",)), limits=LIMITS)
    out, raw = await run_eval.run(
        args(data, tmp_path / "res", "--split", "dev", "--limit", "2"),
        transport=httpx.ASGITransport(app=app),
    )
    s = json.loads(out.read_text())
    assert (
        s["run_kind"] == "smoke"
        and out.parent.name == "smoke"
        and raw.name.endswith("-smoke.jsonl")
    )
    assert s["quality"]["invalid_breakdown"]["parse"] == 2
    app2, _ = make_app(FakeBackend(mode="error"), limits=LIMITS)
    out2, _ = await run_eval.run(
        args(data, tmp_path / "res", "--split", "dev", "--limit", "1"),
        transport=httpx.ASGITransport(app=app2),
    )
    assert json.loads(out2.read_text())["quality"]["invalid_breakdown"]["http_error"] == 1


async def test_test_split_guard(
    make_app: AppFactory, data: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = tmp_path / "res"
    app, _ = make_app(FakeBackend(tokens=(json.dumps(CANNED),)))
    transport = httpx.ASGITransport(app=app)
    with pytest.raises(SystemExit, match="needs --i-am-running-final-test"):
        await run_eval.run(args(data, res, "--split", "test"), transport=transport)
    with pytest.raises(SystemExit, match="no full dev run"):
        await run_eval.run(
            args(data, res, "--split", "test", "--i-am-running-final-test"), transport=transport
        )
    await run_eval.run(
        args(data, res, "--split", "dev", "--limit", "1"), transport=transport
    )  # smoke does not count
    with pytest.raises(SystemExit, match="no full dev run"):
        await run_eval.run(
            args(data, res, "--split", "test", "--i-am-running-final-test"), transport=transport
        )
    await run_eval.run(args(data, res, "--split", "dev"), transport=transport)
    out, _ = await run_eval.run(
        args(data, res, "--split", "test", "--i-am-running-final-test"), transport=transport
    )
    assert json.loads(out.read_text())["label"] == "test (final)"
    monkeypatch.setattr(pag_v1, "prompt_sha256", lambda: "0" * 64)
    with pytest.raises(SystemExit, match="prompt changed"):
        await run_eval.run(
            args(data, res, "--split", "test", "--i-am-running-final-test"), transport=transport
        )
