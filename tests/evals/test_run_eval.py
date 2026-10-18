"""run_eval end to end against the in-process gateway + FakeBackend (canned JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals import freeze, run_eval
from evals.prompts import pag_v1, pag_v1_p2
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
        args(data, tmp_path / "res", "--split", "dev", "--prompt-version", "pag_v1.p2"),
        transport=httpx.ASGITransport(app=app),
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
    assert r0["prediction"] == CANNED and r0["prompt_version"] == "pag_v1.p2"
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
    assert s["fingerprint"]["prompt"]["prompt_sha256"] == pag_v1_p2.prompt_sha256()  # default p2
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


async def _dev_run(data: Any, res: Path, transport: Any, *extra: str) -> Path:
    out, _ = await run_eval.run(args(data, res, "--split", "dev", *extra), transport=transport)
    return out


async def test_freeze_guard(
    make_app: AppFactory, data: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = tmp_path / "res"
    frozen = tmp_path / "frozen.yaml"
    app, _ = make_app(FakeBackend(tokens=(json.dumps(CANNED),)), limits=LIMITS)
    transport = httpx.ASGITransport(app=app)
    p1, p2 = ("--prompt-version", "pag_v1.p1"), ("--prompt-version", "pag_v1.p2")
    final = ("--split", "test", "--i-am-running-final-test", "--frozen-file", str(frozen))

    async def test_run(*extra: str) -> Path:
        out, _ = await run_eval.run(args(data, res, *final, *extra), transport=transport)
        return out

    with pytest.raises(SystemExit, match="needs --i-am-running-final-test"):
        await run_eval.run(args(data, res, "--split", "test", *p1), transport=transport)
    with pytest.raises(SystemExit, match="no prompt is frozen"):
        await test_run(*p1)

    smoke = await _dev_run(data, res, transport, *p1, "--limit", "2")
    freeze.write_frozen("pag_v1.p1", smoke, "t", path=frozen)
    with pytest.raises(SystemExit, match="not a full dev run"):
        await test_run(*p1)

    full_p1 = await _dev_run(data, res, transport, *p1)
    freeze.write_frozen("pag_v1.p1", full_p1, "t", path=frozen)
    out = await test_run(*p1)  # accepted
    assert json.loads(out.read_text())["label"] == "test (final)"
    with pytest.raises(SystemExit, match=r"pag_v1\.p2 is not the frozen prompt \(pag_v1\.p1\)"):
        await test_run(*p2)

    # partial dev result (n smaller than the full dev partition)
    s = json.loads(full_p1.read_text())
    s["config"]["n"] = 2
    partial = full_p1.with_name("partial-dev-x.json")
    partial.write_text(json.dumps(s))
    freeze.write_frozen("pag_v1.p1", partial, "t", path=frozen)
    with pytest.raises(SystemExit, match="covers 2 items, full dev is 3"):
        await test_run(*p1)

    # referenced dev result missing
    freeze.write_frozen("pag_v1.p1", tmp_path / "gone.json", "t", path=frozen)
    with pytest.raises(SystemExit, match="does not exist"):
        await test_run(*p1)

    # dev result produced with another prompt
    full_p2 = await _dev_run(data, res, transport, *p2)
    freeze.write_frozen("pag_v1.p1", full_p2, "t", path=frozen)
    with pytest.raises(SystemExit, match="different prompt sha"):
        await test_run(*p1)

    # sha drift: the prompt file changed after freezing
    freeze.write_frozen("pag_v1.p1", full_p1, "t", path=frozen)
    monkeypatch.setattr(pag_v1, "prompt_sha256", lambda: "0" * 64)
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        await test_run(*p1)


def test_default_prompt_is_the_frozen_one() -> None:
    frozen = freeze.load_frozen()
    assert frozen is not None
    a = run_eval.parse_args(["--split", "dev"])
    assert a.prompt_version == frozen["prompt_version"]


def test_repo_freeze_is_consistent() -> None:
    frozen = freeze.load_frozen()
    assert frozen is not None and frozen["prompt_version"] == "pag_v1.p1"
    assert frozen["prompt_sha256"] == pag_v1.prompt_sha256()
    assert (
        freeze.dev_result_problem(
            frozen["dev_result_file"], frozen["prompt_sha256"], freeze.dev_size()
        )
        is None
    )
    assert freeze.final_test_refusal("pag_v1.p1", frozen) is None
    assert "not the frozen prompt" in (freeze.final_test_refusal("pag_v1.p2", frozen) or "")


def test_prompt_versions() -> None:
    from evals.prompts import get_prompt

    assert get_prompt("pag_v1.p1") is pag_v1 and get_prompt("pag_v1.p2") is pag_v1_p2
    assert pag_v1.prompt_sha256() != pag_v1_p2.prompt_sha256()
    assert "exactly 4 corrective" not in pag_v1.SYSTEM_PROMPT
    assert "exactly 4 actions that fix" in pag_v1_p2.SYSTEM_PROMPT
    assert "exactly as it is written in the Description" in pag_v1_p2.SYSTEM_PROMPT
    assert pag_v1_p2.response_format("json_schema") == pag_v1.response_format("json_schema")
    inp = {k: "x" for k in pag_v1.INPUT_ORDER}
    assert pag_v1_p2.build_messages(inp)[1] == pag_v1.build_messages(inp)[1]  # same user message
    with pytest.raises(ValueError):
        get_prompt("pag_v1.p9")
