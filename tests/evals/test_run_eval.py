"""run_eval end to end against the in-process gateway backed by FakeBackend (canned JSON)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals import run_eval
from evals.dataset import TRIAGE_V1, write_jsonl
from inferscale.backends.fake import FakeBackend
from tests.conftest import AppFactory

CANNED = {
    "severity": "SEV2",
    "component": "payments",
    "category": "dependency",
    "needs_page": True,
    "summary": "PSP returning 503.",
}


def item(i: int, **label: Any) -> dict[str, Any]:
    return {
        "id": f"tri-{i:04d}",
        "input": {"alert_title": f"alert {i}", "alert_body": "body", "log_excerpt": "line1\nline2"},
        "label": {**CANNED, **label},
        "review_status": "draft",
        "meta": {"scenario": "t", "ambiguous": False},
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    shutil.copy(TRIAGE_V1 / "schema.json", d / "schema.json")
    write_jsonl(d / "dev.jsonl", [item(1), item(2, severity="SEV1"), item(3, category="network")])
    return d


def args(dataset: Path, out: Path, *extra: str) -> Any:
    return run_eval.parse_args(
        [
            "--split",
            "dev",
            "--dataset-dir",
            str(dataset),
            "--out-dir",
            str(out),
            "--base-url",
            "http://gw",
            "--engine-url",
            "",
            "--bootstrap",
            "200",
            "--quiet",
            *extra,
        ]
    )


async def test_run_eval_scores_and_writes(
    make_app: AppFactory, dataset: Path, tmp_path: Path
) -> None:
    backend = FakeBackend(tokens=(json.dumps(CANNED),))
    app, _ = make_app(backend)
    out = await run_eval.run(
        args(dataset, tmp_path / "res", "--allow-draft", "--response-format", "json_schema"),
        transport=httpx.ASGITransport(app=app),
    )
    res = json.loads(out.read_text())
    assert out.parent.name == "qwen3-8b-q4km-v1"
    assert res["run_kind"] == "full"
    s = res["summary"]
    assert s["n"] == 3
    assert s["json_valid"]["value"] == 1.0
    assert s["schema_valid"]["value"] == 1.0
    assert s["all_fields_correct"]["value"] == pytest.approx(1 / 3)
    assert s["severity_correct"]["value"] == pytest.approx(2 / 3)
    assert s["category_correct"]["value"] == pytest.approx(2 / 3)
    assert s["requests"]["ok"] == 3
    # Sent temperature 0 and a json_schema response_format through the gateway to the engine.
    sent = backend.requests[0]
    assert sent["temperature"] == 0
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert sent["messages"][0]["role"] == "system"
    fp = res["fingerprint"]
    assert fp["dataset"]["review_status"] == ["draft"]
    assert len(fp["dataset"]["sha256"]) == 64
    assert fp["prompt"]["prompt_version"].startswith("triage_v1.")
    assert fp["model"]["sha256"].startswith("d98cdcbd")
    assert set(fp["host"]) >= {"macos_version", "chip", "power_source"}
    assert "dirty" in fp["git"]
    assert [r["id"] for r in res["items"]] == ["tri-0001", "tri-0002", "tri-0003"]


async def test_full_run_on_draft_requires_flag(
    make_app: AppFactory, dataset: Path, tmp_path: Path
) -> None:
    app, _ = make_app(FakeBackend(tokens=(json.dumps(CANNED),)))
    with pytest.raises(SystemExit, match="unreviewed"):
        await run_eval.run(args(dataset, tmp_path / "res"), transport=httpx.ASGITransport(app=app))


async def test_limit_marks_smoke(make_app: AppFactory, dataset: Path, tmp_path: Path) -> None:
    app, _ = make_app(FakeBackend(tokens=("not json",)))
    out = await run_eval.run(
        args(dataset, tmp_path / "res", "--limit", "2"), transport=httpx.ASGITransport(app=app)
    )
    res = json.loads(out.read_text())
    assert res["run_kind"] == "smoke" and out.name.endswith("-smoke.json")
    assert "not a result" in res["note"]
    assert res["summary"]["n"] == 2
    assert res["summary"]["json_valid"]["value"] == 0.0


async def test_gateway_errors_are_recorded(
    make_app: AppFactory, dataset: Path, tmp_path: Path
) -> None:
    app, _ = make_app(FakeBackend(mode="error"))
    out = await run_eval.run(
        args(dataset, tmp_path / "res", "--limit", "1"), transport=httpx.ASGITransport(app=app)
    )
    row = json.loads(out.read_text())["items"][0]
    assert row["http_status"] == 502 and row["output"] is None
    assert row["score"]["json_valid"] is False
