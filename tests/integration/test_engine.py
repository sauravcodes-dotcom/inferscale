"""Integration tests: in-process gateway against the real llama-server on 127.0.0.1:8081."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
import pytest

from inferscale.api.app import create_app
from inferscale.config import load_settings
from tests.conftest import MODEL_VERSION, SERVING_PATH

pytestmark = pytest.mark.integration

ENGINE_URL = os.environ.get("INFERSCALE_ENGINE_URL", "http://127.0.0.1:8081")
PROMPT = [
    {
        "role": "system",
        "content": "Classify the alert severity as critical, warning or info. "
        'Reply with JSON {"severity": ..., "reason": ...}.',
    },
    {"role": "user", "content": "ALERT: disk usage on db-prod-01 at 97% and rising 2%/hour."},
]


@pytest.fixture
async def gateway() -> AsyncIterator[httpx.AsyncClient]:
    try:
        async with httpx.AsyncClient(timeout=2) as probe:
            ok = (await probe.get(f"{ENGINE_URL}/health")).json().get("status") == "ok"
    except httpx.HTTPError:
        ok = False
    if not ok:
        pytest.fail(f"engine not healthy at {ENGINE_URL}; run `make engine` first")
    app = create_app(settings=load_settings(SERVING_PATH), install_signal_handlers=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=120) as c:
        yield c
    await app.state.gateway.backend.aclose()


async def test_ready(gateway: httpx.AsyncClient) -> None:
    r = await gateway.get("/readyz")
    assert r.status_code == 200, r.text


async def test_non_stream(gateway: httpx.AsyncClient) -> None:
    body = {"messages": PROMPT, "temperature": 0, "max_tokens": 128}
    r = await gateway.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    assert "<think>" not in content
    assert "critical" in content.lower()
    assert data["model"] == MODEL_VERSION
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0


async def test_stream(gateway: httpx.AsyncClient) -> None:
    body = {
        "messages": PROMPT,
        "temperature": 0,
        "max_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    r = await gateway.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    data = [ln[6:] for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text.strip()
    assert "<think>" not in text
    usage = [c["usage"] for c in chunks if c.get("usage")]
    assert usage and usage[-1]["completion_tokens"] > 0
    metrics = (await gateway.get("/metrics")).text
    assert "inferscale_completion_tokens_total" in metrics
    assert 'inferscale_requests_total{outcome="ok"}' in metrics
