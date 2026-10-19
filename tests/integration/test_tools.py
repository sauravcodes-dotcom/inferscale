"""Integration: native tool calling through the in-process gateway against the real engine."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from inferscale.api.app import create_app
from inferscale.config import load_settings
from tests.conftest import SERVING_PATH
from tests.integration.test_engine import ENGINE_URL

pytestmark = pytest.mark.integration


def fn(name: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name.replace("_", " "),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


WEATHER = fn("get_weather", {"city": {"type": "string"}}, ["city"])
STOCK = fn("get_stock_price", {"ticker": {"type": "string"}}, ["ticker"])


@pytest.fixture(params=["engine", "gateway", "gateway_validated"])
async def gateway(request: pytest.FixtureRequest) -> AsyncIterator[httpx.AsyncClient]:
    try:
        async with httpx.AsyncClient(timeout=2) as probe:
            ok = (await probe.get(f"{ENGINE_URL}/health")).json().get("status") == "ok"
    except httpx.HTTPError:
        ok = False
    if not ok:
        pytest.fail(f"engine not healthy at {ENGINE_URL}; run `make engine` first")
    settings = load_settings(SERVING_PATH).model_copy(update={"tool_call_mode": request.param})
    app = create_app(settings=settings, install_signal_handlers=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw", timeout=120
    ) as c:
        yield c
    await app.state.gateway.backend.aclose()


async def test_tool_call_non_stream(gateway: httpx.AsyncClient) -> None:
    body = {
        "messages": [{"role": "user", "content": "What's the weather in Lisbon?"}],
        "tools": [WEATHER, STOCK],
        "temperature": 0,
        "max_tokens": 256,
    }
    r = await gateway.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert json.loads(calls[0]["function"]["arguments"])["city"].lower() == "lisbon"


async def test_tool_call_stream_parallel(gateway: httpx.AsyncClient) -> None:
    body = {
        "messages": [{"role": "user", "content": "Prices of MSFT and AAPL please."}],
        "tools": [STOCK],
        "temperature": 0,
        "max_tokens": 256,
        "stream": True,
        "parallel_tool_calls": True,
    }
    r = await gateway.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    calls: dict[int, dict[str, str]] = {}
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                slot["name"] += (tc.get("function") or {}).get("name") or ""
                slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    tickers = sorted(json.loads(c["arguments"])["ticker"].upper() for c in calls.values())
    assert tickers == ["AAPL", "MSFT"]
    assert all(c["name"] == "get_stock_price" for c in calls.values())


async def test_no_call_when_irrelevant(gateway: httpx.AsyncClient) -> None:
    body = {
        "messages": [{"role": "user", "content": "Say hello in Spanish."}],
        "tools": [WEATHER],
        "temperature": 0,
        "max_tokens": 64,
    }
    r = await gateway.post("/v1/chat/completions", json=body)
    choice = r.json()["choices"][0]
    assert not choice["message"].get("tool_calls") and choice["finish_reason"] == "stop"
    assert "<tool_call>" not in (choice["message"]["content"] or "")
