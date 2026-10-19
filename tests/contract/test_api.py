"""Contract tests: the HTTP API against FakeBackend, in-process via httpx ASGITransport."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from inferscale.backends.fake import FakeBackend
from tests.conftest import MODEL_VERSION, AppFactory, chat_body, client_for

EXPECTED_TEXT = "Hello from the fake backend."


def sse_data(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def sse_chunks(text: str) -> list[dict[str, Any]]:
    return [json.loads(d) for d in sse_data(text) if d != "[DONE]"]


def stream_text(chunks: list[dict[str, Any]]) -> str:
    return "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices")
    )


def metric_value(text: str, name: str, labels: str = "") -> float:
    key = f"{name}{{{labels}}}" if labels else name
    for line in text.splitlines():
        if line.startswith(key + " "):
            return float(line.split()[-1])
    raise AssertionError(f"metric {key} not found")


# --- happy paths -------------------------------------------------------------


async def test_non_stream(make_app: AppFactory) -> None:
    app, _ = make_app()
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(), headers={"X-Request-ID": "abc"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "abc"
    body = r.json()
    assert body["model"] == MODEL_VERSION
    assert body["choices"][0]["message"]["content"] == EXPECTED_TEXT
    assert body["usage"]["completion_tokens"] == 6


async def test_request_id_generated(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", json=chat_body())
    assert len(r.headers["x-request-id"]) == 32


async def test_stream(make_app: AppFactory) -> None:
    backend = FakeBackend()
    app, gw = make_app(backend)
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    data = sse_data(r.text)
    assert data[-1] == "[DONE]"
    chunks = sse_chunks(r.text)
    assert stream_text(chunks) == EXPECTED_TEXT
    assert all(c["model"] == MODEL_VERSION for c in chunks)
    assert not any("usage" in c for c in chunks)  # client did not ask for usage
    # ...but the gateway asked the engine for it and recorded it.
    assert backend.requests[0]["stream_options"] == {"include_usage": True}
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_completion_tokens_total") == 6
    assert metric_value(m, "inferscale_prompt_tokens_total") == 7
    assert metric_value(m, "inferscale_ttft_seconds_count") == 1
    assert gw.admission.inflight == 0


async def test_stream_include_usage(client: httpx.AsyncClient) -> None:
    body = chat_body(stream=True, stream_options={"include_usage": True})
    r = await client.post("/v1/chat/completions", json=body)
    usage = [c for c in sse_chunks(r.text) if "usage" in c]
    assert len(usage) == 1
    assert usage[0]["choices"] == []
    assert usage[0]["usage"]["completion_tokens"] == 6


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (chat_body(temperature=3), 400, "invalid_request"),
        (chat_body(foo=1), 400, "invalid_request"),
        (chat_body(max_tokens=513), 400, "invalid_request"),
        (chat_body(model="gpt-4"), 404, "model_not_found"),
        ({"messages": []}, 400, "invalid_request"),
        ([1, 2], 400, "invalid_request"),
    ],
)
async def test_invalid_requests(
    client: httpx.AsyncClient, body: Any, status: int, code: str
) -> None:
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code


async def test_malformed_json(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 400


async def test_active_model_name_accepted(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", json=chat_body(model=MODEL_VERSION))
    assert r.status_code == 200


# --- admission ---------------------------------------------------------------


async def test_429_when_full(make_app: AppFactory) -> None:
    app, gw = make_app(
        FakeBackend(first_token_delay_s=0.3), admission={"max_inflight": 1, "max_queue": 0}
    )
    async with client_for(app) as c:
        slow = asyncio.create_task(c.post("/v1/chat/completions", json=chat_body()))
        await asyncio.sleep(0.05)
        t0 = asyncio.get_running_loop().time()
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
        elapsed = asyncio.get_running_loop().time() - t0
        assert (await slow).status_code == 200
    assert r.status_code == 429
    assert r.headers["retry-after"] == "1"
    assert r.json()["error"]["code"] == "queue_full"
    assert elapsed < 0.1  # rejected immediately, not after waiting
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_rejected_total", 'reason="queue_full"') == 1
    assert metric_value(m, "inferscale_requests_total", 'outcome="rejected"') == 1


async def test_queued_requests_complete(make_app: AppFactory) -> None:
    app, _ = make_app(
        FakeBackend(first_token_delay_s=0.05), admission={"max_inflight": 1, "max_queue": 3}
    )
    async with client_for(app) as c:
        rs = await asyncio.gather(
            *[c.post("/v1/chat/completions", json=chat_body()) for _ in range(4)]
        )
    assert [r.status_code for r in rs] == [200] * 4


async def test_queue_timeout_429(make_app: AppFactory) -> None:
    app, _ = make_app(
        FakeBackend(first_token_delay_s=0.5),
        admission={"max_inflight": 1, "max_queue": 1, "queue_timeout_s": 0.05},
    )
    async with client_for(app) as c:
        rs = await asyncio.gather(
            *[c.post("/v1/chat/completions", json=chat_body()) for _ in range(2)]
        )
    codes = sorted(r.status_code for r in rs)
    assert codes == [200, 429]
    assert next(r for r in rs if r.status_code == 429).json()["error"]["code"] == "queue_timeout"


# --- deadlines and engine failures -------------------------------------------


async def test_504_on_hang_stream(make_app: AppFactory) -> None:
    backend = FakeBackend(mode="hang")
    app, gw = make_app(backend, deadlines={"first_token_timeout_s": 0.1})
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "deadline_exceeded"
    assert gw.admission.inflight == 0
    assert backend.active_streams == 0  # upstream was cancelled


async def test_504_on_hang_non_stream(make_app: AppFactory) -> None:
    app, gw = make_app(FakeBackend(mode="hang"), deadlines={"total_timeout_s": 0.1})
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 504
    assert gw.admission.inflight == 0
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_requests_total", 'outcome="timeout"') == 1


async def test_total_deadline_mid_stream(make_app: AppFactory) -> None:
    app, gw = make_app(FakeBackend(token_delay_s=0.1), deadlines={"total_timeout_s": 0.25})
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 200  # headers were already sent
    last = sse_chunks(r.text)[-1]
    assert last["error"]["code"] == "deadline_exceeded"
    assert gw.admission.inflight == 0


@pytest.mark.parametrize(
    ("mode", "status", "code"),
    [("error", 502, "engine_error"), ("unavailable", 503, "engine_unavailable")],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_engine_failures(
    make_app: AppFactory, mode: Any, status: int, code: str, stream: bool
) -> None:
    app, gw = make_app(FakeBackend(mode=mode))
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=stream))
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert gw.admission.inflight == 0


async def test_engine_drops_mid_stream(make_app: AppFactory) -> None:
    app, gw = make_app(FakeBackend(mode="disconnect"))
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True))
    assert r.status_code == 200
    chunks = sse_chunks(r.text)
    assert chunks[-1]["error"]["code"] == "engine_error"
    assert "[DONE]" not in sse_data(r.text)
    assert gw.admission.inflight == 0
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_requests_total", 'outcome="engine_error"') == 1


# --- client disconnect (raw ASGI: httpx's ASGITransport buffers whole responses) ---


async def drive_and_disconnect(app: FastAPI, body: dict[str, Any], after_chunks: int) -> int:
    """Send one request, then disconnect after `after_chunks` body messages. Returns count."""
    raw = json.dumps(body).encode()
    disconnect = asyncio.Event()
    got = 0
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal got
        if message["type"] == "http.response.body" and message.get("body"):
            got += 1
            if got >= after_chunks:
                disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
        "state": {},
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    return got


async def test_client_disconnect_frees_slot(make_app: AppFactory) -> None:
    tokens = tuple(f" t{i}" for i in range(100))
    backend = FakeBackend(tokens=tokens, token_delay_s=0.02)
    app, gw = make_app(backend, admission={"max_inflight": 1, "max_queue": 0})
    got = await drive_and_disconnect(app, chat_body(stream=True, max_tokens=100), after_chunks=3)
    assert got < 20  # we really did stop early
    assert backend.streams_closed_early == 1
    assert backend.active_streams == 0
    assert gw.admission.inflight == 0
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_requests_total", 'outcome="client_disconnect"') == 1
    # The freed slot is immediately usable.
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
    assert r.status_code == 200


# --- health, readiness, drain, metrics ---------------------------------------


async def test_healthz_and_readyz(make_app: AppFactory) -> None:
    backend = FakeBackend()
    app, _ = make_app(backend)
    async with client_for(app) as c:
        assert (await c.get("/healthz")).status_code == 200
        r = await c.get("/readyz")
        assert r.status_code == 200
        assert r.json()["checks"] == {"engine": True, "registry": True, "accepting": True}
        backend.healthy = False
        r = await c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["checks"]["engine"] is False
        assert (await c.get("/healthz")).status_code == 200
        m = (await c.get("/metrics")).text
        assert metric_value(m, "inferscale_engine_up") == 0
        backend.healthy = True
        assert (await c.get("/readyz")).status_code == 200


async def test_drain(make_app: AppFactory) -> None:
    backend = FakeBackend(token_delay_s=0.05)
    app, gw = make_app(backend, drain_timeout_s=5)
    async with client_for(app) as c:
        inflight = asyncio.create_task(c.post("/v1/chat/completions", json=chat_body(stream=True)))
        while gw.admission.inflight == 0:
            await asyncio.sleep(0.005)
        drain = asyncio.create_task(gw.drain())
        await asyncio.sleep(0.01)
        r = await c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["checks"]["accepting"] is False
        rejected = await c.post("/v1/chat/completions", json=chat_body())
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "draining"
        assert not drain.done()  # still waiting for the in-flight stream
        done = await inflight
        assert done.status_code == 200
        assert stream_text(sse_chunks(done.text)) == EXPECTED_TEXT
        assert await asyncio.wait_for(drain, 2) is True


async def test_drain_timeout(make_app: AppFactory) -> None:
    app, gw = make_app(FakeBackend(first_token_delay_s=1.0), drain_timeout_s=0.05)
    async with client_for(app) as c:
        task = asyncio.create_task(c.post("/v1/chat/completions", json=chat_body()))
        while gw.admission.inflight == 0:
            await asyncio.sleep(0.005)
        assert await gw.drain() is False
        await task


async def test_metrics_exposition(make_app: AppFactory) -> None:
    app, _ = make_app()
    async with client_for(app) as c:
        await c.post("/v1/chat/completions", json=chat_body())
        await c.post("/v1/chat/completions", json=chat_body(stream=True))
        await c.post("/v1/chat/completions", json=chat_body(temperature=9))
        r = await c.get("/metrics")
    m = r.text
    assert r.headers["content-type"].startswith("text/plain")
    assert metric_value(m, "inferscale_requests_total", 'outcome="ok"') == 2
    assert metric_value(m, "inferscale_requests_total", 'outcome="client_error"') == 1
    assert metric_value(m, "inferscale_prompt_tokens_total") == 14
    assert metric_value(m, "inferscale_completion_tokens_total") == 12
    assert metric_value(m, "inferscale_request_duration_seconds_count") == 3
    assert metric_value(m, "inferscale_inflight") == 0
    assert metric_value(m, "inferscale_queue_depth") == 0
    labels = f'engine="llama.cpp",model_version="{MODEL_VERSION}",quant="Q4_K_M"'
    assert metric_value(m, "inferscale_model_info", labels) == 1
    assert 'le="0.05"' in m and 'le="60.0"' in m


async def test_request_log_line(make_app: AppFactory, capsys: pytest.CaptureFixture[str]) -> None:
    app, _ = make_app()
    async with client_for(app) as c:
        await c.post("/v1/chat/completions", json=chat_body(), headers={"X-Request-ID": "log-1"})
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if '"log-1"' in x]
    assert len(lines) == 1
    rec = lines[0]
    for key in ("model_version", "outcome", "status", "duration_ms", "prompt_tokens"):
        assert key in rec
    assert rec["outcome"] == "ok" and rec["completion_tokens"] == 6
    assert "disk at 97%" not in json.dumps(rec)  # no prompt content
    assert "Hello" not in json.dumps(rec)  # no response content


# --- tool calling ----------------------------------------------------------------

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CALLS = (
    {"name": "get_weather", "arguments": {"city": "Paris"}},
    {"name": "get_weather", "arguments": {"city": "Oslo"}},
)


def assemble_tool_calls(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: dict[int, dict[str, str]] = {}
    for c in chunks:
        for ch in c.get("choices") or []:
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                slot["name"] += (tc.get("function") or {}).get("name") or ""
                slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    return [
        {"name": calls[i]["name"], "arguments": json.loads(calls[i]["arguments"])}
        for i in sorted(calls)
    ]


async def test_tool_call_non_stream(
    make_app: AppFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = FakeBackend(tool_calls=CALLS)
    app, gw = make_app(backend)
    async with client_for(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json=chat_body(tools=[TOOL], parallel_tool_calls=True),
            headers={"X-Request-ID": "tool-1"},
        )
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    got = [
        (t["function"]["name"], json.loads(t["function"]["arguments"]))
        for t in choice["message"]["tool_calls"]
    ]
    assert got == [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Oslo"})]
    assert (
        backend.requests[0]["tools"] == [TOOL]
        and backend.requests[0]["parallel_tool_calls"] is True
    )
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_finish_reasons_total", 'reason="tool_calls"') == 1
    log = next(json.loads(x) for x in capsys.readouterr().out.splitlines() if '"tool-1"' in x)
    assert log["n_tools"] == 1 and log["finish_reason"] == "tool_calls"
    assert "Paris" not in json.dumps(log)  # arguments are content: never logged


async def test_tool_call_stream(make_app: AppFactory) -> None:
    app, gw = make_app(FakeBackend(tool_calls=CALLS))
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(stream=True, tools=[TOOL]))
    assert r.status_code == 200
    chunks = sse_chunks(r.text)
    assert sse_data(r.text)[-1] == "[DONE]"
    assert assemble_tool_calls(chunks) == [dict(c) for c in CALLS]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert gw.admission.inflight == 0


async def test_tool_choice_none_and_text_fallback(make_app: AppFactory) -> None:
    app, _ = make_app(FakeBackend(tool_calls=CALLS))
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=chat_body(tools=[TOOL], tool_choice="none"))
        plain = await c.post("/v1/chat/completions", json=chat_body())  # no tools -> text
    assert r.json()["choices"][0]["message"]["content"] == EXPECTED_TEXT
    assert plain.json()["choices"][0]["message"]["content"] == EXPECTED_TEXT


async def test_tool_result_round_trip(make_app: AppFactory) -> None:
    backend = FakeBackend()
    app, _ = make_app(backend)
    call = {
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
    }
    body = {
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_0", "content": '{"temp_c": 11}'},
        ],
        "tools": [TOOL],
        "max_tokens": 32,
    }
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = backend.requests[0]["messages"]
    assert sent[1]["tool_calls"] == [call] and sent[2]["tool_call_id"] == "call_0"


async def test_invalid_tools_rejected_400(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", json=chat_body(tool_choice="required"))
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


class FallbackFake(FakeBackend):
    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = await super().complete(payload)
        body["x_inferscale"] = {"tool_call_mode": "gateway_validated", "fallback": "engine"}
        return body


async def test_tool_call_fallback_metric_and_log(
    make_app: AppFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    app, gw = make_app(FallbackFake(tool_calls=CALLS))
    async with client_for(app) as c:
        await c.post(
            "/v1/chat/completions", json=chat_body(tools=[TOOL]), headers={"X-Request-ID": "fb-1"}
        )
    m = gw.metrics.render().decode()
    assert metric_value(m, "inferscale_tool_call_fallbacks_total", 'to="engine"') == 1
    log = next(json.loads(x) for x in capsys.readouterr().out.splitlines() if '"fb-1"' in x)
    assert log["tool_call_fallback"] == "engine"
