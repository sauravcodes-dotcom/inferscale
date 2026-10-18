"""OpenAIHTTPBackend error mapping, against a mocked engine (httpx.MockTransport)."""

from __future__ import annotations

import httpx
import pytest

from inferscale.backends.openai_http import OpenAIHTTPBackend
from inferscale.core.errors import (
    ContextLengthExceededError,
    EngineError,
    EngineRejectedError,
    EngineUnavailableError,
    GatewayError,
)

CTX = "request (6552 tokens) exceeds the available context size (4096 tokens), try increasing it"


def backend(status: int, body: dict[str, object]) -> OpenAIHTTPBackend:
    b = OpenAIHTTPBackend("http://engine")
    b._client = httpx.AsyncClient(
        base_url="http://engine",
        transport=httpx.MockTransport(lambda _r: httpx.Response(status, json=body)),
    )
    return b


@pytest.mark.parametrize(
    ("status", "body", "exc", "http", "code"),
    [
        (
            400,
            {"error": {"code": 400, "message": CTX, "type": "exceed_context_size_error"}},
            ContextLengthExceededError,
            400,
            "context_length_exceeded",
        ),
        (
            400,
            {"error": {"message": "Invalid tool schema"}},
            EngineRejectedError,
            400,
            "engine_rejected_request",
        ),
        (500, {"error": {"message": "boom"}}, EngineError, 502, "engine_error"),
        (
            503,
            {"error": {"message": "Loading model"}},
            EngineUnavailableError,
            503,
            "engine_unavailable",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_engine_status_mapping(
    status: int,
    body: dict[str, object],
    exc: type[GatewayError],
    http: int,
    code: str,
    stream: bool,
) -> None:
    b = backend(status, body)
    with pytest.raises(exc) as info:
        if stream:
            async for _ in b.stream({"messages": []}):
                pass
        else:
            await b.complete({"messages": []})
    assert info.value.status == http and info.value.code == code
    await b.aclose()


async def test_context_error_message_is_the_engines() -> None:
    b = backend(400, {"error": {"message": CTX}})
    with pytest.raises(ContextLengthExceededError, match="6552 tokens"):
        await b.complete({"messages": []})
    await b.aclose()
