"""Backend for OpenAI-compatible HTTP engines (llama-server)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from inferscale.backends.base import Json
from inferscale.core.errors import (
    ContextLengthExceededError,
    EngineError,
    EngineRejectedError,
    EngineUnavailableError,
)

CHAT_PATH = "/v1/chat/completions"


def _engine_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else None
        return str(msg or body)[:500]
    except (ValueError, AttributeError):
        return resp.text[:500]


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code == 200:
        return
    detail = _engine_message(resp)
    msg = f"engine returned {resp.status_code}: {detail}"
    if resp.status_code == 503:  # llama-server: model still loading
        raise EngineUnavailableError(msg)
    if 400 <= resp.status_code < 500:
        # The request itself was refused (bad input, prompt too long): report a client
        # error rather than blaming the engine.
        if "context size" in detail or "context length" in detail:
            raise ContextLengthExceededError(detail)
        raise EngineRejectedError(msg)
    raise EngineError(msg)


class OpenAIHTTPBackend:
    def __init__(self, base_url: str, connect_timeout_s: float = 2.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            # Read timing is enforced by gateway deadlines, not by httpx.
            timeout=httpx.Timeout(None, connect=connect_timeout_s),
        )

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=1.0)
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        try:
            return bool(resp.json().get("status") == "ok")
        except ValueError:
            return False

    async def complete(self, payload: Json) -> Json:
        try:
            resp = await self._client.post(CHAT_PATH, json={**payload, "stream": False})
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise EngineUnavailableError(f"engine unreachable: {e}") from e
        except httpx.HTTPError as e:
            raise EngineError(f"engine transport error: {e!r}") from e
        _raise_for_status(resp)
        try:
            body: Json = resp.json()
        except ValueError as e:
            raise EngineError("engine returned invalid JSON") from e
        return body

    async def stream(self, payload: Json) -> AsyncIterator[Json]:
        body = {**payload, "stream": True}
        try:
            async with self._client.stream("POST", CHAT_PATH, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    _raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk: Json = json.loads(data)
                    except ValueError as e:
                        raise EngineError("engine sent an invalid SSE chunk") from e
                    if "error" in chunk:
                        raise EngineError(f"engine stream error: {chunk['error']}")
                    yield chunk
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise EngineUnavailableError(f"engine unreachable: {e}") from e
        except httpx.HTTPError as e:
            raise EngineError(f"engine stream broke: {e!r}") from e
        raise EngineError("engine stream ended without [DONE]")

    async def aclose(self) -> None:
        await self._client.aclose()
