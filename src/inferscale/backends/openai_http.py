"""Backend for OpenAI-compatible HTTP engines (llama-server)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from inferscale.backends.base import Json
from inferscale.backends.tool_parsing import StreamParser, parse_output, validate_calls
from inferscale.core.errors import (
    ContextLengthExceededError,
    EngineError,
    EngineRejectedError,
    EngineUnavailableError,
)

CHAT_PATH = "/v1/chat/completions"
TEMPLATE_PATH = "/apply-template"
COMPLETION_PATH = "/completion"
ToolCallMode = Literal["engine", "gateway", "gateway_validated"]


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
    """llama-server backend.

    tool_call_mode="engine": requests with tools go to the engine's chat endpoint, which
    parses tool calls itself and constrains decoding with a schema grammar.
    tool_call_mode="gateway": requests with tools are rendered with the model's chat template
    (/apply-template), generated without the tool grammar (/completion) and parsed by
    inferscale.backends.tool_parsing. Requests without tools are unaffected by the mode.
    tool_call_mode="gateway_validated": as "gateway", but every parsed call is validated against
    its tool's JSON schema; if any call fails (unknown function, wrong type, missing required
    parameter, value outside an enum) or a tool-call block is malformed, the request is served
    again by the engine's schema-constrained path. Streaming responses in this mode are produced
    after generation completes (tool-call deltas were whole calls anyway).
    """

    def __init__(
        self, base_url: str, connect_timeout_s: float = 2.0, tool_call_mode: ToolCallMode = "engine"
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            # Read timing is enforced by gateway deadlines, not by httpx.
            timeout=httpx.Timeout(None, connect=connect_timeout_s),
        )
        self.tool_call_mode = tool_call_mode

    def _gateway_tools(self, payload: Json) -> bool:
        return (
            self.tool_call_mode in ("gateway", "gateway_validated")
            and bool(payload.get("tools"))
            and payload.get("tool_choice") != "none"
            and "response_format" not in payload
        )

    async def _post(self, path: str, body: Json) -> Json:
        try:
            resp = await self._client.post(path, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise EngineUnavailableError(f"engine unreachable: {e}") from e
        except httpx.HTTPError as e:
            raise EngineError(f"engine transport error: {e!r}") from e
        _raise_for_status(resp)
        try:
            out: Json = resp.json()
        except ValueError as e:
            raise EngineError("engine returned invalid JSON") from e
        return out

    async def _render(self, payload: Json) -> str:
        tools = payload["tools"]
        choice = payload.get("tool_choice")
        if isinstance(choice, dict):  # named tool: offer only that tool
            tools = [t for t in tools if t["function"]["name"] == choice["function"]["name"]]
        rendered = await self._post(
            TEMPLATE_PATH, {"messages": payload["messages"], "tools": tools}
        )
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise EngineError("engine /apply-template returned no prompt")
        return prompt

    @staticmethod
    def _completion_body(payload: Json, prompt: str, stream: bool) -> Json:
        body: Json = {
            "prompt": prompt,
            "stream": stream,
            "cache_prompt": True,
            "n_predict": payload.get("max_tokens", -1),
        }
        for key in ("temperature", "top_p"):
            if key in payload:
                body[key] = payload[key]
        return body

    @staticmethod
    def _usage(d: Json) -> Json:
        p, c = int(d.get("tokens_evaluated") or 0), int(d.get("tokens_predicted") or 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    @staticmethod
    def _finish_reason(d: Json, has_calls: bool) -> str:
        if has_calls:
            return "tool_calls"
        return "length" if d.get("stop_type") == "limit" else "stop"

    async def _complete_tools(self, payload: Json) -> Json:
        prompt = await self._render(payload)
        d = await self._post(COMPLETION_PATH, self._completion_body(payload, prompt, stream=False))
        parsed = parse_output(str(d.get("content") or ""))
        if self.tool_call_mode == "gateway_validated":
            problems = validate_calls(parsed.tool_calls, payload["tools"])
            if parsed.malformed:
                problems.append(f"{parsed.malformed} malformed tool-call block(s)")
            if problems:
                body = await self._complete_engine(payload)
                body["x_inferscale"] = {
                    "tool_call_mode": "gateway_validated",
                    "fallback": "engine",
                    "fallback_reasons": problems[:5],
                }
                return body
        message: Json = {"role": "assistant", "content": parsed.content}
        if parsed.tool_calls:
            message["tool_calls"] = parsed.tool_calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(d.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self._finish_reason(d, bool(parsed.tool_calls)),
                }
            ],
            "usage": self._usage(d),
            "x_inferscale": {
                "tool_call_mode": self.tool_call_mode,
                "fallback": None,
                "malformed_tool_calls": parsed.malformed,
            },
        }

    async def _stream_validated(self, payload: Json) -> AsyncIterator[Json]:
        """Validated mode streams a completed (possibly fallback) response as chunks."""
        body = await self._complete_tools(payload)
        choice = body["choices"][0]
        msg = choice["message"]
        base = {
            "id": body.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion.chunk",
            "created": body.get("created", int(time.time())),
            "model": body.get("model", ""),
        }

        def chunk(delta: Json, finish: str | None = None, **extra: Any) -> Json:
            return {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }

        yield chunk({"role": "assistant", "content": None})
        if msg.get("content"):
            yield chunk({"content": msg["content"]})
        for i, call in enumerate(msg.get("tool_calls") or []):
            yield chunk({"tool_calls": [{"index": i, **call}]})
        yield chunk({}, choice.get("finish_reason"), x_inferscale=body.get("x_inferscale"))
        if (payload.get("stream_options") or {}).get("include_usage"):
            yield {**base, "choices": [], "usage": body.get("usage")}

    async def _stream_tools(self, payload: Json) -> AsyncIterator[Json]:
        prompt = await self._render(payload)
        body = self._completion_body(payload, prompt, stream=True)
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        parser = StreamParser()
        n_calls = 0

        def chunk(delta: Json, finish: str | None = None, **extra: Any) -> Json:
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra,
            }

        def call_chunks(calls: list[Json]) -> list[Json]:
            nonlocal n_calls
            out = []
            for call in calls:
                out.append(chunk({"tool_calls": [{"index": n_calls, **call}]}))
                n_calls += 1
            return out

        started = False
        try:
            async with self._client.stream("POST", COMPLETION_PATH, json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    _raise_for_status(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        d: Json = json.loads(line[5:].strip())
                    except ValueError as e:
                        raise EngineError("engine sent an invalid SSE chunk") from e
                    if "error" in d:
                        raise EngineError(f"engine stream error: {d['error']}")
                    if not started:
                        started = True
                        yield chunk({"role": "assistant", "content": None})
                    text, calls = parser.feed(str(d.get("content") or ""))
                    if d.get("stop"):
                        tail_text, tail_calls = parser.finish()
                        text, calls = text + tail_text, calls + tail_calls
                    if text and (text.strip() or parser.emitted_text):
                        yield chunk({"content": text})
                    for c in call_chunks(calls):
                        yield c
                    if d.get("stop"):
                        yield chunk(
                            {},
                            self._finish_reason(d, n_calls > 0),
                            x_inferscale={
                                "tool_call_mode": "gateway",
                                "malformed_tool_calls": parser.malformed,
                            },
                        )
                        if (payload.get("stream_options") or {}).get("include_usage"):
                            yield {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": "",
                                "choices": [],
                                "usage": self._usage(d),
                            }
                        return
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise EngineUnavailableError(f"engine unreachable: {e}") from e
        except httpx.HTTPError as e:
            raise EngineError(f"engine stream broke: {e!r}") from e
        raise EngineError("engine stream ended without a stop chunk")

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
        if self._gateway_tools(payload):
            return await self._complete_tools(payload)
        return await self._complete_engine(payload)

    async def _complete_engine(self, payload: Json) -> Json:
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
        if self._gateway_tools(payload):
            gen = (
                self._stream_validated(payload)
                if self.tool_call_mode == "gateway_validated"
                else self._stream_tools(payload)
            )
            async for c in gen:
                yield c
            return
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
