"""Deterministic in-process backend for tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

from inferscale.backends.base import Json
from inferscale.core.errors import EngineError, EngineUnavailableError

FakeMode = Literal["ok", "error", "unavailable", "hang", "disconnect"]

DEFAULT_TOKENS = ("Hello", " from", " the", " fake", " backend", ".")


class FakeBackend:
    """Emits a fixed token sequence.

    Modes: ok; error (engine error before output); unavailable (engine unreachable);
    hang (never produces output); disconnect (engine drops mid-stream after 2 tokens).
    """

    def __init__(
        self,
        tokens: tuple[str, ...] = DEFAULT_TOKENS,
        token_delay_s: float = 0.0,
        first_token_delay_s: float = 0.0,
        mode: FakeMode = "ok",
        healthy: bool = True,
        prompt_tokens: int = 7,
    ) -> None:
        self.tokens = tokens
        self.token_delay_s = token_delay_s
        self.first_token_delay_s = first_token_delay_s
        self.mode: FakeMode = mode
        self.healthy = healthy
        self.prompt_tokens = prompt_tokens
        self.active_streams = 0
        self.streams_closed_early = 0
        self.requests: list[Json] = []

    def _usage(self, completion: int) -> Json:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": self.prompt_tokens + completion,
        }

    def _n_tokens(self, payload: Json) -> int:
        max_tokens = payload.get("max_tokens")
        n = len(self.tokens)
        return min(n, int(max_tokens)) if max_tokens else n

    async def _before_output(self) -> None:
        if self.mode == "unavailable":
            raise EngineUnavailableError("fake engine unreachable")
        if self.mode == "error":
            raise EngineError("fake engine error")
        if self.mode == "hang":
            await asyncio.sleep(3600)
        if self.first_token_delay_s:
            await asyncio.sleep(self.first_token_delay_s)

    async def health(self) -> bool:
        return self.healthy

    async def complete(self, payload: Json) -> Json:
        self.requests.append(payload)
        await self._before_output()
        n = self._n_tokens(payload)
        if self.token_delay_s:
            await asyncio.sleep(self.token_delay_s * n)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-engine-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(self.tokens[:n])},
                    "finish_reason": "stop" if n == len(self.tokens) else "length",
                }
            ],
            "usage": self._usage(n),
        }

    def _chunk(self, delta: Json, finish_reason: str | None = None) -> Json:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-engine-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    async def stream(self, payload: Json) -> AsyncIterator[Json]:
        self.requests.append(payload)
        self.active_streams += 1
        finished = False
        try:
            await self._before_output()
            n = self._n_tokens(payload)
            yield self._chunk({"role": "assistant", "content": ""})
            for i, tok in enumerate(self.tokens[:n]):
                if self.mode == "disconnect" and i == 2:
                    raise EngineError("fake engine dropped the stream")
                if i and self.token_delay_s:
                    await asyncio.sleep(self.token_delay_s)
                yield self._chunk({"content": tok})
            yield self._chunk({}, "stop" if n == len(self.tokens) else "length")
            include_usage = (payload.get("stream_options") or {}).get("include_usage")
            if include_usage:
                yield {**self._chunk({}), "choices": [], "usage": self._usage(n)}
            finished = True
        finally:
            self.active_streams -= 1
            if not finished:
                self.streams_closed_early += 1

    async def aclose(self) -> None:
        return None
