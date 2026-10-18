"""Backend protocol: what the gateway needs from an inference engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

Json = dict[str, Any]


@runtime_checkable
class ChatBackend(Protocol):
    async def health(self) -> bool:
        """True when the engine is loaded and ready to serve."""
        ...

    async def complete(self, payload: Json) -> Json:
        """Non-streaming chat completion. Returns the engine's JSON body.

        Raises EngineUnavailableError when unreachable, EngineError on engine failure.
        """
        ...

    def stream(self, payload: Json) -> AsyncIterator[Json]:
        """Streaming chat completion: yields parsed SSE chunks until [DONE].

        Closing the iterator must release the upstream connection.
        """
        ...

    async def aclose(self) -> None: ...
