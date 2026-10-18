"""Deadline enforcement for engine calls (first-token and total budgets)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable

from inferscale.core.errors import DeadlineExceededError


class Deadline:
    def __init__(self, total_s: float) -> None:
        self.total_s = total_s
        self._end = time.monotonic() + total_s

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())


async def run_with_deadline[T](aw: Awaitable[T], deadline: Deadline) -> T:
    """Await `aw`; cancel it and raise DeadlineExceededError when the deadline passes."""
    try:
        async with asyncio.timeout(deadline.remaining()):
            return await aw
    except TimeoutError:
        raise DeadlineExceededError(f"engine did not respond within {deadline.total_s}s") from None


async def stream_with_deadlines[T](
    upstream: AsyncIterator[T],
    first_token_timeout_s: float,
    deadline: Deadline,
) -> AsyncGenerator[T]:
    """Yield from `upstream`, enforcing a first-item timeout and an overall deadline.

    Closing this generator (including on client disconnect) closes `upstream`,
    which releases the engine connection and frees its slot.
    """
    first = True
    try:
        while True:
            budget = deadline.remaining()
            if first:
                budget = min(budget, first_token_timeout_s)
            try:
                async with asyncio.timeout(budget):
                    item = await anext(upstream)
            except StopAsyncIteration:
                return
            except TimeoutError:
                what = "first token" if first else "stream completion"
                raise DeadlineExceededError(f"deadline exceeded waiting for {what}") from None
            first = False
            yield item
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()
