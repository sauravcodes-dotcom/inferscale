"""Admission control: bounded concurrency plus a bounded, time-limited wait queue."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from inferscale.core.errors import DrainingError, QueueFullError, QueueTimeoutError


class AdmissionController:
    def __init__(
        self,
        max_inflight: int,
        max_queue: int,
        queue_timeout_s: float,
        retry_after_s: int = 1,
    ) -> None:
        self.max_inflight = max_inflight
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.retry_after_s = retry_after_s
        self._sem = asyncio.Semaphore(max_inflight)
        self._inflight = 0
        self._waiting = 0
        self._draining = False
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def queue_depth(self) -> int:
        return self._waiting

    @property
    def draining(self) -> bool:
        return self._draining

    def _retry_headers(self) -> dict[str, str]:
        return {"Retry-After": str(self.retry_after_s)}

    def _update_idle(self) -> None:
        if self._inflight == 0 and self._waiting == 0:
            self._idle.set()
        else:
            self._idle.clear()

    async def acquire(self) -> None:
        """Take an execution slot, waiting in the queue if needed. Raises a RejectedError."""
        if self._draining:
            raise DrainingError("server is draining", headers=self._retry_headers())
        if self._sem.locked() or self._waiting > 0:
            if self._waiting >= self.max_queue:
                raise QueueFullError(
                    f"queue full ({self._inflight} in flight, {self._waiting} waiting)",
                    headers=self._retry_headers(),
                )
            self._waiting += 1
            self._update_idle()
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout_s)
            except TimeoutError:
                raise QueueTimeoutError(
                    f"timed out after {self.queue_timeout_s}s waiting for a slot",
                    headers=self._retry_headers(),
                ) from None
            finally:
                self._waiting -= 1
                self._update_idle()
        else:
            await self._sem.acquire()
        self._inflight += 1
        self._update_idle()

    def release(self) -> None:
        self._inflight -= 1
        self._sem.release()
        self._update_idle()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def start_drain(self) -> None:
        self._draining = True

    async def wait_idle(self, timeout_s: float) -> bool:
        """Wait until nothing is in flight or queued. Returns False on timeout."""
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_s)
        except TimeoutError:
            return False
        return True
