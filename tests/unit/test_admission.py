from __future__ import annotations

import asyncio

import pytest

from inferscale.core.admission import AdmissionController
from inferscale.core.errors import DrainingError, QueueFullError, QueueTimeoutError


async def test_admits_up_to_max_inflight_immediately() -> None:
    ac = AdmissionController(max_inflight=2, max_queue=0, queue_timeout_s=1)
    await ac.acquire()
    await ac.acquire()
    assert ac.inflight == 2
    with pytest.raises(QueueFullError) as exc:
        await ac.acquire()
    assert exc.value.status == 429
    assert exc.value.headers["Retry-After"] == "1"
    ac.release()
    ac.release()
    assert ac.inflight == 0


async def test_queued_request_runs_when_slot_frees() -> None:
    ac = AdmissionController(max_inflight=1, max_queue=2, queue_timeout_s=2)
    await ac.acquire()
    waiter = asyncio.create_task(ac.acquire())
    await asyncio.sleep(0.01)
    assert ac.queue_depth == 1
    assert not waiter.done()
    ac.release()
    await asyncio.wait_for(waiter, 1)
    assert ac.inflight == 1
    assert ac.queue_depth == 0


async def test_queue_bound_rejects_immediately() -> None:
    ac = AdmissionController(max_inflight=1, max_queue=1, queue_timeout_s=5)
    await ac.acquire()
    waiter = asyncio.create_task(ac.acquire())
    await asyncio.sleep(0.01)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(QueueFullError):
        await ac.acquire()
    assert loop.time() - t0 < 0.1
    waiter.cancel()


async def test_queue_timeout() -> None:
    ac = AdmissionController(max_inflight=1, max_queue=1, queue_timeout_s=0.05)
    await ac.acquire()
    with pytest.raises(QueueTimeoutError):
        await ac.acquire()
    assert ac.queue_depth == 0
    assert ac.inflight == 1


async def test_fifo_order() -> None:
    ac = AdmissionController(max_inflight=1, max_queue=3, queue_timeout_s=2)
    order: list[int] = []
    await ac.acquire()

    async def worker(i: int) -> None:
        await with_slot(ac, order, i)

    tasks = []
    for i in range(3):
        tasks.append(asyncio.create_task(worker(i)))
        await asyncio.sleep(0.01)
    ac.release()
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2]


async def with_slot(ac: AdmissionController, order: list[int], i: int) -> None:
    await ac.acquire()
    order.append(i)
    await asyncio.sleep(0.01)
    ac.release()


async def test_drain_rejects_and_waits_for_idle() -> None:
    ac = AdmissionController(max_inflight=1, max_queue=1, queue_timeout_s=1)
    await ac.acquire()
    ac.start_drain()
    with pytest.raises(DrainingError) as exc:
        await ac.acquire()
    assert exc.value.status == 503
    assert await ac.wait_idle(0.05) is False
    asyncio.get_running_loop().call_later(0.05, ac.release)
    assert await ac.wait_idle(1) is True
