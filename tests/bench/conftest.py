from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
import uvicorn

from inferscale.backends.fake import FakeBackend
from tests.conftest import AppFactory

LiveFactory = Callable[..., Awaitable[str]]


@pytest.fixture
async def live_gateway(make_app: AppFactory) -> AsyncIterator[LiveFactory]:
    """Start the gateway (FakeBackend) under a real uvicorn server on a random port."""
    servers: list[tuple[uvicorn.Server, asyncio.Task[None]]] = []

    async def start(backend: FakeBackend, **settings: Any) -> str:
        app, _ = make_app(backend, **settings)
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
        )
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        servers.append((server, task))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, task in servers:
        server.should_exit = True
        await task
