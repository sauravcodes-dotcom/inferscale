"""Application factory and the Gateway object that ties config, backend and admission together."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from inferscale.api.routes import router
from inferscale.backends.base import ChatBackend
from inferscale.backends.openai_http import OpenAIHTTPBackend
from inferscale.config import Settings, load_settings
from inferscale.core.admission import AdmissionController
from inferscale.core.errors import GatewayError
from inferscale.observability.logging import configure_logging, get_logger
from inferscale.observability.metrics import Metrics
from inferscale.registry import Registry, load_registry

REQUEST_ID_HEADER = "x-request-id"


class Gateway:
    def __init__(self, settings: Settings, registry: Registry, backend: ChatBackend) -> None:
        self.settings = settings
        self.registry = registry
        self.backend = backend
        a = settings.admission
        self.admission = AdmissionController(
            a.max_inflight, a.max_queue, a.queue_timeout_s, a.retry_after_s
        )
        self.metrics = Metrics()
        self.metrics.bind_admission(
            lambda: self.admission.inflight, lambda: self.admission.queue_depth
        )
        entry = registry.active_entry
        self.metrics.model_info.labels(
            model_version=registry.active, engine=entry.engine, quant=entry.quant
        ).set(1)
        self.log = get_logger("inferscale.request")
        self.drain_task: asyncio.Task[bool] | None = None

    @property
    def model_version(self) -> str:
        return self.registry.active

    async def check_engine(self) -> bool:
        ok = await self.backend.health()
        self.metrics.engine_up.set(1 if ok else 0)
        return ok

    async def drain(self) -> bool:
        """Stop admitting and wait for in-flight work. Returns False if the drain timed out."""
        self.admission.start_drain()
        get_logger("inferscale").info(
            "drain_started",
            inflight=self.admission.inflight,
            queued=self.admission.queue_depth,
            timeout_s=self.settings.drain_timeout_s,
        )
        drained = await self.admission.wait_idle(self.settings.drain_timeout_s)
        get_logger("inferscale").info(
            "drain_finished", clean=drained, inflight=self.admission.inflight
        )
        return drained


class RequestIdMiddleware:
    """Pure ASGI (does not buffer streams): honor or mint X-Request-ID and echo it back."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = ""
        for k, v in scope.get("headers", []):
            if k == REQUEST_ID_HEADER.encode():
                rid = v.decode("latin-1")[:128]
                break
        rid = rid or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k != b"x-request-id"]
                headers.append((REQUEST_ID_HEADER.encode(), rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_id)


async def _poll_engine(gw: Gateway) -> None:
    while True:
        with contextlib.suppress(Exception):
            await gw.check_engine()
        await asyncio.sleep(gw.settings.health_poll_interval_s)


def _install_sigterm_drain(gw: Gateway) -> None:
    """On SIGTERM: flip readyz to 503, drain in-flight work, then hand off to uvicorn's exit."""
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGTERM)

    def on_sigterm() -> None:
        if gw.admission.draining:
            return

        async def drain_then_exit() -> bool:
            drained = await gw.drain()
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, previous)
            if callable(previous):
                previous(signal.SIGTERM, None)
            else:
                signal.raise_signal(signal.SIGTERM)
            return drained

        gw.drain_task = loop.create_task(drain_then_exit())

    loop.add_signal_handler(signal.SIGTERM, on_sigterm)


def create_app(
    settings: Settings | None = None,
    backend: ChatBackend | None = None,
    registry: Registry | None = None,
    install_signal_handlers: bool = True,
) -> FastAPI:
    configure_logging()
    settings = settings or load_settings()
    registry = registry or load_registry(settings.registry_path)
    backend = backend or OpenAIHTTPBackend(
        settings.engine_url, connect_timeout_s=settings.deadlines.connect_timeout_s
    )
    gw = Gateway(settings, registry, backend)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        poller = asyncio.create_task(_poll_engine(gw))
        if install_signal_handlers:
            _install_sigterm_drain(gw)
        get_logger("inferscale").info(
            "startup", model_version=gw.model_version, engine_url=settings.engine_url
        )
        try:
            yield
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            await backend.aclose()

    app = FastAPI(title="InferScale", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gw
    app.include_router(router)
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.to_body(), status_code=exc.status, headers=exc.headers)

    return app


def app_factory() -> Any:
    """Entry point for `uvicorn --factory inferscale.api.app:app_factory`."""
    return create_app()
