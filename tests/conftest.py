from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from inferscale.api.app import Gateway, create_app
from inferscale.backends.base import ChatBackend
from inferscale.backends.fake import FakeBackend
from inferscale.config import (
    AdmissionConfig,
    DeadlineConfig,
    LimitsConfig,
    Settings,
)
from inferscale.registry import Registry, load_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "models" / "registry.yaml"
SERVING_PATH = ROOT / "configs" / "serving" / "mac-llamacpp.yaml"
MODEL_VERSION = "qwen3-8b-q4km-v1"


@pytest.fixture
def registry() -> Registry:
    return load_registry(REGISTRY_PATH)


def make_settings(
    admission: dict[str, Any] | None = None,
    deadlines: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    **kw: Any,
) -> Settings:
    return Settings(
        registry_path=str(REGISTRY_PATH),
        admission=AdmissionConfig(**(admission or {})),
        deadlines=DeadlineConfig(**(deadlines or {})),
        limits=LimitsConfig(**(limits or {})),
        **kw,
    )


AppFactory = Callable[..., tuple[FastAPI, Gateway]]


@pytest.fixture
def make_app(registry: Registry) -> AppFactory:
    def _make(backend: ChatBackend | None = None, **settings_kw: Any) -> tuple[FastAPI, Gateway]:
        app = create_app(
            settings=make_settings(**settings_kw),
            backend=backend or FakeBackend(),
            registry=registry,
            install_signal_handlers=False,
        )
        gw: Gateway = app.state.gateway
        return app, gw

    return _make


def client_for(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client(make_app: AppFactory) -> AsyncIterator[httpx.AsyncClient]:
    app, _ = make_app()
    async with client_for(app) as c:
        yield c


def chat_body(stream: bool = False, **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "disk at 97% on db-prod-01"}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": stream,
    }
    body.update(kw)
    return body
