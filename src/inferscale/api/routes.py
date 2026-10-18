"""HTTP routes: chat completions, health/readiness, metrics."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import ValidationError
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from inferscale.api.schemas import ChatCompletionRequest
from inferscale.api.streaming import SSE_DONE, extract_usage, rewrite_chunk, sse_event
from inferscale.backends.base import Json
from inferscale.core.deadlines import Deadline, run_with_deadline, stream_with_deadlines
from inferscale.core.errors import (
    EngineError,
    GatewayError,
    InvalidRequestError,
    ModelNotFoundError,
    RejectedError,
)

if TYPE_CHECKING:
    from inferscale.api.app import Gateway

router = APIRouter()


def _gateway(request: Request) -> Gateway:
    gw: Gateway = request.app.state.gateway
    return gw


class RequestRecord:
    """Accumulates per-request facts; `finish` emits metrics and the single log line."""

    def __init__(self, gw: Gateway, request_id: str, stream: bool) -> None:
        self.gw = gw
        self.request_id = request_id
        self.stream = stream
        self.start = time.monotonic()
        self.outcome = "ok"
        self.status = 200
        self.ttft_s: float | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.error: str | None = None
        self._done = False

    def fail(self, err: GatewayError) -> None:
        self.outcome = err.outcome
        self.status = err.status
        self.error = err.code

    def set_usage(self, usage: tuple[int, int] | None) -> None:
        if usage is not None:
            self.prompt_tokens, self.completion_tokens = usage

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        duration = time.monotonic() - self.start
        m = self.gw.metrics
        m.requests.labels(outcome=self.outcome).inc()
        m.request_duration.observe(duration)
        if self.ttft_s is not None:
            m.ttft.observe(self.ttft_s)
        if self.prompt_tokens:
            m.prompt_tokens.inc(self.prompt_tokens)
        if self.completion_tokens:
            m.completion_tokens.inc(self.completion_tokens)
        self.gw.log.info(
            "request",
            request_id=self.request_id,
            model_version=self.gw.model_version,
            stream=self.stream,
            outcome=self.outcome,
            status=self.status,
            error=self.error,
            ttft_ms=round(self.ttft_s * 1000, 1) if self.ttft_s is not None else None,
            duration_ms=round(duration * 1000, 1),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class ClosingStreamingResponse(StreamingResponse):
    """StreamingResponse that always runs `on_close`, even if the body never started."""

    def __init__(
        self, content: AsyncIterator[bytes], on_close: Callable[[], Awaitable[None]], **kw: Any
    ) -> None:
        super().__init__(content, **kw)
        self._body = content
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self._body, "aclose", None)
            try:
                if aclose is not None:
                    await aclose()
            finally:
                await self._on_close()


def _parse(body: Any, gw: Gateway) -> ChatCompletionRequest:
    try:
        req = ChatCompletionRequest.model_validate(body)
    except ValidationError as e:
        raise InvalidRequestError(_format_validation(e)) from None
    if req.model is not None and req.model != gw.model_version:
        raise ModelNotFoundError(f"model {req.model!r} not found; active is {gw.model_version!r}")
    req.check_limits(gw.settings.limits)
    return req


def _format_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"]) or "body"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    gw = _gateway(request)
    rec = RequestRecord(gw, request.state.request_id, stream=False)
    try:
        try:
            body = await request.json()
        except ValueError:
            raise InvalidRequestError("request body must be valid JSON") from None
        rec.stream = isinstance(body, dict) and bool(body.get("stream"))
        req = _parse(body, gw)
        try:
            await gw.admission.acquire()
        except RejectedError as e:
            gw.metrics.rejected.labels(reason=e.reason).inc()
            raise
    except GatewayError as e:
        rec.fail(e)
        rec.finish()
        raise

    payload = req.to_engine_payload(gw.settings.limits)
    deadline = Deadline(gw.settings.deadlines.total_timeout_s)
    if not req.stream:
        return await _complete(gw, rec, payload, deadline)
    return await _stream(gw, rec, req, payload, deadline)


async def _complete(gw: Gateway, rec: RequestRecord, payload: Json, deadline: Deadline) -> Response:
    try:
        result = await run_with_deadline(gw.backend.complete(payload), deadline)
        rec.set_usage(extract_usage(result))
    except GatewayError as e:
        rec.fail(e)
        raise
    except BaseException:
        rec.outcome, rec.status = "client_disconnect", 499
        raise
    finally:
        gw.admission.release()
        rec.finish()
    return JSONResponse({**result, "model": gw.model_version})


async def _stream(
    gw: Gateway,
    rec: RequestRecord,
    req: ChatCompletionRequest,
    payload: Json,
    deadline: Deadline,
) -> Response:
    events = stream_with_deadlines(
        gw.backend.stream(payload), gw.settings.deadlines.first_token_timeout_s, deadline
    )
    closed = False
    completed = False

    async def close() -> None:
        # Idempotent: close upstream (frees the engine slot), release admission, record.
        nonlocal closed
        if closed:
            return
        closed = True
        if not completed and rec.outcome == "ok":
            rec.outcome, rec.status = "client_disconnect", 499
        try:
            await events.aclose()
        finally:
            gw.admission.release()
            rec.finish()

    # Wait for the first chunk before committing to a 200 so early failures
    # (engine down, first-token timeout) surface as real HTTP statuses.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        err = EngineError("engine produced an empty stream")
        rec.fail(err)
        await close()
        raise err from None
    except GatewayError as e:
        rec.fail(e)
        await close()
        raise
    except BaseException:
        await close()
        raise
    rec.ttft_s = time.monotonic() - rec.start

    async def body() -> AsyncIterator[bytes]:
        nonlocal completed
        chunk: Json | None = first
        try:
            while chunk is not None:
                rec.set_usage(extract_usage(chunk))
                out = rewrite_chunk(chunk, gw.model_version, req.wants_usage_chunk)
                if out is not None:
                    yield sse_event(out)
                chunk = await anext(events, None)
            completed = True
            yield SSE_DONE
        except GatewayError as e:
            # Headers are already sent; report the failure in-band and end the stream.
            rec.fail(e)
            completed = True
            yield sse_event(e.to_body())
        finally:
            await close()

    return ClosingStreamingResponse(
        body(),
        on_close=close,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", response_model=None)
async def readyz(request: Request) -> JSONResponse:
    gw = _gateway(request)
    engine_ok = await gw.check_engine()
    checks = {
        "engine": engine_ok,
        "registry": gw.registry is not None,
        "accepting": not gw.admission.draining,
    }
    ready = all(checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    return Response(_gateway(request).metrics.render(), media_type=CONTENT_TYPE_LATEST)
