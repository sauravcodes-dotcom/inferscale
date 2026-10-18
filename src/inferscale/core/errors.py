"""Gateway error types, each carrying its HTTP status and metrics outcome."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    status: int = 500
    error_type: str = "server_error"
    code: str = "internal_error"
    outcome: str = "error"

    def __init__(self, message: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.headers = headers or {}

    def to_body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.error_type, "code": self.code}}


class InvalidRequestError(GatewayError):
    status = 400
    error_type = "invalid_request_error"
    code = "invalid_request"
    outcome = "client_error"


class ContextLengthExceededError(InvalidRequestError):
    """The prompt (messages + tool definitions) does not fit the engine's per-slot context."""

    code = "context_length_exceeded"


class EngineRejectedError(InvalidRequestError):
    """The engine refused the request (4xx): a client-side problem, not an engine fault."""

    code = "engine_rejected_request"


class ModelNotFoundError(InvalidRequestError):
    status = 404
    code = "model_not_found"


class RejectedError(GatewayError):
    """Admission rejected the request; `reason` labels inferscale_rejected_total."""

    reason: str = "rejected"
    outcome = "rejected"


class QueueFullError(RejectedError):
    status = 429
    error_type = "rate_limit_error"
    code = "queue_full"
    reason = "queue_full"


class QueueTimeoutError(RejectedError):
    status = 429
    error_type = "rate_limit_error"
    code = "queue_timeout"
    reason = "queue_timeout"


class DrainingError(RejectedError):
    status = 503
    error_type = "server_error"
    code = "draining"
    reason = "draining"


class DeadlineExceededError(GatewayError):
    status = 504
    error_type = "timeout_error"
    code = "deadline_exceeded"
    outcome = "timeout"


class EngineError(GatewayError):
    status = 502
    error_type = "server_error"
    code = "engine_error"
    outcome = "engine_error"


class EngineUnavailableError(GatewayError):
    status = 503
    error_type = "server_error"
    code = "engine_unavailable"
    outcome = "engine_unavailable"
