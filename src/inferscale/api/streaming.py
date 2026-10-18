"""SSE helpers for the streaming chat-completions passthrough."""

from __future__ import annotations

import json
from typing import Any

from inferscale.backends.base import Json

SSE_DONE = b"data: [DONE]\n\n"


def sse_event(data: Json) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"


def extract_usage(chunk: Json) -> tuple[int, int] | None:
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def rewrite_chunk(chunk: Json, model_version: str, include_usage: bool) -> Json | None:
    """Prepare an engine chunk for the client. Returns None to drop it.

    The engine's model field (a local file path) is replaced by the model version.
    Usage is forwarded only when the client asked for it via stream_options.
    """
    out: dict[str, Any] = {**chunk, "model": model_version}
    if not include_usage and "usage" in out:
        if not out.get("choices"):
            return None
        out.pop("usage")
    return out
