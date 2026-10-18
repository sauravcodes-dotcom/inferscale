from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from inferscale.api.schemas import ChatCompletionRequest
from inferscale.config import LimitsConfig
from inferscale.core.errors import InvalidRequestError

MSG = [{"role": "user", "content": "hi"}]


def parse(**kw: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"messages": MSG, **kw})


def test_minimal_request_is_valid() -> None:
    req = parse()
    assert req.stream is False
    assert req.messages[0].content == "hi"


def test_full_request_is_valid() -> None:
    req = parse(
        model="qwen3-8b-q4km-v1",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        stream=True,
        stream_options={"include_usage": True},
        response_format={"type": "json_object"},
    )
    assert req.wants_usage_chunk


@pytest.mark.parametrize(
    "kw",
    [
        {"tools": []},  # unknown field
        {"temperature": 2.5},
        {"temperature": -0.1},
        {"top_p": 0},
        {"top_p": 1.5},
        {"max_tokens": 0},
        {"response_format": {"type": "xml"}},
        {"response_format": {"type": "json_schema"}},  # schema missing
        {"stream_options": {"include_usage": True, "x": 1}},
    ],
)
def test_rejections(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse(**kw)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "user"}],
        [{"role": "user", "content": "x", "name": "bob"}],
    ],
)
def test_bad_messages(messages: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": messages})


def test_limits() -> None:
    limits = LimitsConfig(max_messages=2, max_input_chars=10, max_tokens_cap=100)
    parse(max_tokens=100).check_limits(limits)
    with pytest.raises(InvalidRequestError, match="too many messages"):
        ChatCompletionRequest.model_validate({"messages": MSG * 3}).check_limits(limits)
    with pytest.raises(InvalidRequestError, match="input too long"):
        ChatCompletionRequest.model_validate(
            {"messages": [{"role": "user", "content": "x" * 11}]}
        ).check_limits(limits)
    with pytest.raises(InvalidRequestError, match="exceeds cap"):
        parse(max_tokens=101).check_limits(limits)


def test_engine_payload() -> None:
    limits = LimitsConfig(default_max_tokens=77)
    payload = parse(model="m", temperature=0).to_engine_payload(limits)
    assert payload == {"messages": MSG, "max_tokens": 77, "temperature": 0}
    streamed = parse(stream=True).to_engine_payload(limits)
    assert streamed["stream_options"] == {"include_usage": True}
    assert "stream" not in streamed
