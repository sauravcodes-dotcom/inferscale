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
        [{"role": "user", "content": "x", "tool_call_id": "c1"}],
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


JS = {
    "name": "triage_v1",
    "strict": True,
    "schema": {"type": "object", "properties": {"a": {"type": "string"}}},
}


def test_json_schema_passthrough() -> None:
    req = parse(response_format={"type": "json_schema", "json_schema": JS})
    payload = req.to_engine_payload(LimitsConfig())
    assert payload["response_format"] == {"type": "json_schema", "json_schema": JS}


@pytest.mark.parametrize(
    "rf",
    [
        {"type": "json_schema", "json_schema": {**JS, "schema": {"type": "array"}}},
        {"type": "json_schema", "json_schema": {"name": "x"}},
        {"type": "json_schema", "json_schema": {**JS, "name": "bad name!"}},
        {"type": "json_schema", "json_schema": {**JS, "extra": 1}},
        {"type": "text", "json_schema": JS},
        {
            "type": "json_schema",
            "json_schema": {**JS, "schema": {"type": "object", "x": "y" * 20000}},
        },
    ],
)
def test_json_schema_rejections(rf: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse(response_format=rf)


# --- tool calling ---------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather.get_current",  # BFCL-style dotted name
        "description": "Current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_tool_request_is_valid_and_passed_through() -> None:
    req = parse(tools=[WEATHER_TOOL], tool_choice="auto", parallel_tool_calls=True)
    payload = req.to_engine_payload(LimitsConfig())
    assert payload["tools"] == [WEATHER_TOOL]
    assert payload["tool_choice"] == "auto" and payload["parallel_tool_calls"] is True


def test_tool_round_trip_messages_are_valid() -> None:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "weather.get_current", "arguments": '{"city":"Oslo"}'},
    }
    msgs = [
        {"role": "user", "content": "weather in Oslo?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 3}'},
    ]
    req = ChatCompletionRequest.model_validate({"messages": msgs, "tools": [WEATHER_TOOL]})
    out = req.to_engine_payload(LimitsConfig())["messages"]
    assert out[1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Oslo"}'
    assert "content" not in out[1]  # null content is dropped, as the engine accepts
    assert out[2] == {"role": "tool", "content": '{"temp_c": 3}', "tool_call_id": "c1"}


@pytest.mark.parametrize(
    ("kw", "messages"),
    [
        ({"tool_choice": "required"}, MSG),  # tool_choice without tools
        ({"parallel_tool_calls": True}, MSG),
        ({"tools": []}, MSG),
        ({"tools": [{**WEATHER_TOOL, "type": "code"}]}, MSG),
        ({"tools": [{"type": "function", "function": {"name": "bad name"}}]}, MSG),
        (
            {
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {"type": "array"}}}
                ]
            },
            MSG,
        ),
        (
            {
                "tools": [WEATHER_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "other"}},
            },
            MSG,
        ),
        ({}, [{"role": "tool", "content": "x"}]),  # missing tool_call_id
        ({}, [{"role": "assistant"}]),  # neither content nor tool_calls
        ({}, [{"role": "user", "content": "x", "tool_calls": []}]),
        ({}, [{"role": "user"}]),
    ],
)
def test_tool_rejections(kw: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": messages, **kw})


def test_named_tool_choice_valid() -> None:
    parse(
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "weather.get_current"}},
    )


def test_tool_limits() -> None:
    limits = LimitsConfig(max_tools=2, max_tools_chars=400, max_input_chars=50)
    parse(tools=[WEATHER_TOOL]).check_limits(limits)
    with pytest.raises(InvalidRequestError, match="too many tools"):
        parse(tools=[WEATHER_TOOL] * 3).check_limits(limits)
    big = {**WEATHER_TOOL, "function": {**WEATHER_TOOL["function"], "description": "x" * 500}}
    with pytest.raises(InvalidRequestError, match="tool definitions too large"):
        parse(tools=[big]).check_limits(limits)
    call = {"type": "function", "function": {"name": "f", "arguments": "y" * 60}}
    with pytest.raises(
        InvalidRequestError, match="input too long"
    ):  # tool-call arguments count as input
        ChatCompletionRequest.model_validate(
            {"messages": [{"role": "assistant", "tool_calls": [call]}]}
        ).check_limits(limits)
