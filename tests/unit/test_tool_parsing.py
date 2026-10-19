"""Gateway-side tool-call parsing and the tool_call_mode=gateway backend path."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from inferscale.backends.openai_http import OpenAIHTTPBackend
from inferscale.backends.tool_parsing import StreamParser, parse_output
from inferscale.core.errors import ContextLengthExceededError

ONE = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo", "unit": "c"}}\n</tool_call>'
)
TWO = ONE + '\n<tool_call>\n{"name": "get_stock", "arguments": {"ticker": "AAPL"}}\n</tool_call>'


def names_args(calls: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def test_parse_single_and_parallel() -> None:
    p = parse_output(ONE)
    assert p.content is None and p.malformed == 0
    assert names_args(p.tool_calls) == [("get_weather", {"city": "Oslo", "unit": "c"})]
    assert p.tool_calls[0]["id"].startswith("call_") and p.tool_calls[0]["type"] == "function"
    assert [n for n, _ in names_args(parse_output(TWO).tool_calls)] == ["get_weather", "get_stock"]


def test_argument_order_is_preserved() -> None:
    p = parse_output('<tool_call>{"name": "f", "arguments": {"b": 1, "a": 2}}</tool_call>')
    assert p.tool_calls[0]["function"]["arguments"] == '{"b": 1, "a": 2}'


def test_text_only_and_mixed() -> None:
    assert parse_output("Hello there.").content == "Hello there."
    assert parse_output("Hello there.").tool_calls == []
    p = parse_output("Let me check.\n" + ONE)
    assert p.content == "Let me check." and len(p.tool_calls) == 1


@pytest.mark.parametrize(
    "block",
    [
        "<tool_call>\n{'name': 'f', 'arguments': {}}\n</tool_call>",  # python dict, not JSON
        '<tool_call>\n{"arguments": {}}\n</tool_call>',  # no name
        '<tool_call>\n{"name": "f", "arguments": [1]}\n</tool_call>',  # arguments not an object
        "<tool_call>\nnot json\n</tool_call>",
    ],
)
def test_malformed_blocks_are_kept_not_dropped(block: str) -> None:
    p = parse_output(block)
    assert p.tool_calls == [] and p.malformed == 1
    assert p.content == block.strip()


def test_string_arguments_and_unterminated_block() -> None:
    p = parse_output('<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}')  # no closing tag
    assert names_args(p.tool_calls) == [("f", {"x": 1})]


def test_think_blocks_are_removed() -> None:
    p = parse_output("<think>\nreasoning\n</think>\n\n" + ONE)
    assert p.content is None and len(p.tool_calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        ONE,
        TWO,
        "Sure! " + ONE,
        "plain answer with < and > signs",
        "<think>x</think>" + TWO,
        "<tool_call>not json</tool_call> ok",
    ],
)
def test_stream_parser_matches_batch_for_every_split(text: str) -> None:
    batch = parse_output(text)
    for size in (1, 2, 3, 7, len(text)):
        sp = StreamParser()
        out_text, calls = "", []
        for i in range(0, len(text), size):
            t, c = sp.feed(text[i : i + size])
            out_text += t
            calls += c
        t, c = sp.finish()
        out_text += t
        calls += c
        assert names_args(calls) == names_args(batch.tool_calls), size
        assert (out_text.strip() or None) == batch.content, size
        assert "<tool_ca" not in out_text or batch.malformed  # no partial tags leak


# --- backend gateway mode (mock engine) -----------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "get_stock", "parameters": {"type": "object"}}},
]


class MockEngine:
    def __init__(self, text: str, stop_type: str = "eos", status: int = 200) -> None:
        self.text, self.stop_type, self.status = text, stop_type, status
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        self.requests.append((req.url.path, body))
        if req.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": f"<rendered {len(body['tools'])} tools>"})
        if self.status != 200:
            return httpx.Response(
                self.status,
                json={
                    "error": {
                        "message": "request (9000 tokens) exceeds the "
                        "available context size (8192 tokens)"
                    }
                },
            )
        final = {
            "content": "",
            "stop": True,
            "stop_type": self.stop_type,
            "tokens_evaluated": 120,
            "tokens_predicted": 21,
        }
        if not body.get("stream"):
            return httpx.Response(200, json={**final, "content": self.text})
        pieces = [self.text[i : i + 5] for i in range(0, len(self.text), 5)]
        lines = [f"data: {json.dumps({'content': p, 'stop': False})}\n\n" for p in pieces]
        lines.append(f"data: {json.dumps(final)}\n\n")
        return httpx.Response(
            200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
        )


def backend(engine: MockEngine, mode: str = "gateway") -> OpenAIHTTPBackend:
    b = OpenAIHTTPBackend("http://engine", tool_call_mode=mode)  # type: ignore[arg-type]
    b._client = httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(engine))
    return b


async def test_gateway_mode_non_stream() -> None:
    engine = MockEngine(TWO)
    b = backend(engine)
    out = await b.complete(
        {
            "messages": [{"role": "user", "content": "q"}],
            "tools": TOOLS,
            "max_tokens": 64,
            "temperature": 0,
        }
    )
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    assert [n for n, _ in names_args(choice["message"]["tool_calls"])] == [
        "get_weather",
        "get_stock",
    ]
    assert out["usage"] == {"prompt_tokens": 120, "completion_tokens": 21, "total_tokens": 141}
    paths = [p for p, _ in engine.requests]
    assert paths == ["/apply-template", "/completion"]
    comp = engine.requests[1][1]
    assert comp["n_predict"] == 64 and comp["temperature"] == 0 and comp["cache_prompt"] is True
    assert "grammar" not in comp and "tools" not in comp  # unconstrained generation
    await b.aclose()


async def test_gateway_mode_stream_chunks() -> None:
    engine = MockEngine("Checking. " + ONE)
    b = backend(engine)
    chunks = [
        c
        async for c in b.stream(
            {"messages": [], "tools": TOOLS, "stream_options": {"include_usage": True}}
        )
    ]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    text = "".join((c["choices"][0]["delta"].get("content") or "") for c in chunks if c["choices"])
    assert text.strip() == "Checking."
    calls = [
        tc
        for c in chunks
        if c["choices"]
        for tc in c["choices"][0]["delta"].get("tool_calls") or []
    ]
    assert (
        len(calls) == 1 and calls[0]["index"] == 0 and calls[0]["function"]["name"] == "get_weather"
    )
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 21
    await b.aclose()


async def test_gateway_mode_text_length_and_named_choice() -> None:
    engine = MockEngine("just text", stop_type="limit")
    b = backend(engine)
    out = await b.complete(
        {
            "messages": [],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "get_stock"}},
        }
    )
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "just text"}
    assert engine.requests[0][1]["tools"] == [TOOLS[1]]  # only the named tool is rendered
    await b.aclose()


async def test_engine_mode_and_tool_choice_none_bypass_gateway_path() -> None:
    for mode, payload in (
        ("engine", {"tools": TOOLS}),
        ("gateway", {"tools": TOOLS, "tool_choice": "none"}),
        ("gateway", {}),
    ):
        engine = MockEngine("x")
        b = backend(engine, mode)
        try:
            await b.complete({"messages": [], **payload})
        except Exception:
            pass
        assert [p for p, _ in engine.requests] == ["/v1/chat/completions"], (mode, payload)
        await b.aclose()


async def test_gateway_mode_context_error_mapping() -> None:
    b = backend(MockEngine("x", status=400))
    with pytest.raises(ContextLengthExceededError):
        await b.complete({"messages": [], "tools": TOOLS})
    await b.aclose()


# --- schema validation and gateway_validated fallback --------------------------------

from inferscale.backends.tool_parsing import validate_calls  # noqa: E402

SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "f",
        "parameters": {
            "type": "object",
            "required": ["s"],
            "properties": {
                "s": {"type": "string"},
                "n": {"type": "integer"},
                "e": {"type": "string", "enum": ["a", "b"]},
            },
        },
    },
}


def call(name: str, args: Any) -> dict[str, Any]:
    return {
        "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}
    }


@pytest.mark.parametrize(
    ("c", "problem"),
    [
        (call("f", {"s": "x", "n": 1, "e": "a"}), None),
        (call("f", {"s": ["x"]}), "is not of type 'string'"),
        (call("f", {"n": 1}), "'s' is a required property"),
        (call("f", {"s": "x", "e": "z"}), "is not one of"),
        (call("g", {"s": "x"}), "unknown function"),
        (call("f", "{not json"), "not JSON"),
    ],
)
def test_validate_calls(c: dict[str, Any], problem: str | None) -> None:
    problems = validate_calls([c], [SCHEMA_TOOL])
    assert (problems == []) if problem is None else any(problem in p for p in problems)


def test_invalid_schema_is_not_checked() -> None:
    bad = {
        "type": "function",
        "function": {
            "name": "f",
            "parameters": {"type": "object", "properties": {"x": {"type": "dict"}}},
        },
    }
    assert validate_calls([call("f", {"x": 1})], [bad]) == []


class FallbackEngine(MockEngine):
    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/chat/completions":
            self.requests.append((req.url.path, json.loads(req.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c",
                                        "type": "function",
                                        "function": {"name": "f", "arguments": '{"s": "x"}'},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return super().__call__(req)


async def test_validated_mode_falls_back_on_schema_violation() -> None:
    engine = FallbackEngine(
        '<tool_call>{"name": "f", "arguments": {"s": ["not", "a", "string"]}}</tool_call>'
    )
    b = backend(engine, "gateway_validated")
    out = await b.complete({"messages": [], "tools": [SCHEMA_TOOL]})
    assert [p for p, _ in engine.requests] == [
        "/apply-template",
        "/completion",
        "/v1/chat/completions",
    ]
    assert out["x_inferscale"]["fallback"] == "engine"
    assert "is not of type 'string'" in out["x_inferscale"]["fallback_reasons"][0]
    assert json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "s": "x"
    }
    await b.aclose()


async def test_validated_mode_keeps_valid_unconstrained_output_and_streams_it() -> None:
    engine = FallbackEngine('<tool_call>{"name": "f", "arguments": {"n": 2, "s": "y"}}</tool_call>')
    b = backend(engine, "gateway_validated")
    chunks = [
        c
        async for c in b.stream(
            {"messages": [], "tools": [SCHEMA_TOOL], "stream_options": {"include_usage": True}}
        )
    ]
    assert "/v1/chat/completions" not in [p for p, _ in engine.requests]  # no fallback
    calls = [
        tc
        for c in chunks
        if c["choices"]
        for tc in c["choices"][0]["delta"].get("tool_calls") or []
    ]
    assert calls[0]["function"]["arguments"] == '{"n": 2, "s": "y"}'  # model's own order kept
    finish = next(c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"])
    assert finish["x_inferscale"]["fallback"] is None
    assert chunks[-1]["usage"]["completion_tokens"] == 21
    await b.aclose()


async def test_validated_mode_falls_back_on_malformed_block() -> None:
    engine = FallbackEngine("<tool_call>{'name': 'f'}</tool_call>")
    b = backend(engine, "gateway_validated")
    out = await b.complete({"messages": [], "tools": [SCHEMA_TOOL]})
    assert out["x_inferscale"]["fallback"] == "engine"
    assert "malformed" in out["x_inferscale"]["fallback_reasons"][-1]
    await b.aclose()
