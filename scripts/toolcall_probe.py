"""Feasibility probe: native tool calling (10 hand-written BFCL-style cases, not BFCL data).

Usage: uv run python scripts/toolcall_probe.py [BASE_URL]   (default: engine on :8081; use
http://127.0.0.1:8000 to go through the gateway). Re-run after any engine or template change.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"


def fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


WEATHER = fn(
    "get_weather",
    "Get the current weather for a city.",
    {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
    },
    ["city"],
)
MORTGAGE = fn(
    "calculate_mortgage",
    "Monthly payment for a fixed-rate mortgage.",
    {
        "principal": {"type": "number"},
        "annual_rate_pct": {"type": "number"},
        "years": {"type": "integer"},
    },
    ["principal", "annual_rate_pct", "years"],
)
STOCK = fn(
    "get_stock_price",
    "Latest price for a stock ticker.",
    {"ticker": {"type": "string"}},
    ["ticker"],
)
TRANSLATE = fn(
    "translate_text",
    "Translate text into a target language.",
    {
        "text": {"type": "string"},
        "target_language": {"type": "string", "description": "ISO 639-1 code"},
    },
    ["text", "target_language"],
)
EVENT = fn(
    "create_event",
    "Create a calendar event.",
    {
        "title": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "start": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM 24h"},
            },
            "required": ["date", "time"],
        },
    },
    ["title", "attendees", "start"],
)
RESTART = fn(
    "restart_service",
    "Restart a service on a host.",
    {"service": {"type": "string"}, "host": {"type": "string"}, "force": {"type": "boolean"}},
    ["service", "host", "force"],
)

# (name, user message, tools, expected [(fn_name, required arg subset)] or [] for no call, stream)
CASES: list[tuple[str, str, list[dict[str, Any]], list[tuple[str, dict[str, Any]]], bool]] = [
    (
        "simple",
        "What's the weather in Paris right now, in celsius?",
        [WEATHER],
        [("get_weather", {"city": "Paris", "unit": "celsius"})],
        False,
    ),
    (
        "typed_numbers",
        "What is the monthly payment on a $350,000 loan at 6.5% for 30 years?",
        [MORTGAGE],
        [("calculate_mortgage", {"principal": 350000, "annual_rate_pct": 6.5, "years": 30})],
        False,
    ),
    (
        "multiple_choose",
        "How much is one share of NVDA trading for?",
        [WEATHER, STOCK, TRANSLATE],
        [("get_stock_price", {"ticker": "NVDA"})],
        False,
    ),
    (
        "parallel_same",
        "Give me the weather in Tokyo and in Berlin.",
        [WEATHER],
        [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Berlin"})],
        False,
    ),
    (
        "parallel_multiple",
        "Translate 'good morning' into French, and also tell me the AAPL stock price.",
        [WEATHER, STOCK, TRANSLATE],
        [
            ("translate_text", {"text": "good morning", "target_language": "fr"}),
            ("get_stock_price", {"ticker": "AAPL"}),
        ],
        False,
    ),
    ("irrelevance", "Write a two-line poem about autumn.", [WEATHER, STOCK], [], False),
    (
        "nested_array",
        "Schedule 'Incident review' on 2026-11-03 at 14:30 with "
        "alice@example.com and bob@example.com.",
        [EVENT],
        [
            (
                "create_event",
                {
                    "title": "Incident review",
                    "attendees": ["alice@example.com", "bob@example.com"],
                    "start": {"date": "2026-11-03", "time": "14:30"},
                },
            )
        ],
        False,
    ),
    (
        "boolean",
        "Force-restart nginx on web-01.",
        [RESTART],
        [("restart_service", {"service": "nginx", "host": "web-01", "force": True})],
        False,
    ),
    (
        "stream_simple",
        "Weather in Madrid please.",
        [WEATHER],
        [("get_weather", {"city": "Madrid"})],
        True,
    ),
    (
        "stream_parallel",
        "What are the prices of MSFT and GOOG?",
        [STOCK],
        [("get_stock_price", {"ticker": "MSFT"}), ("get_stock_price", {"ticker": "GOOG"})],
        True,
    ),
]


def norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [norm(x) for x in v]
    return v


def subset_match(got: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(k in got and norm(got[k]) == norm(v) for k, v in want.items())


def stream_call(client: httpx.Client, body: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    calls: dict[int, dict[str, str]] = {}
    content, finish, ttft, t0 = "", None, None, time.perf_counter()
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if (d.get("content") or d.get("tool_calls")) and ttft is None:
                    ttft = time.perf_counter() - t0
                content += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    slot["name"] += f.get("name") or ""
                    slot["arguments"] += f.get("arguments") or ""
                finish = c.get("finish_reason") or finish
    msg = {
        "content": content or None,
        "tool_calls": [{"function": calls[i]} for i in sorted(calls)] or None,
    }
    return {"choices": [{"message": msg, "finish_reason": finish}]}, ttft


def main() -> int:
    results = []
    with httpx.Client(base_url=URL, timeout=180) as client:
        for name, user, tools, expected, stream in CASES:
            body = {
                "messages": [{"role": "user", "content": user}],
                "tools": tools,
                "temperature": 0,
                "max_tokens": 512,
                "parallel_tool_calls": True,
            }
            t0 = time.perf_counter()
            ttft = None
            if stream:
                data, ttft = stream_call(client, body)
            else:
                r = client.post("/v1/chat/completions", json=body)
                data = r.json()
                if r.status_code != 200:
                    print(name, "HTTP", r.status_code, json.dumps(data)[:300])
                    results.append((name, False, "http"))
                    continue
            dt = time.perf_counter() - t0
            ch = data["choices"][0]
            msg = ch["message"]
            raw_calls = msg.get("tool_calls") or []
            got = []
            parse_ok = True
            for tc in raw_calls:
                f = tc["function"]
                try:
                    args = (
                        json.loads(f["arguments"])
                        if isinstance(f["arguments"], str)
                        else f["arguments"]
                    )
                except ValueError:
                    parse_ok = False
                    args = {}
                got.append((f["name"], args))
            content = msg.get("content") or ""
            leaked = any(t in content for t in ("<tool_call>", "<think>", "</think>"))
            if expected:
                ok = (
                    parse_ok
                    and len(got) == len(expected)
                    and all(
                        any(gn == en and subset_match(ga, ea) for gn, ga in got)
                        for en, ea in expected
                    )
                )
            else:
                ok = not got and bool(content.strip())
            results.append((name, ok, ""))
            ttft_txt = f"{ttft:.2f}s" if ttft else "-"
            print(
                f"{'PASS' if ok else 'FAIL'} {name:18s} {dt:5.2f}s ttft={ttft_txt:>6} "
                f"finish={ch.get('finish_reason')} calls={json.dumps(got)[:220]} "
                f"content={content[:80]!r} leaked_tags={leaked}"
            )
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
