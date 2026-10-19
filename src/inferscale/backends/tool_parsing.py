"""Gateway-side tool-call parsing for Qwen/Hermes-style output.

Why: with `tools`, llama.cpp b11669 constrains decoding with a grammar built from each tool's
JSON schema that fixes the order of properties. When the model emits valid arguments in a
different order, the grammar forces other tokens and corrupts them (e.g. a `cast` value that
swallows the next key). In `tool_call_mode: gateway` the gateway renders the prompt with the
model's chat template, generates without that grammar, and parses the output here:

    <tool_call>
    {"name": "fn", "arguments": {...}}
    </tool_call>

Text outside the blocks is message content. A block that is not valid JSON, or has no string
`name`, is kept verbatim in the content (and counted) instead of being dropped or repaired.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

Json = dict[str, Any]


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def parse_block(block: str) -> Json | None:
    """One <tool_call> body -> OpenAI tool_call object, or None if malformed."""
    try:
        obj = json.loads(block.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {
        "id": new_call_id(),
        "type": "function",
        "function": {"name": obj["name"], "arguments": json.dumps(args, ensure_ascii=False)},
    }


@dataclass
class Parsed:
    content: str | None
    tool_calls: list[Json] = field(default_factory=list)
    malformed: int = 0


def parse_output(text: str) -> Parsed:
    text = _THINK.sub("", text)
    parts: list[str] = []
    calls: list[Json] = []
    malformed = 0
    pos = 0
    while True:
        start = text.find(OPEN, pos)
        if start < 0:
            parts.append(text[pos:])
            break
        parts.append(text[pos:start])
        end = text.find(CLOSE, start + len(OPEN))
        body = text[start + len(OPEN) : end if end >= 0 else len(text)]
        call = parse_block(body)
        if call is None:
            malformed += 1
            parts.append(text[start : end + len(CLOSE) if end >= 0 else len(text)])
        else:
            calls.append(call)
        if end < 0:
            break
        pos = end + len(CLOSE)
    content = "".join(parts).strip()
    return Parsed(content or None, calls, malformed)


class StreamParser:
    """Incremental version of parse_output: feed text deltas, get content text and complete calls.

    Text that could be the start of an opening tag is held back until it is disambiguated, so
    clients never see a partial "<tool_ca" in content.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_call = False
        self._in_think = False
        self.calls: list[Json] = []
        self.malformed = 0
        self.emitted_text = False

    def feed(self, delta: str) -> tuple[str, list[Json]]:
        self._buf += delta
        out_text: list[str] = []
        new_calls: list[Json] = []
        while self._buf:
            if self._in_think:
                end = self._buf.find("</think>")
                if end < 0:
                    self._buf = self._buf[-len("</think>") :]
                    break
                self._buf = self._buf[end + len("</think>") :].lstrip()
                self._in_think = False
                continue
            if self._in_call:
                end = self._buf.find(CLOSE)
                if end < 0:
                    break
                body, self._buf = self._buf[:end], self._buf[end + len(CLOSE) :]
                self._in_call = False
                call = parse_block(body)
                if call is None:
                    self.malformed += 1
                    out_text.append(OPEN + body + CLOSE)
                else:
                    self.calls.append(call)
                    new_calls.append(call)
                continue
            idx = self._buf.find("<")
            if idx < 0:
                out_text.append(self._buf)
                self._buf = ""
                break
            out_text.append(self._buf[:idx])
            rest = self._buf[idx:]
            if rest.startswith(OPEN):
                self._in_call = True
                self._buf = rest[len(OPEN) :]
            elif rest.startswith("<think>"):
                self._in_think = True
                self._buf = rest[len("<think>") :]
            elif OPEN.startswith(rest) or "<think>".startswith(rest):
                self._buf = rest  # possible partial tag: wait for more text
                break
            else:
                out_text.append("<")
                self._buf = rest[1:]
        text = "".join(out_text)
        if text.strip():
            self.emitted_text = True
        return text, new_calls

    def finish(self) -> tuple[str, list[Json]]:
        """Flush at end of generation. An unterminated block is parsed if it is complete JSON."""
        text, calls = "", []
        if self._in_call:
            call = parse_block(self._buf)
            if call is None:
                self.malformed += 1
                text = OPEN + self._buf
            else:
                self.calls.append(call)
                calls.append(call)
        elif not self._in_think:
            text = self._buf
        self._buf = ""
        self._in_call = False
        return text, calls
