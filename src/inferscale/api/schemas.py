"""Request schema: the OpenAI chat-completions subset the gateway accepts."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferscale.config import LimitsConfig
from inferscale.core.errors import InvalidRequestError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


FUNCTION_NAME = r"^[A-Za-z0-9_.\-]{1,128}$"  # OpenAI allows [A-Za-z0-9_-]; BFCL also uses dots


class ToolCallFunction(_Strict):
    name: str = Field(pattern=FUNCTION_NAME)
    arguments: str


class ToolCall(_Strict):
    """A tool call echoed back by the client in an assistant message."""

    id: str | None = None
    type: Literal["function"] = "function"
    index: int | None = None
    function: ToolCallFunction


class ChatMessage(_Strict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _role_fields(self) -> ChatMessage:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls are only allowed on assistant messages")
        if self.role == "tool":
            if self.tool_call_id is None or self.content is None:
                raise ValueError("tool messages need tool_call_id and content")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only allowed on tool messages")
        if self.role in ("system", "user") and self.content is None:
            raise ValueError(f"{self.role} messages need content")
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages need content or tool_calls")
        return self

    def char_count(self) -> int:
        n = len(self.content or "")
        for tc in self.tool_calls or []:
            n += len(tc.function.name) + len(tc.function.arguments)
        return n


class FunctionDef(_Strict):
    name: str = Field(pattern=FUNCTION_NAME)
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    @model_validator(mode="after")
    def _parameters_shape(self) -> FunctionDef:
        if self.parameters is not None and self.parameters.get("type", "object") != "object":
            raise ValueError("function parameters must be a JSON schema of type 'object'")
        return self


class Tool(_Strict):
    type: Literal["function"]
    function: FunctionDef


class NamedFunction(_Strict):
    name: str = Field(pattern=FUNCTION_NAME)


class NamedToolChoice(_Strict):
    type: Literal["function"]
    function: NamedFunction


MAX_SCHEMA_BYTES = 16_384


class JsonSchemaSpec(_Strict):
    """OpenAI json_schema response format; passed through to the engine as-is."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str | None = None
    strict: bool | None = None
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _schema_shape(self) -> JsonSchemaSpec:
        if self.schema_.get("type") != "object":
            raise ValueError("json_schema.schema must have type 'object'")
        if len(json.dumps(self.schema_)) > MAX_SCHEMA_BYTES:
            raise ValueError(f"json_schema.schema larger than {MAX_SCHEMA_BYTES} bytes")
        return self


class ResponseFormat(_Strict):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaSpec | None = None

    @model_validator(mode="after")
    def _schema_present(self) -> ResponseFormat:
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("response_format.json_schema is required when type is json_schema")
        if self.type != "json_schema" and self.json_schema is not None:
            raise ValueError("response_format.json_schema is only allowed when type is json_schema")
        return self


class StreamOptions(_Strict):
    include_usage: bool = False


class ChatCompletionRequest(_Strict):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: ResponseFormat | None = None
    tools: list[Tool] | None = Field(default=None, min_length=1)
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode="after")
    def _tools_consistent(self) -> ChatCompletionRequest:
        if self.tools is None and (
            self.tool_choice not in (None, "none") or self.parallel_tool_calls
        ):
            raise ValueError("tool_choice / parallel_tool_calls require tools")
        if isinstance(self.tool_choice, NamedToolChoice) and self.tools is not None:
            names = {t.function.name for t in self.tools}
            if self.tool_choice.function.name not in names:
                raise ValueError(
                    f"tool_choice names unknown function {self.tool_choice.function.name!r}"
                )
        return self

    @property
    def wants_usage_chunk(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    def check_limits(self, limits: LimitsConfig) -> None:
        if len(self.messages) > limits.max_messages:
            raise InvalidRequestError(
                f"too many messages: {len(self.messages)} > {limits.max_messages}"
            )
        chars = sum(m.char_count() for m in self.messages)
        if chars > limits.max_input_chars:
            raise InvalidRequestError(f"input too long: {chars} > {limits.max_input_chars} chars")
        if self.tools is not None:
            if len(self.tools) > limits.max_tools:
                raise InvalidRequestError(f"too many tools: {len(self.tools)} > {limits.max_tools}")
            tool_chars = len(json.dumps([t.model_dump(exclude_none=True) for t in self.tools]))
            if tool_chars > limits.max_tools_chars:
                raise InvalidRequestError(
                    f"tool definitions too large: {tool_chars} > {limits.max_tools_chars} chars"
                )
        if self.max_tokens is not None and self.max_tokens > limits.max_tokens_cap:
            raise InvalidRequestError(
                f"max_tokens {self.max_tokens} exceeds cap {limits.max_tokens_cap}"
            )

    def to_engine_payload(self, limits: LimitsConfig) -> dict[str, Any]:
        payload = self.model_dump(
            exclude_none=True, by_alias=True, exclude={"model", "stream", "stream_options"}
        )
        payload.setdefault("max_tokens", limits.default_max_tokens)
        if self.stream:
            # Always ask the engine for usage so tokens get recorded.
            payload["stream_options"] = {"include_usage": True}
        return payload
