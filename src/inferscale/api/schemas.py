"""Request schema: the OpenAI chat-completions subset the gateway accepts."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferscale.config import LimitsConfig
from inferscale.core.errors import InvalidRequestError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str


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

    @property
    def wants_usage_chunk(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    def check_limits(self, limits: LimitsConfig) -> None:
        if len(self.messages) > limits.max_messages:
            raise InvalidRequestError(
                f"too many messages: {len(self.messages)} > {limits.max_messages}"
            )
        chars = sum(len(m.content) for m in self.messages)
        if chars > limits.max_input_chars:
            raise InvalidRequestError(f"input too long: {chars} > {limits.max_input_chars} chars")
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
