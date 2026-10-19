"""Gateway settings: YAML serving profile, overridden by INFERSCALE_* environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_FILE = "configs/serving/mac-llamacpp.yaml"
CONFIG_FILE_ENV = "INFERSCALE_CONFIG_FILE"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdmissionConfig(_Section):
    max_inflight: int = Field(default=2, ge=1)
    max_queue: int = Field(default=8, ge=0)
    queue_timeout_s: float = Field(default=10.0, gt=0)
    retry_after_s: int = Field(default=1, ge=0)
    # llama.cpp keeps processing a prompt after the client is gone (it only checks for
    # cancellation after prefill), so a request that ends early may still occupy an engine
    # slot. When enabled, its admission permit is held until the engine reports a free slot
    # (polled via /slots), for at most hold_max_s.
    hold_permits_until_engine_idle: bool = True
    hold_max_s: float = Field(default=120.0, gt=0)
    hold_poll_s: float = Field(default=0.25, gt=0)


class DeadlineConfig(_Section):
    connect_timeout_s: float = Field(default=2.0, gt=0)
    first_token_timeout_s: float = Field(default=30.0, gt=0)
    total_timeout_s: float = Field(default=120.0, gt=0)


class LimitsConfig(_Section):
    max_messages: int = Field(default=32, ge=1)
    max_input_chars: int = Field(default=16000, ge=1)
    max_tokens_cap: int = Field(default=512, ge=1)
    default_max_tokens: int = Field(default=256, ge=1)
    max_tools: int = Field(default=128, ge=1)
    max_tools_chars: int = Field(default=65536, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERSCALE_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    config_file: str = DEFAULT_CONFIG_FILE
    engine_url: str = "http://127.0.0.1:8081"
    registry_path: str = "configs/models/registry.yaml"
    # Serve another registry version than the file's `active` (e.g. a quantization sweep); the
    # engine must be running that version's file.
    active_model: str | None = None
    admission: AdmissionConfig = AdmissionConfig()
    deadlines: DeadlineConfig = DeadlineConfig()
    limits: LimitsConfig = LimitsConfig()
    drain_timeout_s: float = Field(default=30.0, ge=0)
    health_poll_interval_s: float = Field(default=2.0, gt=0)
    log_content: bool = False
    # How requests with tools are served (see inferscale.backends.openai_http):
    # "engine" = llama.cpp chat endpoint (grammar-constrained), "gateway" = unconstrained
    # generation + gateway-side tool-call parsing, "gateway_validated" = gateway + JSON-schema
    # validation of each call with fallback to the engine's constrained path.
    tool_call_mode: Literal["engine", "gateway", "gateway_validated"] = "engine"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: explicit kwargs > environment > YAML file.
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if settings_cls.model_config.get("yaml_file"):
            sources.append(YamlConfigSettingsSource(settings_cls))
        return tuple(sources)


def load_settings(config_file: str | Path | None = None) -> Settings:
    """Load settings from a YAML profile (default: $INFERSCALE_CONFIG_FILE or the Mac profile)."""
    path = Path(config_file or os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))
    if not path.is_file():
        raise FileNotFoundError(f"serving config not found: {path}")
    bound = type(
        "BoundSettings",
        (Settings,),
        {"model_config": {**Settings.model_config, "yaml_file": str(path)}},
    )
    settings: Settings = bound(config_file=str(path))
    return settings
