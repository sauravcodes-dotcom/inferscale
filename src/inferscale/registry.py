"""Model registry: pinned model artifacts and the active model version."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_repo: str = Field(min_length=1)
    hf_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine: Literal["llama.cpp"]
    engine_version: str = Field(min_length=1)
    quant: str = Field(min_length=1)
    thinking: bool = False


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active: str
    models: dict[str, ModelEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _active_exists(self) -> Registry:
        if self.active not in self.models:
            raise ValueError(f"active model {self.active!r} is not in models")
        return self

    @property
    def active_entry(self) -> ModelEntry:
        return self.models[self.active]


def load_registry(path: str | Path, active: str | None = None) -> Registry:
    """`active` overrides the file's active version (it must be one of its models)."""
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if active is not None:
        data["active"] = active
    return Registry.model_validate(data)
