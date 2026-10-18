"""Versioned prompts. `get_prompt(version)` returns the module implementing that version."""

from __future__ import annotations

import importlib
from types import ModuleType

PAG_PROMPTS = {
    "pag_v1.p1": "evals.prompts.pag_v1",
    "pag_v1.p2": "evals.prompts.pag_v1_p2",
}
DEFAULT_PAG_PROMPT = "pag_v1.p2"


def get_prompt(version: str) -> ModuleType:
    if version not in PAG_PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; known: {sorted(PAG_PROMPTS)}")
    mod = importlib.import_module(PAG_PROMPTS[version])
    assert mod.PROMPT_VERSION == version
    return mod
