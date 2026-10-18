"""Benchmark workloads: chat prompts built from a dataset partition with a versioned prompt."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Protocol

from evals.dataset import ROOT, sha256_file


class PromptModule(Protocol):
    PROMPT_VERSION: str

    def build_messages(self, inp: dict[str, Any]) -> list[dict[str, str]]: ...

    def response_format(self, kind: str) -> dict[str, Any] | None: ...

    def fingerprint(self) -> dict[str, Any]: ...


def load_prompt(name: str) -> PromptModule:
    mod: PromptModule = importlib.import_module(f"evals.prompts.{name}")
    return mod


def pag_prompts(
    split_file: str, partition: str, prompt_name: str, response_format: str
) -> tuple[list[list[dict[str, str]]], dict[str, Any] | None, dict[str, Any]]:
    """Messages for every record of a PAG split partition, plus response_format and fingerprint."""
    from evals.datasets.pag import SOURCE, load_records

    split_path = ROOT / split_file
    split = json.loads(split_path.read_text())
    ids = split["partitions"][partition]["incident_ids"]
    records = load_records()
    prompt = load_prompt(prompt_name)
    prompts = [prompt.build_messages(records[i].model_input()) for i in ids]
    fp = {
        "dataset": {
            "name": "pag",
            "source_sha256": SOURCE["sha256"],
            "source_commit": SOURCE["commit"],
            "split_file": split_file,
            "split_sha256": sha256_file(Path(split_path)),
            "partition": partition,
            "n": len(ids),
        },
        "prompt": prompt.fingerprint(),
        "response_format": response_format,
    }
    return prompts, prompt.response_format(response_format), fp
