"""Read the BFCL data shipped in the pinned bfcl-eval package (no import of bfcl_eval)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.dataset import ROOT

BFCL_VERSION = "2026.3.23"
DATA_DIR = (
    ROOT
    / "tools"
    / "bfcl"
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "bfcl_eval"
    / "data"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_category(category: str, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    return _jsonl(data_dir / f"BFCL_v4_{category}.json")


def load_possible_answers(
    category: str, data_dir: Path = DATA_DIR
) -> dict[str, list[dict[str, Any]]]:
    """id -> ground truth [{function_name: {param: [accepted values]}}]; absent for irrelevance."""
    p = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
    if not p.exists():
        return {}
    return {e["id"]: e["ground_truth"] for e in _jsonl(p)}


def result_subdir(category: str) -> str:
    return "live" if category.startswith("live") else "non_live"
