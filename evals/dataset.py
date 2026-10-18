"""Small file helpers shared by evals and bench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

Item = dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[Item]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: list[Item]) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
