"""Dataset I/O shared by the generator, apply_review and run_eval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRIAGE_V1 = ROOT / "evals" / "datasets" / "triage_v1"
SPLITS = ("dev", "test")

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


def load_schema(dataset_dir: Path = TRIAGE_V1) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((dataset_dir / "schema.json").read_text())
    return schema


def split_counts(items: list[Item]) -> dict[str, Any]:
    return {
        "n": len(items),
        "category": dict(sorted(Counter(i["label"]["category"] for i in items).items())),
        "severity": dict(sorted(Counter(i["label"]["severity"] for i in items).items())),
        "component": dict(sorted(Counter(i["label"]["component"] for i in items).items())),
        "ambiguous": sum(1 for i in items if i.get("meta", {}).get("ambiguous")),
        "review_status": dict(sorted(Counter(i["review_status"] for i in items).items())),
    }


def write_manifest(dataset_dir: Path, base: dict[str, Any]) -> dict[str, Any]:
    """Recompute file hashes and counts; keep other keys from `base`."""
    files = {}
    for name in ("dev.jsonl", "test.jsonl", "schema.json"):
        p = dataset_dir / name
        files[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    counts = {s: split_counts(load_jsonl(dataset_dir / f"{s}.jsonl")) for s in SPLITS}
    manifest = {**base, "dataset": "triage_v1", "files": files, "counts": counts}
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
