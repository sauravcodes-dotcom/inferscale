"""Fail if any git-tracked file contains PAG dataset text.

Checked strings (from the local pinned data file, all 950 records): Summary, Description,
Root_Cause_Description and every corrective/preventive action. Matching is case-insensitive on
whitespace-collapsed text, also in JSON-escaped form. Strings shorter than MIN_LEN characters
are ignored to avoid false positives on generic phrases.

Exit codes: 0 clean, 1 dataset text found, 0 with a SKIPPED message when the data file is
absent (nothing to compare against; run `make fetch-data` to enable the check).

Usage: uv run python scripts/check_no_dataset_text.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import ROOT
from evals.datasets.pag import PAG_FILE, load_raw, normalize

MIN_LEN = 40
SKIP_SUFFIXES = {".lock", ".png", ".jpg", ".gguf", ".pdf"}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def dataset_strings(path: Path = PAG_FILE) -> set[str]:
    raw = load_raw(path)
    out: set[str] = set()
    for i, r in enumerate(raw):
        rec = normalize(r, i)
        for s in (
            rec.summary,
            rec.description,
            rec.root_cause_description,
            *rec.corrective_actions,
            *rec.preventive_actions,
        ):
            n = _norm(s)
            if len(n) >= MIN_LEN:
                out.add(n)
    return out


def tracked_files(root: Path = ROOT) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    ).stdout
    return [root / p for p in out.decode().split("\0") if p]


def find_hits(files: Iterable[Path], strings: set[str]) -> dict[str, list[str]]:
    """Map file -> dataset strings it contains (first few)."""
    escaped = {s: _norm(json.dumps(s)[1:-1]) for s in strings}
    hits: dict[str, list[str]] = {}
    for f in files:
        if f.suffix in SKIP_SUFFIXES or not f.is_file():
            continue
        try:
            text = _norm(f.read_text())
        except UnicodeDecodeError:
            continue
        found = [s for s in strings if s in text or escaped[s] in text]
        if found:
            hits[str(f)] = sorted(found)[:3]
    return hits


def main() -> int:
    if not PAG_FILE.is_file():
        print(
            f"SKIPPED: {PAG_FILE} not present, nothing to compare against (run `make fetch-data`)"
        )
        return 0
    strings = dataset_strings()
    files = tracked_files()
    hits = find_hits(files, strings)
    if hits:
        print(f"FAIL: PAG dataset text found in {len(hits)} tracked file(s):")
        for f, ss in hits.items():
            print(f"  {f}")
            for s in ss:
                print(f"    - {s[:100]}...")
        return 1
    n_files, n_strings = len(files), len(strings)
    print(f"ok: {n_files} tracked files contain none of {n_strings} PAG strings (>= {MIN_LEN} ch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
