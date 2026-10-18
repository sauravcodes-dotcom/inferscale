"""Prompt freeze for the held-out test split: configs/eval/frozen.yaml.

The file pins {prompt_version, prompt_sha256, dev_result_file, frozen_at, reason}. run_eval
accepts --split test only when the requested prompt is the frozen one, its current sha256
still matches, and the referenced dev result exists, was a full dev run (n == dev size) and
was produced with that prompt sha.

Usage:
  uv run python -m evals.freeze --prompt pag_v1.p1 --reason "..." [--dev-result FILE]
Without --dev-result, the most recent full dev run of that prompt (by sha) is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from evals.dataset import ROOT
from evals.prompts import PAG_PROMPTS, get_prompt

FROZEN_FILE = ROOT / "configs" / "eval" / "frozen.yaml"
RESULTS = ROOT / "results" / "eval"
SPLIT_FILE = ROOT / "splits" / "pag_v1.json"
MODEL_VERSION = "qwen3-8b-q4km-v1"


def dev_size(split_file: Path = SPLIT_FILE) -> int:
    n: int = len(json.loads(split_file.read_text())["partitions"]["dev"]["incident_ids"])
    return n


def load_frozen(path: Path = FROZEN_FILE) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def dev_result_problem(result_file: str | Path, prompt_sha: str, n_dev: int) -> str | None:
    """Why this file is not a full dev run of the given prompt sha (None if it is)."""
    path = _resolve(result_file)
    if not path.is_file():
        return f"dev result {result_file} does not exist"
    s = json.loads(path.read_text())
    if s.get("run_kind") != "full" or s.get("config", {}).get("split") != "dev":
        return f"{result_file} is not a full dev run"
    if s["config"].get("n") != n_dev:
        return f"{result_file} covers {s['config'].get('n')} items, full dev is {n_dev}"
    if s["fingerprint"]["prompt"]["prompt_sha256"] != prompt_sha:
        return f"{result_file} was produced with a different prompt sha"
    return None


def find_dev_result(
    prompt_sha: str, n_dev: int, results: Path = RESULTS, model: str = MODEL_VERSION
) -> Path | None:
    for path in sorted((results / model).glob("*-dev-*.json"), reverse=True):
        if dev_result_problem(path, prompt_sha, n_dev) is None:
            return path
    return None


def final_test_refusal(
    requested_version: str,
    frozen: dict[str, Any] | None,
    split_file: Path = SPLIT_FILE,
) -> str | None:
    """Return why a test-split run with this prompt is refused, or None if allowed."""
    if frozen is None:
        rel = FROZEN_FILE.relative_to(ROOT)
        return f"no prompt is frozen ({rel} missing); run `make freeze-prompt PROMPT=...`"
    if requested_version != frozen["prompt_version"]:
        return f"prompt {requested_version} is not the frozen prompt ({frozen['prompt_version']})"
    current = get_prompt(requested_version).prompt_sha256()
    if current != frozen["prompt_sha256"]:
        was = frozen["prompt_sha256"][:12]
        return f"prompt {requested_version} changed since it was frozen ({was} -> {current[:12]})"
    problem = dev_result_problem(frozen["dev_result_file"], current, dev_size(split_file))
    if problem:
        return problem
    return None


def write_frozen(
    version: str, dev_result: Path, reason: str, path: Path = FROZEN_FILE
) -> dict[str, Any]:
    rel = dev_result.relative_to(ROOT) if dev_result.is_relative_to(ROOT) else dev_result
    data = {
        "prompt_version": version,
        "prompt_sha256": get_prompt(version).prompt_sha256(),
        "dev_result_file": str(rel),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Prompt frozen for the held-out test split. "
        "Written by `make freeze-prompt`; do not edit by hand.\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, width=100))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze a prompt for the held-out test split")
    ap.add_argument("--prompt", required=True, choices=sorted(PAG_PROMPTS))
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dev-result", type=Path, default=None)
    args = ap.parse_args(argv)
    sha = get_prompt(args.prompt).prompt_sha256()
    n_dev = dev_size()
    dev = _resolve(args.dev_result) if args.dev_result else find_dev_result(sha, n_dev)
    if dev is None:
        print(
            f"refusing: no full dev run ({n_dev} items) found for {args.prompt} ({sha[:12]})",
            file=sys.stderr,
        )
        return 1
    problem = dev_result_problem(dev, sha, n_dev)
    if problem:
        print(f"refusing: {problem}", file=sys.stderr)
        return 1
    data = write_frozen(args.prompt, dev, args.reason)
    print(yaml.safe_dump(data, sort_keys=False, width=100), end="")
    print(f"wrote {FROZEN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
