"""Run-configuration freeze for the BFCL held-out partition: configs/eval/bfcl_frozen.yaml.

For BFCL the prompt is BFCL's own; what can be tuned on dev is the run configuration. The
config fingerprint covers: the adapter source, the BFCL lockfile, the engine profile file, the
gateway serving profile, the temperature, and what the running engine reports (/props: build,
model path, per-slot context). The held-out partition is accepted only when the current
fingerprint equals the frozen one and the referenced dev result is a full dev run (all dev
categories, no per-category limit) produced with that fingerprint.

  uv run python -m evals.bfcl.freeze --dev-result results/bfcl/<model>/<run>.json --reason "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.dataset import ROOT

FROZEN_FILE = ROOT / "configs" / "eval" / "bfcl_frozen.yaml"
ADAPTER = ROOT / "tools" / "bfcl" / "inferscale_bfcl.py"
LOCK = ROOT / "tools" / "bfcl" / "requirements.lock"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def engine_state(engine_url: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            p = (await c.get(f"{engine_url}/props")).json()
    except (httpx.HTTPError, ValueError):
        return {}
    return {
        "build_info": p.get("build_info"),
        "model_path": p.get("model_path"),
        "n_ctx_per_slot": (p.get("default_generation_settings") or {}).get("n_ctx"),
        "total_slots": p.get("total_slots"),
    }


async def current_config(args: argparse.Namespace) -> dict[str, Any]:
    profile = ROOT / "configs" / "engine" / f"{args.engine_profile}.env"
    serving = ROOT / args.serving_profile
    parts = {
        "adapter_sha256": _sha(ADAPTER),
        "bfcl_lock_sha256": _sha(LOCK),
        "engine_profile": args.engine_profile,
        "engine_profile_sha256": _sha(profile),
        "serving_profile": args.serving_profile,
        "serving_profile_sha256": _sha(serving),
        "temperature": args.temperature,
        "engine": await engine_state(args.engine_url),
    }
    parts["config_sha256"] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return parts


def load_frozen(path: Path = FROZEN_FILE) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def dev_result_problem(result_file: str | Path, config_sha: str) -> str | None:
    p = Path(result_file)
    p = p if p.is_absolute() else ROOT / p
    if not p.is_file():
        return f"dev result {result_file} does not exist"
    s = json.loads(p.read_text())
    if s.get("partition") != "dev" or s.get("run_kind") != "full":
        return f"{result_file} is not a full dev run"
    split = json.loads((ROOT / "splits" / "bfcl_v4_single_turn_v1.json").read_text())
    expected = sorted(c for c, v in split["partitions"]["dev"]["ids"].items() if v)
    if sorted(s["fingerprint"]["split"]["categories"]) != expected:
        return f"{result_file} does not cover every dev category"
    if s["config"]["config_sha256"] != config_sha:
        return f"{result_file} was produced with a different run configuration"
    return None


def test_refusal(config: dict[str, Any], frozen: dict[str, Any] | None) -> str | None:
    """Why a held-out run with `config` is refused (None if allowed). `frozen` = load_frozen()."""
    if frozen is None:
        return f"no BFCL configuration is frozen ({FROZEN_FILE.relative_to(ROOT)} missing)"
    if config["config_sha256"] != frozen["config_sha256"]:
        changed = sorted(k for k, v in frozen.get("config", {}).items() if config.get(k) != v)
        what = ", ".join(changed) or "unknown"
        return f"run configuration differs from the frozen one (changed: {what})"
    return dev_result_problem(frozen["dev_result_file"], frozen["config_sha256"])


def write_frozen(dev_result: Path, reason: str, path: Path = FROZEN_FILE) -> dict[str, Any]:
    s = json.loads(dev_result.read_text())
    problem = dev_result_problem(dev_result, s["config"]["config_sha256"])
    if problem:
        raise SystemExit(f"refusing to freeze: {problem}")
    rel = dev_result.relative_to(ROOT) if dev_result.is_relative_to(ROOT) else dev_result
    data = {
        "config_sha256": s["config"]["config_sha256"],
        "dev_result_file": str(rel),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "config": {k: v for k, v in s["config"].items() if k != "config_sha256"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# BFCL run configuration frozen for the held-out partition (evals.bfcl.freeze).\n"
        + yaml.safe_dump(data, sort_keys=False, width=100)
    )
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the BFCL run configuration")
    ap.add_argument("--dev-result", type=Path, required=True)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    p = args.dev_result if args.dev_result.is_absolute() else ROOT / args.dev_result
    print(yaml.safe_dump(write_frozen(p, args.reason), sort_keys=False, width=100), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
