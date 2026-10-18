"""Download the PAG synthetic cloud-incident dataset at a pinned commit and verify its sha256.

The source repository has no LICENSE file, so the data is never committed: it lives in the
git-ignored data/pag/ directory.

Usage: uv run python scripts/fetch_pag.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.datasets.pag import PAG_FILE, SOURCE


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the pinned PAG dataset file")
    ap.add_argument(
        "--force", action="store_true", help="re-download even if a verified copy exists"
    )
    args = ap.parse_args()
    if PAG_FILE.exists() and not args.force:
        digest = sha256_bytes(PAG_FILE.read_bytes())
        if digest == SOURCE["sha256"]:
            print(f"ok: {PAG_FILE} already present (sha256 {digest})")
            return 0
        print(
            f"existing file has sha256 {digest}, expected {SOURCE['sha256']}; re-downloading",
            file=sys.stderr,
        )
    url = SOURCE["raw_url"]
    print(f"downloading {url}", file=sys.stderr)
    resp = httpx.get(url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    digest = sha256_bytes(resp.content)
    if digest != SOURCE["sha256"]:
        print(
            f"sha256 mismatch: got {digest}, expected {SOURCE['sha256']}; nothing written",
            file=sys.stderr,
        )
        return 1
    PAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAG_FILE.write_bytes(resp.content)
    (PAG_FILE.parent / "SOURCE.json").write_text(
        json.dumps({**SOURCE, "bytes": len(resp.content)}, indent=2) + "\n"
    )
    print(f"ok: wrote {PAG_FILE} ({len(resp.content)} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
