"""Export BFCL requests exactly as the official FC handler builds them (messages + tools).

Runs in the isolated BFCL environment. Output (dataset text; git-ignored):
  results/bfcl/raw/requests-<partition>.jsonl  one {"id", "category", "messages", "tools"} per line

  tools/bfcl/.venv/bin/python tools/bfcl/export_requests.py --partition dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--partition", choices=("dev", "test"), default="dev")
    ap.add_argument("--split", default=str(ROOT / "splits" / "bfcl_v4_single_turn_v1.json"))
    args = ap.parse_args(argv)
    os.environ.setdefault("BFCL_PROJECT_ROOT", str(ROOT / "results" / "bfcl" / "raw" / "export"))
    os.environ.setdefault("OPENAI_API_KEY", "inferscale-local-no-key")
    os.environ.setdefault("QWEN_API_KEY", "inferscale-local-no-key")
    from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler
    from bfcl_eval.utils import load_dataset_entry

    ids = json.loads(Path(args.split).read_text())["partitions"][args.partition]["ids"]
    handler = QwenAPIHandler("qwen3-8b", 0.001, "qwen3-8b-FC", True)
    out = ROOT / "results" / "bfcl" / "raw" / f"requests-{args.partition}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for category, cat_ids in ids.items():
            if not cat_ids:
                continue
            wanted = set(cat_ids)
            for entry in load_dataset_entry(category):
                if entry["id"] not in wanted:
                    continue
                data = handler._pre_query_processing_FC({"message": []}, entry)
                data = handler._compile_tools(data, entry)
                data = handler.add_first_turn_message_FC(data, entry["question"][0])
                f.write(
                    json.dumps(
                        {
                            "id": entry["id"],
                            "category": category,
                            "messages": data["message"],
                            "tools": data["tools"],
                        }
                    )
                    + "\n"
                )
                n += 1
    print(f"wrote {n} requests to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
