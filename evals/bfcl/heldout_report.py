"""Held-out BFCL report: the frozen config's dev and test runs side by side.

Recomputes both runs' metrics from their per-entry rows with the current statistics (CIs
resample live source groups; see metrics.cluster_key), and reports what the summaries alone do
not show:

- official vs conservative scoring of serving errors: our ingest counts an entry whose request
  failed as wrong; BFCL's scorer counts a failed irrelevance entry as correct (no call made);
- responses that hit the gateway's default max_tokens (256: BFCL sets no max_tokens);
- the published Qwen3-8B (FC) reference.

  uv run python -m evals.bfcl.heldout_report <dev_summary.json> <test_summary.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evals.bfcl.compare import load_rows
from evals.bfcl.metrics import PUBLISHED_QWEN3_8B, Row, summarize

IRRELEVANCE = {"irrelevance", "live_irrelevance"}
DEFAULT_MAX_TOKENS = 256  # configs/serving/mac-llamacpp-bfcl-gwv.yaml limits.default_max_tokens
HEADLINE = ("non_live_ast", "live", "irrelevance", "live_irrelevance")
DETAIL = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)


def official(rows: list[Row]) -> list[Row]:
    """Rows as BFCL's scorer counts them: a failed irrelevance request is not a call."""
    return [
        {**r, "correct": True, "error_class": None}
        if r["error_class"] == "serving_error" and r["category"] in IRRELEVANCE
        else r
        for r in rows
    ]


def run_report(summary: dict[str, Any], rows: list[Row]) -> dict[str, Any]:
    conservative = summarize(rows)
    off = summarize(official(rows))
    truncated = [r for r in rows if r.get("output_tokens") == DEFAULT_MAX_TOKENS]
    return {
        "run_id": summary["run_id"],
        "partition": summary["partition"],
        "config_sha256": summary["config"]["config_sha256"],
        "n": len(rows),
        "metrics_official": off["metrics"],
        "metrics_conservative": conservative["metrics"],
        "serving_errors": [r["id"] for r in rows if r["error_class"] == "serving_error"],
        "error_classes": conservative["error_classes"],
        "truncated_at_default_max_tokens": {
            "n": len(truncated),
            "by_category_and_correct": {
                f"{c}|{k}": n
                for (c, k), n in Counter((r["category"], r["correct"]) for r in truncated).items()
            },
        },
        "bootstrap": conservative["bootstrap"],
        "serving": summary.get("serving"),
    }


def fmt(m: dict[str, Any]) -> str:
    if m.get("value") is None:
        return "n/a"
    ci = m.get("ci95")
    return f"{m['value'] * 100:.2f} [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if ci else ""


def render(dev: dict[str, Any], test: dict[str, Any]) -> str:
    pub = PUBLISHED_QWEN3_8B["FC"]
    lines = [
        f"| metric | dev (n={dev['n']}) | **test (n={test['n']})** | published Qwen3-8B (FC) |",
        "|---|---|---|---|",
    ]
    for k in HEADLINE + DETAIL:
        p = pub.get(k)
        lines.append(
            f"| {'**' + k + '**' if k in HEADLINE else k} | {fmt(dev['metrics_official'][k])} "
            f"| {fmt(test['metrics_official'][k])} | {'' if p is None else f'{p:.2f}'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dev", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    sd, rd = load_rows(args.dev)
    st, rt = load_rows(args.test)
    if sd["config"]["config_sha256"] != st["config"]["config_sha256"]:
        raise SystemExit("dev and test runs used different configurations")
    if st["partition"] != "test" or sd["partition"] != "dev":
        raise SystemExit("expected a dev run and a test run")
    res = {"dev": run_report(sd, rd), "test": run_report(st, rt)}
    out = args.out or args.test.parent / f"heldout-report-{st['run_id']}.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(render(res["dev"], res["test"]))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
