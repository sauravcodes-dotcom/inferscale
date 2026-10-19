"""Model-free BFCL baselines, scored by the official BFCL scorer.

- never_call: answer every entry with text and no tool call. Correct only where the right
  behaviour is to not call a tool (irrelevance categories): the floor.
- bm25_noargs: call the BM25-best function (query = user turns, documents = function name +
  description + parameter names) with empty arguments. Shows that choosing a plausible tool
  without understanding its arguments scores ~0 under AST checking.

Also reports function-selection accuracy on the categories where the model must choose one
function among several (multiple, live_multiple): top-1 function name vs the ground-truth name,
for BM25 and, when a model run directory is given, for the model.

  uv run python -m evals.bfcl.baselines --partition dev [--model-run results/bfcl/raw/<run_id>]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.bfcl.data import load_category, load_possible_answers, result_subdir
from evals.bfcl.metrics import ingest, summarize
from evals.bfcl.run import ADAPTER, BFCL_PYTHON, RESULTS, SPLIT_FILE, ids_sha256
from evals.dataset import ROOT, sha256_file

SELECTION_CATEGORIES = ("multiple", "live_multiple")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(
        re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").replace(".", " ").lower()
    )


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [tokens(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in toks) / n if n else 0.0
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in set(tokens(query)):
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / (avgdl or 1)))
        scores.append(s)
    return scores


def user_text(entry: dict[str, Any]) -> str:
    return " ".join(
        m["content"] for turn in entry["question"] for m in turn if m.get("role") == "user"
    )


def function_doc(fn: dict[str, Any]) -> str:
    params = " ".join((fn.get("parameters") or {}).get("properties", {}).keys())
    return f"{fn['name']} {fn.get('description', '')} {params}"


def bm25_choice(entry: dict[str, Any]) -> str | None:
    fns = entry.get("function") or []
    if not fns:
        return None
    scores = bm25_rank(user_text(entry), [function_doc(f) for f in fns])
    return str(fns[max(range(len(fns)), key=lambda i: (scores[i], -i))]["name"])


def as_fc_name(name: str) -> str:
    """The FC handler sends names with '.' replaced by '_' (underscore_to_dot models)."""
    return name.replace(".", "_")


def gold_names(truth: list[dict[str, Any]]) -> set[str]:
    return {as_fc_name(n) for call in truth for n in call}


def write_results(run_dir: Path, model_dir: str, ids: dict[str, list[str]], kind: str) -> None:
    for cat, cat_ids in ids.items():
        wanted = set(cat_ids)
        lines = []
        for e in load_category(cat):
            if e["id"] not in wanted:
                continue
            if kind == "never_call":
                res: Any = "I can't help with that using the available tools."
            else:
                choice = bm25_choice(e)
                res = [{as_fc_name(choice): "{}"}] if choice else ""
            lines.append(
                json.dumps(
                    {
                        "id": e["id"],
                        "result": res,
                        "input_token_count": 0,
                        "output_token_count": 0,
                        "latency": 0.0,
                    }
                )
            )
        out = run_dir / "result" / model_dir / result_subdir(cat) / f"BFCL_v4_{cat}_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")


def evaluate(run_dir: Path, model_version: str) -> None:
    argv = [
        str(BFCL_PYTHON),
        str(ADAPTER),
        "evaluate",
        "--run-dir",
        str(run_dir),
        "--model-version",
        model_version,
    ]
    with (run_dir / "evaluate.log").open("w") as f:
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"bfcl evaluate failed; see {run_dir / 'evaluate.log'}")


def model_choices(model_run: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for f in model_run.glob("result/*/*/BFCL_v4_*_result.json"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            res = r.get("result")
            out[r["id"]] = (
                next(iter(res[0]))
                if isinstance(res, list) and res and isinstance(res[0], dict)
                else None
            )
    return out


def selection_accuracy(ids: dict[str, list[str]], chooser: dict[str, str | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cat in SELECTION_CATEGORIES:
        truth = load_possible_answers(cat)
        cat_ids = [i for i in ids.get(cat, []) if i in truth and i in chooser]
        correct = sum(
            1 for i in cat_ids if chooser[i] is not None and chooser[i] in gold_names(truth[i])
        )
        out[cat] = {"n": len(cat_ids), "accuracy": correct / len(cat_ids) if cat_ids else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score model-free BFCL baselines with the official scorer"
    )
    ap.add_argument("--partition", choices=("dev", "test"), default="dev")
    ap.add_argument(
        "--model-run", type=Path, default=None, help="raw run dir of a model run, for selection"
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    split = json.loads(SPLIT_FILE.read_text())
    ids = {c: v for c, v in split["partitions"][args.partition]["ids"].items() if v}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, Any] = {
        "partition": args.partition,
        "split_sha256": sha256_file(SPLIT_FILE),
        "ids_sha256": ids_sha256(ids),
        "baselines": {},
    }
    for kind in ("never_call", "bm25_noargs"):
        version = f"baseline-{kind.replace('_', '-')}"
        run_dir = RESULTS / "raw" / f"{stamp}-{args.partition}-{version}"
        run_dir.mkdir(parents=True)
        (run_dir / "test_case_ids_to_generate.json").write_text(json.dumps(ids))
        write_results(run_dir, f"inferscale_{version}-FC", ids, kind)
        evaluate(run_dir, version)
        rows = ingest(run_dir, f"inferscale_{version}-FC")
        out["baselines"][kind] = summarize(rows, n_boot=args.bootstrap)
    bm25 = {
        e["id"]: (as_fc_name(c) if (c := bm25_choice(e)) else None)
        for cat in SELECTION_CATEGORIES
        for e in load_category(cat)
    }
    out["function_selection"] = {"bm25": selection_accuracy(ids, bm25)}
    if args.model_run:
        model_run = args.model_run if args.model_run.is_absolute() else ROOT / args.model_run
        out["function_selection"]["model"] = selection_accuracy(ids, model_choices(model_run))
        out["function_selection"]["model_run"] = str(args.model_run)
    dest = RESULTS / "baselines" / f"{args.partition}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for kind, s in out["baselines"].items():
        m = s["metrics"]
        print(
            kind,
            {k: m[k]["value"] for k in ("non_live_ast", "live", "irrelevance", "live_irrelevance")},
        )
    print("function selection:", json.dumps(out["function_selection"]))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
