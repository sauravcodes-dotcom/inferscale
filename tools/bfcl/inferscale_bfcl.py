"""Run the official BFCL generator and scorer against the InferScale gateway.

Runs under the isolated BFCL environment (tools/bfcl/.venv, pinned bfcl-eval==2026.3.23),
never under the InferScale project environment.

The model handler is the official QwenAPIHandler (the handler behind the leaderboard's
"Qwen3-8B (FC)" entry) with two changes only:
  - requests go to the InferScale gateway (OpenAI-compatible chat completions with tools,
    streaming, include_usage) instead of DashScope;
  - no `enable_thinking` extra_body field: thinking is an engine-level setting in InferScale
    (llama-server --reasoning on|off) and is recorded in the run metadata.
Prompt construction, tool compilation, response parsing and scoring are BFCL's own code.

Usage (from the repo root):
  tools/bfcl/.venv/bin/python tools/bfcl/inferscale_bfcl.py generate --run-dir DIR --ids IDS.json
  tools/bfcl/.venv/bin/python tools/bfcl/inferscale_bfcl.py evaluate --run-dir DIR --ids IDS.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REGISTRY_NAME = "inferscale/{model_version}-FC"


def _setup(run_dir: Path, ids_file: Path | None) -> None:
    """Point BFCL at the per-run directory. Must run before any bfcl_eval import."""
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BFCL_PROJECT_ROOT"] = str(run_dir.resolve())
    # The parent handlers construct OpenAI/DashScope clients that insist on a key; they are
    # replaced by the gateway client. The placeholder is not a credential.
    os.environ.setdefault("OPENAI_API_KEY", "inferscale-local-no-key")
    os.environ.setdefault("QWEN_API_KEY", "inferscale-local-no-key")
    if ids_file is not None:
        shutil.copyfile(ids_file, run_dir / "test_case_ids_to_generate.json")


def _register(model_version: str, gateway_url: str, timeout_s: float) -> str:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler
    from openai import OpenAI

    class InferScaleQwenFCHandler(QwenAPIHandler):  # type: ignore[misc]
        def __init__(
            self,
            model_name: str,
            temperature: float,
            registry_name: str,
            is_fc_model: bool,
            **kwargs: Any,
        ) -> None:
            super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
            self.client = OpenAI(
                base_url=f"{gateway_url}/v1", api_key="inferscale", timeout=timeout_s, max_retries=0
            )

        def _query_FC(self, inference_data: dict[str, Any]) -> Any:
            message = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
            return self.generate_with_backoff(
                messages=message,
                model=model_version,
                tools=tools,
                parallel_tool_calls=True,
                temperature=self.temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

    name = REGISTRY_NAME.format(model_version=model_version)
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=model_version,
        display_name=f"InferScale {model_version} (FC)",
        url="https://huggingface.co/Qwen/Qwen3-8B",
        org="InferScale",
        license="apache-2.0",
        model_handler=InferScaleQwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,  # same as the leaderboard's Qwen3-8B (FC) API config
    )
    return name


def generate(args: argparse.Namespace) -> int:
    _setup(Path(args.run_dir), Path(args.ids))
    name = _register(args.model_version, args.gateway, args.timeout)
    from bfcl_eval import _llm_response_generation as gen

    ns = argparse.Namespace(
        model=[name],
        test_category=["all"],
        temperature=args.temperature,
        include_input_log=False,
        exclude_state_log=False,
        num_threads=args.threads,
        num_gpus=1,
        backend="vllm",
        gpu_memory_utilization=0.9,
        result_dir=None,
        run_ids=True,
        allow_overwrite=args.overwrite,
        skip_server_setup=True,
        local_model_path=None,
        lora_modules=None,
        enable_lora=False,
        max_lora_rank=None,
    )
    gen.main(ns)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    _setup(Path(args.run_dir), Path(args.ids) if args.ids else None)
    name = _register(args.model_version, args.gateway, args.timeout)
    from bfcl_eval.eval_checker import eval_runner

    categories = sorted(
        k
        for k, v in json.loads(
            (Path(args.run_dir) / "test_case_ids_to_generate.json").read_text()
        ).items()
        if v
    )
    eval_runner.main([name], categories, None, None, partial_eval=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("generate", "evaluate"):
        p = sub.add_parser(cmd)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--ids", required=cmd == "generate")
        p.add_argument("--model-version", default="qwen3-8b-q4km-v1")
        p.add_argument("--gateway", default="http://127.0.0.1:8000")
        p.add_argument("--timeout", type=float, default=300.0)
        if cmd == "generate":
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--temperature", type=float, default=0.001)  # BFCL's default
            p.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    return generate(args) if args.cmd == "generate" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
