"""Prompt for the PAG incident task. Bump PROMPT_VERSION on any change to the text or schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROMPT_VERSION = "pag_v1.p1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "pag_output_v1.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())

_props = SCHEMA["properties"]
TYPES: list[str] = _props["root_cause_type"]["enum"]
TEAMS: list[str] = _props["root_cause_team"]["enum"]

SYSTEM_PROMPT = f"""\
You are a senior site reliability engineer writing the post-incident CAPM (corrective and \
preventive action plan) for a cloud incident. Read the incident report and respond with a \
single JSON object only (no prose, no markdown).

Fields:
- "severity": "Sev-1" or "Sev-2".
- "root_cause_type": exactly one of: {", ".join(json.dumps(t) for t in TYPES)}.
- "root_cause_team": exactly one of: {", ".join(json.dumps(t) for t in TEAMS)}.
- "root_cause_summary": one or two sentences (at most 300 characters) stating the root cause.
- "corrective_actions": 3 to 6 concrete actions that fix this incident now (each 20-600 \
characters).
- "preventive_actions": 3 to 6 concrete actions that stop it from recurring (each 20-600 \
characters).

Actions must be specific to the root cause, the affected service and the responsible team. \
Use the root-cause type and team stated in the report."""

INPUT_ORDER = (
    "Summary",
    "Description",
    "Detection_Method",
    "Impact_Type",
    "Impacted_Customer_Count",
    "Region",
    "Repeat_Outage",
    "Impact_Start_Time",
    "Detected_Time",
    "Mitigation_Time",
)


def user_message(inp: dict[str, Any]) -> str:
    lines = ["Incident report:"]
    for key in INPUT_ORDER:
        val = inp[key]
        if isinstance(val, bool):
            val = "yes" if val else "no"
        lines.append(f"{key.replace('_', ' ')}: {val}")
    return "\n".join(lines)


def build_messages(inp: dict[str, Any]) -> list[dict[str, str]]:
    unknown = set(inp) - set(INPUT_ORDER)
    if unknown:
        raise ValueError(f"fields not allowed in model input: {sorted(unknown)}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(inp)},
    ]


def response_format(kind: str) -> dict[str, Any] | None:
    if kind == "none":
        return None
    if kind == "json_schema":
        clean = {k: v for k, v in SCHEMA.items() if not k.startswith("$")}
        return {
            "type": "json_schema",
            "json_schema": {"name": "pag_output_v1", "strict": True, "schema": clean},
        }
    raise ValueError(f"unknown response format {kind!r}")


def prompt_sha256() -> str:
    """Hash of everything that defines the prompt: this file's text and the output schema."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(SCHEMA_PATH.read_bytes())
    return h.hexdigest()


def fingerprint() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "schema": SCHEMA_PATH.name,
    }
