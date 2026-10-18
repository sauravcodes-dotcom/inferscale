"""Prompt pag_v1.p2 for the PAG incident task.

Changes vs pag_v1.p1 (evals/prompts/pag_v1.py, kept unchanged), and nothing else:
1. Ask for exactly 4 corrective and exactly 4 preventive actions, each specific and concrete
   (roughly 120-200 characters, naming the affected team/component and the root cause where
   relevant). The JSON schema is unchanged (3-6 items allowed), so format compliance stays
   measurable.
2. root_cause_team must be copied exactly as written in the Description and root_cause_type
   exactly as written in the Summary.
The user message, input fields and response_format are identical to p1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.prompts import pag_v1 as _p1

PROMPT_VERSION = "pag_v1.p2"
SCHEMA_PATH = _p1.SCHEMA_PATH
SCHEMA = _p1.SCHEMA
TYPES = _p1.TYPES
TEAMS = _p1.TEAMS
INPUT_ORDER = _p1.INPUT_ORDER
user_message = _p1.user_message
response_format = _p1.response_format

SYSTEM_PROMPT = f"""\
You are a senior site reliability engineer writing the post-incident CAPM (corrective and \
preventive action plan) for a cloud incident. Read the incident report and respond with a \
single JSON object only (no prose, no markdown).

Fields:
- "severity": "Sev-1" or "Sev-2".
- "root_cause_type": exactly one of: {", ".join(json.dumps(t) for t in TYPES)}. Copy it exactly \
as it is written in the Summary.
- "root_cause_team": exactly one of: {", ".join(json.dumps(t) for t in TEAMS)}. Copy it exactly \
as it is written in the Description.
- "root_cause_summary": one or two sentences (at most 300 characters) stating the root cause.
- "corrective_actions": exactly 4 actions that fix this incident now.
- "preventive_actions": exactly 4 actions that stop it from recurring.

Each action must be specific and concrete, roughly 120-200 characters long, and name the \
affected team or component and the root cause where relevant."""


def build_messages(inp: dict[str, Any]) -> list[dict[str, str]]:
    unknown = set(inp) - set(INPUT_ORDER)
    if unknown:
        raise ValueError(f"fields not allowed in model input: {sorted(unknown)}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(inp)},
    ]


def prompt_sha256() -> str:
    """Hash of this file, the p1 module it reuses, and the output schema."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(Path(_p1.__file__).read_bytes())
    h.update(SCHEMA_PATH.read_bytes())
    return h.hexdigest()


def fingerprint() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "schema": SCHEMA_PATH.name,
    }
