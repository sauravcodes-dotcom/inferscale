"""Loader for the PAG synthetic cloud-incident dataset (pinned file, normalized records).

Normalization:
- CAPM keys: record 526 uses lowercase 'corrective_actions'/'preventive_actions'; accept both.
- Action entries that are dicts {"Action", "Responsible_Team"} (records 68, 677, 867) are
  flattened to their "Action" string.
Exclusion: source indices 0-99 duplicate the Incident_IDs of 799-898 and carry byte-identical
CAPM copied from them, so they are excluded. The eval universe is indices 100-949 (850 records).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.dataset import ROOT

SOURCE: dict[str, str] = {
    "repo": "https://github.com/sancharichowdhuri2/PAG",
    "commit": "ea4860499ab2fd8b0f7d1ac835257b70bb6e4d3a",
    "file": "PAG-Synthetic-Cloud-Incident-Dataset.json",
    "sha256": "5a68a736d9ab53a9049daabda27818388812c734c2f8eba836d37279e88c7be7",
    "raw_url": "https://raw.githubusercontent.com/sancharichowdhuri2/PAG/"
    "ea4860499ab2fd8b0f7d1ac835257b70bb6e4d3a/PAG-Synthetic-Cloud-Incident-Dataset.json",
    "license": "none (no LICENSE file in the repository); do not redistribute",
}
PAG_FILE = ROOT / "data" / "pag" / SOURCE["file"]
N_RAW = 950
EXCLUDED_INDICES = range(0, 100)
EXCLUSION_REASON = (
    "duplicate Incident_ID of source index i+799 with byte-identical CAPM copied from it"
)

# Fields the model may see. Severity, Root_Cause_*, CAPM and the ticket link are never inputs.
INPUT_FIELDS = (
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


class PagDataError(ValueError):
    pass


@dataclass(frozen=True)
class PagRecord:
    source_index: int
    incident_id: str
    summary: str
    description: str
    severity: str
    detection_method: str
    impact_type: str
    impacted_customer_count: int
    region: str
    root_cause_team: str
    root_cause_type: str
    root_cause_description: str
    impact_start_time: str
    detected_time: str
    mitigation_time: str
    repeat_outage: bool
    capm_ticket_link: str
    corrective_actions: tuple[str, ...]
    preventive_actions: tuple[str, ...]

    def model_input(self) -> dict[str, Any]:
        """The only fields shown to the model (keys as in the source file)."""
        return {
            "Summary": self.summary,
            "Description": self.description,
            "Detection_Method": self.detection_method,
            "Impact_Type": self.impact_type,
            "Impacted_Customer_Count": self.impacted_customer_count,
            "Region": self.region,
            "Repeat_Outage": self.repeat_outage,
            "Impact_Start_Time": self.impact_start_time,
            "Detected_Time": self.detected_time,
            "Mitigation_Time": self.mitigation_time,
        }

    def reference(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "root_cause_type": self.root_cause_type,
            "root_cause_team": self.root_cause_team,
            "root_cause_description": self.root_cause_description,
            "corrective_actions": list(self.corrective_actions),
            "preventive_actions": list(self.preventive_actions),
        }


def _action(entry: Any, where: str) -> str:
    if isinstance(entry, str):
        text = entry
    elif isinstance(entry, dict) and isinstance(entry.get("Action"), str):
        text = entry["Action"]
    else:
        raise PagDataError(f"{where}: unsupported action entry {entry!r:.80}")
    text = text.strip()
    if not text:
        raise PagDataError(f"{where}: empty action")
    return text


def _actions(capm: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    for k in (key, key.lower()):
        if k in capm:
            lst = capm[k]
            if not isinstance(lst, list) or not lst:
                raise PagDataError(f"{where}: {k} must be a non-empty list")
            return tuple(_action(a, f"{where}.{k}[{i}]") for i, a in enumerate(lst))
    raise PagDataError(f"{where}: CAPM has no {key}")


def normalize(raw: dict[str, Any], index: int) -> PagRecord:
    where = f"record {index}"
    capm = raw.get("CAPM")
    if not isinstance(capm, dict):
        raise PagDataError(f"{where}: CAPM missing")
    try:
        return PagRecord(
            source_index=index,
            incident_id=str(raw["Incident_ID"]),
            summary=str(raw["Summary"]),
            description=str(raw["Description"]),
            severity=str(raw["Severity"]),
            detection_method=str(raw["Detection_Method"]),
            impact_type=str(raw["Impact_Type"]),
            impacted_customer_count=int(raw["Impacted_Customer_Count"]),
            region=str(raw["Region"]),
            root_cause_team=str(raw["Root_Cause_Team"]),
            root_cause_type=str(raw["Root_Cause_Type"]),
            root_cause_description=str(raw["Root_Cause_Description"]),
            impact_start_time=str(raw["Impact_Start_Time"]),
            detected_time=str(raw["Detected_Time"]),
            mitigation_time=str(raw["Mitigation_Time"]),
            repeat_outage=bool(raw["Repeat_Outage"]),
            capm_ticket_link=str(raw["CAPM_Ticket_Link"]),
            corrective_actions=_actions(capm, "Corrective_Actions", where),
            preventive_actions=_actions(capm, "Preventive_Actions", where),
        )
    except KeyError as e:
        raise PagDataError(f"{where}: missing field {e}") from None


def load_raw(path: Path = PAG_FILE, verify_sha: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `uv run python scripts/fetch_pag.py`")
    data = path.read_bytes()
    if verify_sha:
        digest = hashlib.sha256(data).hexdigest()
        if digest != SOURCE["sha256"]:
            raise PagDataError(f"{path}: sha256 {digest} != pinned {SOURCE['sha256']}")
    raw = json.loads(data)
    if not isinstance(raw, list):
        raise PagDataError("expected a JSON list of records")
    return raw


def load_records(
    path: Path = PAG_FILE,
    verify_sha: bool = True,
    raw: list[dict[str, Any]] | None = None,
) -> dict[str, PagRecord]:
    """Normalized records of the eval universe (exclusions applied), keyed by Incident_ID."""
    rows = raw if raw is not None else load_raw(path, verify_sha)
    out: dict[str, PagRecord] = {}
    for i, r in enumerate(rows):
        if i in EXCLUDED_INDICES:
            continue
        rec = normalize(r, i)
        if rec.incident_id in out:
            first = out[rec.incident_id].source_index
            raise PagDataError(f"duplicate Incident_ID {rec.incident_id} at {i} and {first}")
        out[rec.incident_id] = rec
    return out


def excluded(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"source_index": i, "incident_id": str(raw[i]["Incident_ID"]), "reason": EXCLUSION_REASON}
        for i in EXCLUDED_INDICES
        if i < len(raw)
    ]
