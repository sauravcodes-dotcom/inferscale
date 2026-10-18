"""Prompt for the triage_v1 eval. Bump PROMPT_VERSION on any change to the text."""

from __future__ import annotations

import hashlib
from typing import Any

PROMPT_VERSION = "triage_v1.p1"

SYSTEM_PROMPT = """\
You are an on-call SRE triaging a production alert. Read the alert and logs, then output a \
single JSON object and nothing else (no prose, no markdown fences).

Fields:
- "severity": one of "SEV1","SEV2","SEV3","SEV4". Apply in order, first match wins:
  SEV1 = production and (customer-facing function fully down, or >=25% of customer requests \
failing, or customer data loss/corruption or wrong charges actively occurring, or a confirmed \
security breach).
  SEV2 = production and (2% to <25% of customer requests failing, or customer-facing timeouts, \
or a customer-facing outage projected within <=60 minutes, or customers seeing wrong/stale data, \
or a suspected credential compromise / unblocked attack).
  SEV3 = production with no or <2% customer impact and (reduced redundancy, internal tools/batch/\
CI/CD impaired, exhaustion projected in >60 min and <=24 h, attack fully blocked, or action \
needed within days).
  SEV4 = non-production (staging/dev, even if used for demos), warnings with exhaustion >24 h \
away, or informational signals with no impact.
  Unexplained traffic/order drops without errors: >=25% drop is SEV2, smaller is SEV3. Wrong \
charges that have stopped but are not remediated are SEV2; still happening is SEV1.
- "needs_page": true if and only if severity is SEV1 or SEV2.
- "category": the ROOT CAUSE (earliest causal event shown, not the symptom). One of:
  "resource_exhaustion" (our disk/memory/CPU/IO/pools/file descriptors/capacity exhausted),
  "network" (packet loss, link flaps, partitions, resets, resolver unreachable),
  "dependency" (a third-party/external service failing or slow),
  "config" (non-release config change or misconfiguration: flags, env vars, limits, TTLs, \
DNS records, lifecycle rules, Helm values, TLS certificate expiry/renewal),
  "deploy_regression" (a new release of code/image/chart/edge rules/CI template caused it),
  "auth" (credentials, keys, tokens, permissions: rotation, stale JWKS, RBAC/IAM denials, \
credential stuffing, leaked keys),
  "data" (integrity/consistency/durability not caused by a release: replication, corruption, \
duplicates, offset resets, stale caches, failed backups),
  "unknown" (no conclusive causal evidence).
  If two causes start at the same time use: deploy_regression > auth > config > \
resource_exhaustion > dependency > network > data.
- "component": the system owning the root cause. One of "api-gateway","auth-service",\
"payments","database","cache","message-queue","storage","kubernetes","cdn","ci-cd","dns",\
"unknown". If the cause is in an unlisted service, pick the listed component most directly \
affected; for an external dependency, pick our component that integrates with it.
- "summary": one sentence, at most 200 characters: what is broken and why.

Output exactly: {"severity": ..., "component": ..., "category": ..., "needs_page": ..., \
"summary": ...}"""


def user_message(inp: dict[str, Any]) -> str:
    parts = [
        f"ALERT TITLE: {inp['alert_title']}",
        f"ALERT BODY: {inp['alert_body']}",
        "LOG EXCERPT:",
        inp["log_excerpt"],
    ]
    if inp.get("metrics_snippet"):
        parts += ["METRICS:", inp["metrics_snippet"]]
    return "\n".join(parts)


def build_messages(inp: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(inp)},
    ]


def response_format(kind: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """OpenAI-style response_format for the gateway ('none' sends nothing)."""
    if kind == "none":
        return None
    if kind == "json_schema":
        clean = {k: v for k, v in schema.items() if not k.startswith("$")}
        return {
            "type": "json_schema",
            "json_schema": {"name": "triage_v1", "strict": True, "schema": clean},
        }
    raise ValueError(f"unknown response format {kind!r}")


def fingerprint() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
