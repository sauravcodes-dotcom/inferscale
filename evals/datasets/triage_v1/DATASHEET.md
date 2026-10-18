# Datasheet: triage_v1

## Summary
300 synthetic production-incident alerts, each labelled with a triage decision (severity,
owning component, root-cause category, whether to page, and a one-line summary). The dataset is
used to evaluate how well LLMs served by InferScale triage alerts.

| | |
|---|---|
| Version | triage_v1 (generator `draft_triage_v1/1.0`, seed 20261018) |
| Size | 300 items: 60 `dev` / 240 `test`, stratified by category |
| Files | `dev.jsonl`, `test.jsonl`, `schema.json` (label schema), `manifest.json` (sha256 + counts), `review.csv`, `LABELING.md` |
| License | CC-BY-4.0 |
| Status | **Draft. Human review by Saurav is pending.** Every item has `review_status: "draft"`. |

## Provenance
- **Claude-drafted (LLM-assisted).** Claude (an Anthropic model) wrote 68 scenario templates in
  `scripts/draft_triage_v1.py`. Each template has an alert, a realistic log excerpt and a label
  derived from `LABELING.md`. The script instantiates them with seeded random details (hosts,
  numbers, timestamps, impact levels). Running it again with the same seed reproduces the files
  byte for byte.
- **Human review (pending).** Saurav reviews every row in `review.csv`. `scripts/apply_review.py`
  merges the decisions back, sets `review_status: "reviewed"` and recomputes the manifest. Results
  on the draft data must not be reported as results on triage_v1.
- No production data, real company names, real hostnames, secrets or personal data. IPs come only
  from documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24). Hostnames use
  `*.internal.example` or `example.com/.org`. Access-key strings are placeholders (`AKEXAMPLE…`).

## Composition (draft)
- Category: resource_exhaustion 40, deploy_regression 40, network 39, dependency 39, config 39,
  auth 39, data 39, unknown 25.
- Severity: SEV1 46 (15 %), SEV2 91 (30 %), SEV3 102 (34 %), SEV4 61 (20 %).
- `component = unknown` on 16 items (5 %).
- 30 items (10 %) are deliberately ambiguous (`meta.ambiguous = true`, with `meta.ambiguity_note`).
- Log formats: nginx access/error, Kubernetes events, Postgres, JVM stack traces, Python
  tracebacks, systemd/kernel syslog, Redis, Kafka, RabbitMQ, CoreDNS, cloud audit JSON,
  structured app JSON, CI job logs, status-page and probe output.
- `manifest.json` has exact per-split counts.

## Item format
```json
{"id": "tri-0001",
 "input": {"alert_title": "...", "alert_body": "...", "log_excerpt": "5-15 lines", "metrics_snippet": "optional"},
 "label": {"severity": "SEV1|SEV2|SEV3|SEV4", "component": "...", "category": "...", "needs_page": true, "summary": "..."},
 "review_status": "draft|reviewed",
 "meta": {"scenario": "...", "ambiguous": false, "ambiguity_note": null, "generator": "..."}}
```
`meta` is for analysis only and must never be shown to the model under test.

## Intended use
- Comparing models, quantizations, prompts and decoding settings on structured alert triage
  served through InferScale (`evals/run_eval.py`).
- Measuring JSON and schema validity, per-field accuracy, category macro-F1 and paired
  differences (McNemar) between configurations.
- `dev` is for prompt iteration. `test` is for reported numbers only.

Not intended for: training models; evaluating real on-call processes; measuring safety or
security-detection capability.

## Known limitations
- **Synthetic and templated.** Items come from 68 templates with varied details. Many items share
  structure and phrasing (only 159 of 300 titles are unique), so scores may overstate
  generalisation to real alerts. A model could learn template cues, so don't fine-tune on test.
- **Single labeler.** Labels encode one guide written by the same LLM that drafted the items,
  followed by review from one person. There's no inter-annotator agreement measurement.
- **Clean evidence.** Most items contain unusually clear causal evidence. Real alerts are noisier
  and more often unknown.
- **Severity rules are opinionated.** Severity rules are fixed thresholds (25 %/2 % error rate,
  60 min/24 h exhaustion). Real organisations differ.
- **needs_page is derived from severity**, so it adds no independent signal beyond severity.
- English only. Fictional infrastructure (one storefront/payments company archetype).
