# triage_v1 labeling guide

Each item is one alert as an on-call engineer would see it: `alert_title`, `alert_body`,
a `log_excerpt` and sometimes a `metrics_snippet`. The label is the triage decision:
`severity`, `component`, `category`, `needs_page` and a one-line `summary`.

Label **only from what the item shows**. Don't assume facts that aren't in the text. Where the
rules below give an order, apply it top to bottom and stop at the first match.

## 1. severity

Work out the impact first, then apply the rules in order (first match wins). All thresholds are
**inclusive** at the boundary that names them.

| | Rule |
|---|---|
| **SEV1** | Production **and** any of: a customer-facing function fully unavailable; **≥ 25 %** of customer requests failing; customer data loss or corruption **actively occurring** (including customers being charged incorrectly right now); a **confirmed** security breach (unauthorized access or exfiltration in progress, or account takeover). |
| **SEV2** | Production **and** any of: **2 % – < 25 %** of customer requests failing; customer-facing latency high enough that requests time out (or p99 ≥ 3× baseline); a customer-facing outage **projected within ≤ 60 minutes** if nobody acts; customers seeing wrong or stale data; a **suspected** credential compromise or an attack that is not blocked. |
| **SEV3** | Production with **no or minimal (< 2 %)** customer-facing impact **and** any of: reduced redundancy; internal tools, batch jobs, CI/CD or reporting impaired; exhaustion projected in **> 60 minutes and ≤ 24 hours**; an attack fully blocked by existing controls; a problem needing action within days (for example a certificate expiring in < 7 days). |
| **SEV4** | Everything else: **non-production** environments (staging, dev, CI sandboxes), whatever their error rate; warnings with exhaustion projected **> 24 hours** away; informational or self-recovered signals with no impact. |

Clarifications:
- Environment comes from the `[production]` / `[staging]` / `[dev]` tag in the title or from the body.
  Staging counts as non-production **even when an external party is using it** (demos, partner
  testing), so it's SEV4.
- CI/CD failures that block deploys but don't affect running production are SEV3 when they block
  all or most deploys, and SEV4 when they're limited to one runner, sandbox or nightly job.
- If the body states an error rate, use it. Otherwise judge from the impact statement.
- Unexplained drops in traffic or orders **without** errors: a drop of **≥ 25 %** is SEV2 (it may be
  a hidden outage), and a smaller drop is SEV3.
- Latency counts as customer-facing degradation (SEV2) only when customer requests time out or
  p99 is ≥ 3× baseline. Slower but within those bounds is SEV3.
- Wrong charges that have **stopped** but aren't yet remediated count as "customers seeing wrong
  data", so SEV2. Wrong charges still happening are SEV1.

## 2. needs_page

`needs_page = true` **if and only if** severity is SEV1 or SEV2. There are no exceptions; this
field is derived from severity and must agree with it.

## 3. category

The category describes the **root cause**, not the symptom. To decide:

1. Find the **earliest causal event** shown (in the timeline, or stated in the body). The category
   of that event is the label. Example: a config change lowered a memory limit and then pods were
   OOMKilled, so the category is `config`, not `resource_exhaustion`.
2. If two candidate causes start **at the same time** (or the order can't be told), use this
   priority: `deploy_regression` > `auth` > `config` > `resource_exhaustion` > `dependency` >
   `network` > `data`.
3. If the evidence doesn't support any cause, use `unknown` (section 5).

| category | Use when the root cause is… |
|---|---|
| `resource_exhaustion` | Capacity of **our** resource exhausted or saturated: disk, memory/OOM, CPU, disk IO, connection pools, file descriptors, queue storage, node capacity. |
| `network` | Connectivity between components: packet loss, link/NIC flaps, partitions, connection resets, unreachable routes, DNS resolution failing **because of network or resolver reachability**. |
| `dependency` | A **third-party or external** service failing or slow (payment provider, SMS or email vendor, external IdP, CDN or DNS provider incident, public package registry, cloud object-storage API errors). Our own upstream services don't count. |
| `config` | A configuration change or misconfiguration that **isn't a code/image release**: feature flags, env vars, rate limits, TTLs, DNS records, lifecycle rules, Helm values, and **TLS certificate expiry or renewal misconfiguration** (including internal mTLS certificates). |
| `deploy_regression` | A new release of code, a container image, a chart or controller version, edge rules or a CI template introduced the problem. This includes schema migrations and data bugs shipped in a release. |
| `auth` | Authentication or authorization failures from credentials, keys, tokens or permissions: rotated or expired passwords and keys, stale JWKS, RBAC/IAM denials, credential stuffing, leaked keys. (Certificate expiry is `config`; a release that breaks auth is `deploy_regression`.) |
| `data` | Correctness, integrity or durability of data, **not caused by a release**: replication broken or lagging (when not caused by resource saturation), corruption/checksum failures, duplicates, offset resets causing reprocessing, stale or inconsistent caches, failed backups. |
| `unknown` | See section 5. |

## 4. component

`component` is the system that **owns the root cause**, the one whose team must act. Allowed values:

| component | Covers |
|---|---|
| `api-gateway` | Edge/API gateway and reverse proxies (nginx, routing, rate limiting, JWT verification at the edge). |
| `auth-service` | Login, token issuance, MFA, SSO integration. |
| `payments` | Payment API, ledger, payment workers and their integrations (PSPs, fraud scoring), including their configuration and credentials. |
| `database` | Relational databases (Postgres primaries and replicas), their storage and connectivity. |
| `cache` | Redis/memcached and application caches. |
| `message-queue` | Kafka, RabbitMQ, consumer groups and queue backlogs. |
| `storage` | Object storage buckets, NFS, backups and their permissions and lifecycle rules. |
| `kubernetes` | Cluster-level resources: nodes, scheduling, ingress controller. |
| `cdn` | CDN edge, edge rules, public TLS certificates served at the edge. |
| `ci-cd` | Build/deploy pipelines, runners, CI templates and CI service accounts. |
| `dns` | Name resolution: CoreDNS, authoritative DNS, DNS records. |
| `unknown` | The owning component can't be determined from the item. |

Rules:
- If the root cause sits in a service that isn't in the list (for example `catalog-api` or
  `orders-api`), pick the listed component **most directly affected**. Example: an unindexed query
  from a catalog release saturates the database, so the component is `database`.
- For `dependency`, use **our** component that integrates with the external service (for example,
  a PSP outage is `payments`).
- For `network` problems between an app and a listed component, use the listed component on the
  far side (for example, resets to the orders DB are `database`). For edge or upstream timeouts
  seen only at the gateway, use `api-gateway`.
- A problem with an application's own credentials (password rotated but pods still use the old
  one) belongs to that application.

## 5. When to use `unknown`

- `category = unknown` when the item shows symptoms but **no** causal evidence, or only weak hints
  pointing in different directions that are each inconclusive. Examples: bare 502s, a traffic
  drop with no errors, exit code 1 with no logs, generic 500s with healthy dependencies.
- `component = unknown` when the failing system can't be identified (for example a synthetic
  check failing with no upstream information, or an unowned host). If the alert clearly names a
  listed component (such as a `payments-worker` crashloop), use that component even when the
  category is `unknown`.
- Severity is still assigned from impact. Uncertainty about the cause doesn't lower or raise
  severity.

## 6. summary

One sentence, at most 200 characters, stating **what is broken and why** (the cause if known).
Optionally add the impact. No speculation beyond the evidence, and no secrets. The summary isn't
scored by exact match.

## 7. Ambiguous items

About 10 % of items are deliberately ambiguous: competing causes, an exact threshold, or
production-like use of a non-production environment. Label them by the rules above; the earliest
cause and the tie-break order make the answer unique. Their `meta.ambiguity_note` explains the
intended reading. Reviewers who disagree should record the correction and a note rather than
silently changing the rule.
