"""Draft the triage_v1 dataset (300 synthetic incident alerts) from Claude-authored scenarios.

Each scenario is a hand-written template (title, body, log excerpt in a realistic format,
label) that is instantiated with seeded random details (hosts, numbers, timestamps) at a
given severity. Labels follow evals/datasets/triage_v1/LABELING.md. Output is deterministic
for a given seed. Refuses to overwrite a dataset that has reviewed items unless --force.

Usage: uv run python scripts/draft_triage_v1.py [--seed N] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.dataset import TRIAGE_V1, write_jsonl, write_manifest

GENERATOR_VERSION = "draft_triage_v1/1.0"
DEFAULT_SEED = 20261018
OUT_DIR = TRIAGE_V1
N_ITEMS = 300
N_DEV = 60

CATEGORY_TARGETS = {
    "resource_exhaustion": 40,
    "network": 39,
    "dependency": 39,
    "config": 39,
    "deploy_regression": 40,
    "auth": 39,
    "data": 39,
    "unknown": 25,
}
SEV_SHARE = {1: 0.15, 2: 0.30, 3: 0.35, 4: 0.20}
# Ambiguous items per category (~10% overall), labelled by the guide's tie-break rules.
AMBIGUOUS_TARGETS = {
    "resource_exhaustion": 5,
    "config": 5,
    "deploy_regression": 4,
    "dependency": 4,
    "network": 3,
    "auth": 3,
    "data": 3,
    "unknown": 3,
}

Rng = random.Random

# ---------------------------------------------------------------------------
# Random detail helpers (documentation IP ranges, fictional *.internal.example hosts)
# ---------------------------------------------------------------------------

DOC_NETS = ("192.0.2", "198.51.100", "203.0.113")
REGIONS = ("eu-central", "us-east", "us-west", "ap-southeast")
ZONES = ("a", "b", "c")


def ip(rng: Rng) -> str:
    return f"{rng.choice(DOC_NETS)}.{rng.randint(2, 254)}"


def hexs(rng: Rng, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def pod(rng: Rng, svc: str) -> str:
    return f"{svc}-{hexs(rng, 9)}-{hexs(rng, 5)}"


def host(rng: Rng, role: str, env: str = "prod") -> str:
    return f"{role}-{env}-{rng.randint(1, 12):02d}.internal.example"


def start_time(rng: Rng) -> datetime:
    return datetime(
        2026,
        rng.randint(5, 9),
        rng.randint(1, 28),
        rng.randint(0, 23),
        rng.randint(0, 59),
        rng.randint(0, 59),
        rng.randint(0, 999) * 1000,
        tzinfo=UTC,
    )


class Clock:
    def __init__(self, rng: Rng, step: tuple[float, float] = (0.2, 20.0)) -> None:
        self.rng = rng
        self.t = start_time(rng)
        self.step = step

    def tick(self) -> datetime:
        self.t += timedelta(seconds=self.rng.uniform(*self.step))
        return self.t


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def pg_ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond // 1000:03d} UTC"


def nginx_err_ts(t: datetime) -> str:
    return t.strftime("%Y/%m/%d %H:%M:%S")


def nginx_acc_ts(t: datetime) -> str:
    return t.strftime("%d/%b/%Y:%H:%M:%S +0000")


def syslog_ts(t: datetime) -> str:
    return t.strftime("%b %d %H:%M:%S")


def kafka_ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S,") + f"{t.microsecond // 1000:03d}"


def redis_ts(t: datetime) -> str:
    return t.strftime("%d %b %Y %H:%M:%S.") + f"{t.microsecond // 1000:03d}"


_TS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r'"(?:ts|eventTime)": "(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3})Z"'),
        "%Y-%m-%dT%H:%M:%S.%f",
    ),
    (re.compile(r"^\[?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3})Z"), "%Y-%m-%dT%H:%M:%S.%f"),
    (re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) UTC"), "%Y-%m-%d %H:%M:%S.%f"),
    (re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\]"), "%Y-%m-%d %H:%M:%S,%f"),
    (re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)"), "%Y/%m/%d %H:%M:%S"),
    (re.compile(r"\[(\d\d/\w{3}/\d{4}:\d\d:\d\d:\d\d) \+0000\]"), "%d/%b/%Y:%H:%M:%S"),
    (re.compile(r"^\d+:M (\d\d \w{3} \d{4} \d\d:\d\d:\d\d\.\d{3})"), "%d %b %Y %H:%M:%S.%f"),
    (re.compile(r"^(\w{3} \d\d \d\d:\d\d:\d\d) "), "%Y %b %d %H:%M:%S"),
]


def line_ts(line: str) -> datetime | None:
    for pat, fmt in _TS_PATTERNS:
        m = pat.search(line)
        if m:
            raw = m.group(1)
            if fmt.startswith("%Y %b"):  # syslog has no year; the dataset is all 2026
                raw = "2026 " + raw
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
    return None


def order_logs(lines: list[str]) -> list[str]:
    """Sort timestamped lines chronologically; untimestamped lines stay attached to the
    preceding timestamped line (stack traces), or stay in place at the top (headers)."""
    blocks: list[tuple[datetime | None, int, list[str]]] = []
    for i, ln in enumerate(lines):
        t = line_ts(ln)
        if t is None and blocks:
            blocks[-1][2].append(ln)
        else:
            blocks.append((t, i, [ln]))
    if any(t is None for t, _, _ in blocks[1:]) or all(t is None for t, _, _ in blocks):
        return lines
    head = [b for b in blocks if b[0] is None]
    rest = sorted((b for b in blocks if b[0] is not None), key=lambda b: (b[0], b[1]))
    return [ln for b in head + rest for ln in b[2]]


def before_first(lines: list[str], rng: Rng, minutes: tuple[float, float] = (2, 20)) -> datetime:
    ts = [t for t in (line_ts(x) for x in lines) if t is not None]
    base = min(ts) if ts else start_time(rng)
    return base - timedelta(minutes=rng.uniform(*minutes))


def jlog(t: datetime, level: str, svc: str, msg: str, **kw: Any) -> str:
    return json.dumps({"ts": iso(t), "level": level, "service": svc, "msg": msg, **kw})


def audit(t: datetime, **kw: Any) -> str:
    return json.dumps({"eventTime": iso(t), **kw})


def k8s_events(rows: list[tuple[str, str, str, str]], rng: Rng) -> list[str]:
    out = [
        "LAST SEEN   TYPE      REASON              OBJECT                                   MESSAGE"
    ]
    for typ, reason, obj, msg in rows:
        seen = f"{rng.randint(1, 59)}s" if rng.random() < 0.6 else f"{rng.randint(1, 9)}m"
        out.append(f"{seen:<11} {typ:<9} {reason:<19} {obj:<40} {msg}")
    return out


def pad(
    lines: list[str], filler: Callable[[], str], rng: Rng, lo: int = 5, hi: int = 15
) -> list[str]:
    """Insert filler lines at random positions until the excerpt has lo..hi lines."""
    target = max(lo, min(hi, len(lines) + rng.randint(0, 4)))
    out = list(lines)
    while len(out) < target:
        out.insert(rng.randint(1, len(out)), filler())
    return out[:hi]


# ---------------------------------------------------------------------------
# Impact model: numbers that put an item unambiguously (or at the boundary) in a severity
# ---------------------------------------------------------------------------


@dataclass
class Impact:
    sev: int
    env: str
    err_pct: float
    eta_min: int
    users: str
    borderline: bool = False

    @property
    def prod(self) -> bool:
        return self.env == "production"


def impact(rng: Rng, sev: int, borderline: bool = False, prod_sev4: bool = False) -> Impact:
    if sev == 1:
        err = 25.0 if borderline else float(rng.randint(30, 97))
        return Impact(sev, "production", err, 0, "all customers", borderline)
    if sev == 2:
        err = 2.0 if borderline else round(rng.uniform(3, 22), 1)
        eta = 60 if borderline else rng.randint(15, 55)
        return Impact(
            sev, "production", err, eta, f"customers in {rng.choice(REGIONS)}", borderline
        )
    if sev == 3:
        err = round(rng.uniform(0.1, 1.6), 1)
        eta = 1440 if borderline else rng.randint(120, 1200)
        return Impact(sev, "production", err, eta, "no customer-facing impact", borderline)
    env = (
        rng.choice(("staging", "staging", "dev", "production"))
        if prod_sev4
        else rng.choice(("staging", "dev"))
    )
    return Impact(sev, env, 0.0, rng.randint(2 * 1440, 9 * 1440), "no impact", borderline)


def eta_text(minutes: int) -> str:
    if minutes == 1440:
        return "24 hours"
    if minutes > 1440:
        return f"{minutes / 1440:.1f} days"
    if minutes >= 120:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes} minutes"


def error_line(imp: Impact, what: str) -> str:
    if imp.sev == 1:
        return f"{imp.err_pct:g}% of {what} failing in production; {imp.users} affected."
    if imp.sev == 2:
        return f"{imp.err_pct:g}% of {what} failing ({imp.users}); remaining traffic served."
    if imp.sev == 3:
        return f"Error rate {imp.err_pct:g}% on internal/retried calls; {imp.users}."
    if imp.env != "production":
        return f"Environment: {imp.env}. No production impact."
    return "Production unaffected; warning-level signal only."


def capacity_line(imp: Impact, resource: str, pct_full: dict[int, int]) -> str:
    p = pct_full[imp.sev]
    if imp.sev == 1:
        return f"{resource} at 100%; writes are failing now and customer requests error out."
    if imp.sev == 4 and imp.env != "production":
        return f"Environment: {imp.env}. {resource} at {p}%. No production impact."
    return f"{resource} at {p}% and growing; projected full in {eta_text(imp.eta_min)}. {imp.users.capitalize()}."


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class Draft:
    title: str
    body: str
    logs: list[str]
    summary: str
    component: str
    category: str
    metrics: str | None = None


@dataclass
class Scenario:
    key: str
    category: str
    sevs: tuple[int, ...]
    build: Callable[[Rng, Impact], Draft]
    ambiguous: bool = False
    note: str = ""
    tags: list[str] = field(default_factory=list)


SCENARIOS: list[Scenario] = []
# Scenarios whose SEV4 text is a production warning with no impact (forecast > 24h, background
# noise, self-recovered). All other scenarios use a non-production environment for SEV4.
PROD_SEV4_OK = {
    "pg_disk_full",
    "redis_maxmemory",
    "kafka_logdir_full",
    "credential_stuffing",
    "nfs_not_responding",
    "object_store_slowdown",
    "pg_replication_lag",
}


def scenario(
    key: str,
    category: str,
    sevs: tuple[int, ...] = (1, 2, 3, 4),
    ambiguous: bool = False,
    note: str = "",
) -> Callable[[Callable[[Rng, Impact], Draft]], Callable[[Rng, Impact], Draft]]:
    def deco(fn: Callable[[Rng, Impact], Draft]) -> Callable[[Rng, Impact], Draft]:
        SCENARIOS.append(Scenario(key, category, sevs, fn, ambiguous, note))
        return fn

    return deco


def envtag(imp: Impact) -> str:
    return {"production": "prod", "staging": "stg", "dev": "dev"}[imp.env]


# --- resource_exhaustion ----------------------------------------------------


@scenario("pg_disk_full", "resource_exhaustion")
def _pg_disk(rng: Rng, imp: Impact) -> Draft:
    h = host(rng, "pg", envtag(imp))
    c = Clock(rng)
    pid = rng.randint(1000, 60000)
    pct = {1: 100, 2: rng.randint(96, 99), 3: rng.randint(88, 93), 4: rng.randint(78, 85)}
    lines: list[str] = []
    if imp.sev == 1:
        lines += [
            f'{pg_ts(c.tick())} [{pid}] ERROR:  could not extend file "base/16384/{rng.randint(20000, 90000)}": No space left on device',
            f"{pg_ts(c.tick())} [{pid}] HINT:  Check free disk space.",
            f'{pg_ts(c.tick())} [{pid + 3}] PANIC:  could not write to file "pg_wal/xlogtemp.{pid + 3}": No space left on device',
            f"{pg_ts(c.tick())} [1] LOG:  WAL writer process (PID {pid + 3}) was terminated by signal 6: Aborted",
            f"{pg_ts(c.tick())} [1] LOG:  terminating any other active server processes",
        ]
    else:
        lines += [
            f"{pg_ts(c.tick())} [{pid}] LOG:  checkpoint complete: wrote {rng.randint(2000, 9000)} buffers",
            f"{pg_ts(c.tick())} [{pid}] WARNING:  archive_command failed with exit code 1, WAL segments accumulating",
            f"{pg_ts(c.tick())} [{pid}] LOG:  pg_wal size {rng.randint(40, 300)} GB, archive backlog {rng.randint(500, 5000)} segments",
        ]
    lines = pad(
        lines,
        lambda: (
            f"{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  duration: {rng.uniform(1, 90):.3f} ms  statement: SELECT 1"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Disk usage critical on {h} /var/lib/postgresql",
        body=capacity_line(imp, "/var/lib/postgresql volume", pct)
        + " Primary database for orders.",
        logs=lines,
        metrics=f'node_filesystem_avail_bytes{{mount="/var/lib/postgresql"}} = {max(0, 100 - pct[imp.sev]) * 21:.0f}GiB',
        summary=(
            "Postgres primary out of disk; writes failing."
            if imp.sev == 1
            else f"Postgres data volume at {pct[imp.sev]}%, WAL archive backlog filling disk."
        ),
        component="database",
        category="resource_exhaustion",
    )


@scenario("jvm_oom_payments", "resource_exhaustion")
def _jvm_oom(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 3))
    n = rng.randint(6, 12)
    bad = {1: n, 2: max(2, n // 3), 3: 1, 4: 1}[imp.sev]
    p = pod(rng, "payments-api")
    lines = [
        f'{iso(c.tick())} Exception in thread "http-nio-8080-exec-{rng.randint(1, 200)}" java.lang.OutOfMemoryError: Java heap space',
        "\tat java.base/java.util.Arrays.copyOf(Arrays.java:3537)",
        "\tat java.base/java.lang.AbstractStringBuilder.ensureCapacityInternal(AbstractStringBuilder.java:228)",
        "\tat com.example.payments.ledger.StatementExporter.render(StatementExporter.java:214)",
        "\tat com.example.payments.api.StatementController.export(StatementController.java:88)",
        f"{iso(c.tick())} [GC (Allocation Failure) {rng.randint(3800, 4090)}M->{rng.randint(3800, 4090)}M(4096M), {rng.uniform(1, 9):.4f} secs]",
        f"{iso(c.tick())} kubelet: container payments-api in pod {p} OOMKilled (exit code 137), restart count {rng.randint(3, 40)}",
    ]
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} [GC (Allocation Failure) {rng.randint(3000, 4000)}M->{rng.randint(2500, 3900)}M(4096M), {rng.uniform(0.1, 2):.4f} secs]"
        ),
        rng,
        7,
        13,
    )
    return Draft(
        title=f"[{imp.env}] payments-api pods restarting: OOMKilled ({bad}/{n} replicas)",
        body=error_line(imp, "checkout payment requests")
        + f" Heap usage pinned at limit on {bad} of {n} replicas.",
        logs=lines,
        metrics=f'jvm_memory_used_bytes{{area="heap"}} / max = 0.{rng.randint(96, 99)}',
        summary=f"payments-api JVM heap exhaustion (OOMKilled) on {bad}/{n} replicas during statement export.",
        component="payments",
        category="resource_exhaustion",
    )


@scenario("hikari_pool_auth", "resource_exhaustion", (1, 2, 3))
def _hikari(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 2))
    lines = [
        jlog(
            c.tick(),
            "warn",
            "auth-service",
            "HikariPool-1 - Thread starvation or clock leap detected",
        ),
        jlog(
            c.tick(),
            "error",
            "auth-service",
            "HikariPool-1 - Connection is not available, request timed out after 30000ms.",
            pool={"active": 50, "idle": 0, "waiting": rng.randint(80, 400), "max": 50},
        ),
        jlog(
            c.tick(),
            "error",
            "auth-service",
            "POST /oauth/token failed",
            status=500,
            latency_ms=30012,
        ),
    ]
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "error",
            "auth-service",
            "Connection is not available, request timed out after 30000ms.",
            pool={"active": 50, "idle": 0},
        ),
        rng,
        6,
        12,
    )
    return Draft(
        title="auth-service: DB connection pool exhausted (HikariPool-1 active=50/50)",
        body=error_line(imp, "login and token requests") + " Database itself reports normal load.",
        logs=lines,
        metrics=f"hikaricp_connections_pending = {rng.randint(80, 400)}; pg cpu = {rng.randint(15, 35)}%",
        summary="auth-service DB connection pool exhausted (50/50 active), token requests timing out.",
        component="auth-service",
        category="resource_exhaustion",
    )


@scenario("redis_maxmemory", "resource_exhaustion")
def _redis_mem(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 1))
    pid = rng.randint(1, 40)
    lines = [
        f"{pid}:M {redis_ts(c.tick())} # WARNING: used_memory {rng.uniform(15.6, 16):.2f}G exceeds maxmemory 16.00G (policy noeviction)"
    ]
    if imp.sev <= 2:
        lines.append(
            f"{pid}:M {redis_ts(c.tick())} # -OOM command not allowed when used memory > 'maxmemory'."
        )
    lines = pad(
        lines,
        lambda: (
            f"{pid}:M {redis_ts(c.tick())} * {rng.randint(1, 100)} changes in {rng.randint(60, 300)} seconds. Saving..."
        ),
        rng,
    )
    pct = {1: 100, 2: 99, 3: rng.randint(90, 95), 4: rng.randint(80, 88)}
    return Draft(
        title=f"[{imp.env}] cache-sessions Redis memory {pct[imp.sev]}% of maxmemory",
        body=(
            error_line(imp, "session reads/writes")
            if imp.sev <= 2
            else capacity_line(imp, "Redis memory", pct)
        ),
        logs=lines,
        metrics=f"redis_memory_used_bytes / redis_memory_max_bytes = {pct[imp.sev] / 100:.2f}",
        summary="Session Redis hit maxmemory with noeviction; writes rejected with OOM."
        if imp.sev <= 2
        else f"Session Redis memory at {pct[imp.sev]}% of maxmemory (noeviction policy).",
        component="cache",
        category="resource_exhaustion",
    )


@scenario("kafka_logdir_full", "resource_exhaustion", (2, 3, 4))
def _kafka_disk(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng)
    b = rng.randint(1, 6)
    pct = {2: rng.randint(95, 98), 3: rng.randint(86, 92), 4: rng.randint(75, 84)}
    lines = [
        f"[{kafka_ts(c.tick())}] WARN [Log partition=events-{rng.randint(0, 47)}, dir=/var/lib/kafka/data] Log retention bytes not reached; segment deletion skipped (kafka.log.UnifiedLog)",
        f"[{kafka_ts(c.tick())}] INFO [LogCleaner] Log cleaner thread 0 cleaned 0 bytes in {rng.randint(1, 50)} ms (kafka.log.LogCleaner)",
        f"[{kafka_ts(c.tick())}] WARN [ReplicaManager broker={b}] Disk usage on /var/lib/kafka/data is {pct[imp.sev]}% (kafka.server.ReplicaManager)",
    ]
    lines = pad(
        lines,
        lambda: (
            f"[{kafka_ts(c.tick())}] INFO [ProducerStateManager partition=events-{rng.randint(0, 47)}] Wrote producer snapshot (kafka.log.ProducerStateManager)"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Kafka broker {b} log dir {pct[imp.sev]}% full",
        body=capacity_line(imp, f"Broker {b} data disk", pct)
        + " Topic 'events' retention set to 14d.",
        logs=lines,
        summary=f"Kafka broker {b} data disk at {pct[imp.sev]}%, retention not freeing space.",
        component="message-queue",
        category="resource_exhaustion",
    )


@scenario("k8s_node_pressure", "resource_exhaustion")
def _node_pressure(rng: Rng, imp: Impact) -> Draft:
    nodes = rng.randint(2, 5) if imp.sev <= 2 else 1
    rows = []
    for _ in range(nodes):
        node = f"node/worker-{rng.randint(10, 99)}"
        rows.append(("Warning", "EvictionThresholdMet", node, "Attempting to reclaim memory"))
        rows.append(
            (
                "Normal",
                "NodeHasInsufficientMemory",
                node,
                "Node status is now: NodeHasInsufficientMemory",
            )
        )
    rows.append(
        (
            "Warning",
            "Evicted",
            f"pod/{pod(rng, 'checkout-web')}",
            "The node was low on resource: memory. Threshold quantity: 100Mi, available: 64Mi.",
        )
    )
    rows.append(
        (
            "Warning",
            "FailedScheduling",
            f"pod/{pod(rng, 'checkout-web')}",
            f"0/{rng.randint(8, 30)} nodes are available: {nodes} Insufficient memory.",
        )
    )
    lines = k8s_events(rows, rng)
    lines = pad(
        lines,
        lambda: k8s_events(
            [
                (
                    "Normal",
                    "Pulled",
                    f"pod/{pod(rng, 'search-api')}",
                    "Container image already present on machine",
                )
            ],
            rng,
        )[1],
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Kubernetes nodes under memory pressure, pods evicted ({nodes} node(s))",
        body=error_line(imp, "storefront requests") + " Cluster autoscaler at max node count.",
        logs=lines,
        summary=f"{nodes} Kubernetes node(s) out of memory at autoscaler max; pods evicted and unschedulable.",
        component="kubernetes",
        category="resource_exhaustion",
    )


@scenario("ci_runner_disk", "resource_exhaustion", (3, 4))
def _ci_disk(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 5))
    r = f"runner-{rng.randint(1, 30):02d}"
    lines = [
        f"[{iso(c.tick())}] Running on {r} (docker executor)",
        f"[{iso(c.tick())}] Step 3/11 : RUN npm ci",
        f"[{iso(c.tick())}] npm ERR! code ENOSPC",
        f"[{iso(c.tick())}] npm ERR! syscall write",
        f"[{iso(c.tick())}] npm ERR! errno -28 nospc ENOSPC: no space left on device, write",
        f"[{iso(c.tick())}] ERROR: Job failed: exit code 228",
    ]
    lines = pad(lines, lambda: f"[{iso(c.tick())}] Using cached layer {hexs(rng, 12)}", rng)
    scope = "all main-branch builds" if imp.sev == 3 else "builds on one runner"
    return Draft(
        title=f"CI: jobs failing with ENOSPC on {r}",
        body=f"{scope.capitalize()} failing; deploys blocked. No production impact."
        if imp.sev == 3
        else f"Intermittent job failures on {r}; retries pass on other runners. No production impact.",
        logs=lines,
        summary=f"CI runner {r} out of disk (ENOSPC); {scope} failing.",
        component="ci-cd",
        category="resource_exhaustion",
    )


@scenario("nginx_fd_exhaustion", "resource_exhaustion", (1, 2, 3))
def _nginx_fd(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 1))
    w = rng.randint(1000, 9000)
    lines = [
        f"{nginx_err_ts(c.tick())} [crit] {w}#{w}: accept4() failed (24: Too many open files)"
        for _ in range(3)
    ]
    lines.append(
        f"{nginx_err_ts(c.tick())} [alert] {w}#{w}: *{rng.randint(10**6, 10**7)} socket() failed (24: Too many open files) while connecting to upstream"
    )
    lines = pad(
        lines,
        lambda: (
            f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /v1/catalog HTTP/1.1" 200 {rng.randint(200, 9000)} "-" "app/5.{rng.randint(0, 9)}"'
        ),
        rng,
    )
    return Draft(
        title="api-gateway edge nginx: Too many open files",
        body=error_line(imp, "API requests")
        + f" worker_rlimit_nofile=1024 on gw-{rng.randint(1, 9)}.",
        logs=lines,
        summary="api-gateway nginx workers out of file descriptors (EMFILE); connections refused.",
        component="api-gateway",
        category="resource_exhaustion",
    )


# --- network ----------------------------------------------------------------


@scenario("coredns_timeouts", "network")
def _coredns(rng: Rng, imp: Impact) -> Draft:
    up = ip(rng)
    lines = [
        f"[ERROR] plugin/errors: 2 payments-db.internal.example. A: read udp {ip(rng)}:{rng.randint(30000, 60000)}->{up}:53: i/o timeout"
        for _ in range(rng.randint(3, 6))
    ]
    lines.append(
        f'[WARNING] plugin/health: Local health request to "http://:8080/health" took more than 1s: {rng.uniform(1, 4):.3f}s'
    )
    lines = pad(
        lines,
        lambda: (
            f'[INFO] {ip(rng)}:{rng.randint(30000, 60000)} - {rng.randint(1, 65000)} "A IN search.internal.example. udp 45 false 512" NOERROR qr,aa,rd 80 0.000{rng.randint(100, 999)}s'
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] CoreDNS upstream i/o timeouts, SERVFAIL rate rising",
        body=error_line(imp, "service-to-service name lookups")
        + f" Upstream resolver {up} unreachable intermittently.",
        logs=lines,
        metrics=f'coredns_dns_responses_total{{rcode="SERVFAIL"}} rate = {rng.randint(5, 900)}/s',
        summary="CoreDNS forwarding to upstream resolver timing out; in-cluster lookups SERVFAIL.",
        component="dns",
        category="network",
    )


@scenario("az_packet_loss", "network", (1, 2, 3))
def _packet_loss(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 2))
    z = rng.choice(ZONES)
    lines = []
    for _ in range(rng.randint(3, 6)):
        lines.append(
            f'{nginx_err_ts(c.tick())} [error] {rng.randint(100, 9000)}#0: *{rng.randint(10**5, 10**7)} upstream timed out (110: Connection timed out) while connecting to upstream, client: {ip(rng)}, server: api.example.com, request: "GET /v2/orders HTTP/1.1", upstream: "http://{ip(rng)}:8080/v2/orders"'
        )
    lines = pad(
        lines,
        lambda: (
            f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /v2/orders HTTP/1.1" 504 160 "-" "app/5.{rng.randint(0, 9)}" rt=10.00{rng.randint(0, 9)}'
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] api-gateway upstream timeouts to zone {z}",
        body=error_line(imp, "API requests")
        + f" Cross-zone packet loss {rng.randint(8, 40)}% between zone {z} and other zones; app hosts healthy.",
        logs=lines,
        metrics=f'probe_packet_loss_ratio{{src_zone!="{z}",dst_zone="{z}"}} = 0.{rng.randint(8, 40):02d}',
        summary=f"Cross-zone packet loss to zone {z}; api-gateway upstream connections timing out.",
        component="api-gateway",
        category="network",
    )


@scenario("cdn_origin_unreachable", "network", (1, 2, 3))
def _cdn_origin(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 1))
    pops = rng.randint(1, 4) if imp.sev >= 2 else rng.randint(20, 60)
    lines = [
        json.dumps(
            {
                "ts": iso(c.tick()),
                "pop": f"pop-{rng.choice('abcdefghk')}{rng.randint(1, 9)}",
                "status": 502,
                "origin_status": None,
                "error": "origin_connect_timeout",
                "origin_ip": ip(rng),
                "path": "/static/app.js",
            }
        )
        for _ in range(rng.randint(3, 6))
    ]
    lines = pad(
        lines,
        lambda: json.dumps(
            {
                "ts": iso(c.tick()),
                "pop": "pop-m2",
                "status": 200,
                "cache": "HIT",
                "path": "/img/logo.svg",
            }
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] CDN 502s: origin connect timeouts from {pops} PoP(s)",
        body=error_line(imp, "page loads")
        + " Origin health checks from inside the VPC pass; failure is on the PoP-to-origin path.",
        logs=lines,
        summary=f"CDN PoPs ({pops}) cannot reach origin (connect timeouts); cache misses return 502.",
        component="cdn",
        category="network",
    )


@scenario("db_conn_reset_peering", "network", (1, 2, 3))
def _db_reset(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 3))
    lines = [
        jlog(
            c.tick(),
            "error",
            "orders-api",
            "query failed",
            error="read tcp: connection reset by peer",
            db="orders-primary",
            peer=f"{ip(rng)}:5432",
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines.append(
        jlog(
            c.tick(),
            "warn",
            "netmon",
            "vpc peering link flapping",
            link="pcx-app-data",
            state="down",
            drops=rng.randint(500, 9000),
        )
    )
    lines.append(
        jlog(
            c.tick(), "warn", "netmon", "vpc peering link flapping", link="pcx-app-data", state="up"
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "orders-api",
            "request ok",
            path="/v1/orders",
            latency_ms=rng.randint(10, 80),
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Connection resets between app subnet and orders database",
        body=error_line(imp, "order reads/writes")
        + " Database CPU, memory and connections normal.",
        logs=lines,
        summary="VPC peering link to database subnet flapping; DB connections reset by peer.",
        component="database",
        category="network",
    )


@scenario("rabbitmq_partition", "network", (1, 2, 3))
def _rmq(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.1, 5))
    n1, n2 = f"rabbit@mq-{rng.randint(1, 3)}", f"rabbit@mq-{rng.randint(4, 6)}"
    lines = [
        f"{iso(c.tick())} [error] <0.{rng.randint(100, 999)}.0> ** Node {n2} not responding **",
        f"{iso(c.tick())} [error] <0.{rng.randint(100, 999)}.0> ** Removing (timedout) connection **",
        f"{iso(c.tick())} [error] <0.{rng.randint(100, 999)}.0> Partial partition detected: * We saw DOWN from {n2} * We can still see {n1}",
        f"{iso(c.tick())} [warning] <0.{rng.randint(100, 999)}.0> Cluster partition_handling=pause_minority; pausing",
    ]
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} [info] <0.{rng.randint(100, 999)}.0> accepting AMQP connection {ip(rng)}:{rng.randint(30000, 60000)}"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] RabbitMQ network partition between {n1} and {n2}",
        body=error_line(imp, "async order notifications")
        + " net_tick_timeout exceeded; switch port errors on rack 4.",
        logs=lines,
        summary=f"RabbitMQ cluster partition ({n1}/{n2}) from network loss; minority paused.",
        component="message-queue",
        category="network",
    )


@scenario("nfs_not_responding", "network", (2, 3, 4))
def _nfs(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 30))
    h = host(rng, "nfs", envtag(imp))
    lines = [
        f"{syslog_ts(c.tick())} app-{rng.randint(1, 9)} kernel: nfs: server {h} not responding, still trying"
        for _ in range(rng.randint(2, 4))
    ]
    lines.append(f"{syslog_ts(c.tick())} app-{rng.randint(1, 9)} kernel: nfs: server {h} OK")
    lines.append(
        f"{syslog_ts(c.tick())} app-{rng.randint(1, 9)} kernel: bond0: link status down for interface eth1, disabling it in 200 ms"
    )
    lines = pad(
        lines,
        lambda: (
            f"{syslog_ts(c.tick())} app-{rng.randint(1, 9)} systemd[1]: Started Session {rng.randint(100, 999)} of user deploy."
        ),
        rng,
    )
    body = {
        2: f"Uploads stalling for {imp.users}; {imp.err_pct:g}% of upload requests time out.",
        3: "Report generation jobs stalling; no customer-facing impact.",
        4: f"Environment: {imp.env}. Brief stalls, recovered.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] NFS server {h} not responding (intermittent)",
        body=body + " NIC bond on client hosts flapping.",
        logs=lines,
        summary=f"NFS mounts from {h} stalling due to NIC/bond link flaps on client hosts.",
        component="storage",
        category="network",
    )


# --- dependency -------------------------------------------------------------


@scenario("psp_503", "dependency", (1, 2, 3))
def _psp(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 2))
    lines = [
        jlog(
            c.tick(),
            "error",
            "payments",
            "PSP authorize failed",
            provider="acquirer-a",
            http_status=503,
            body="upstream maintenance",
            latency_ms=rng.randint(200, 3000),
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines.append(
        jlog(
            c.tick(),
            "warn",
            "payments",
            "circuit breaker OPEN",
            provider="acquirer-a",
            failure_rate=round(rng.uniform(0.3, 0.99), 2),
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "payments",
            "PSP authorize ok",
            provider="acquirer-b",
            latency_ms=rng.randint(100, 400),
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Payment authorizations failing: acquirer-a returning 503",
        body=error_line(imp, "card authorizations")
        + " Provider status page reports 'degraded performance'. Failover to acquirer-b covers some card types.",
        logs=lines,
        summary="External payment provider acquirer-a returning 503; card authorizations failing.",
        component="payments",
        category="dependency",
    )


@scenario("sms_provider_timeouts", "dependency", (2, 3))
def _sms(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.1, 4))
    lines = [
        jlog(
            c.tick(),
            "error",
            "auth-service",
            "MFA SMS send failed",
            provider="sms-gw",
            error="context deadline exceeded",
            timeout_ms=5000,
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines = pad(
        lines,
        lambda: jlog(c.tick(), "info", "auth-service", "MFA TOTP verified", method="totp"),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] MFA SMS delivery failing via sms-gw",
        body=(
            f"{imp.err_pct:g}% of logins requiring SMS MFA cannot complete; TOTP users unaffected."
            if imp.sev == 2
            else "SMS MFA delayed by up to 2 minutes; logins eventually succeed."
        ),
        logs=lines,
        summary="Third-party SMS gateway timing out; SMS MFA codes not delivered.",
        component="auth-service",
        category="dependency",
    )


@scenario("object_store_slowdown", "dependency", (2, 3, 4))
def _s3(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 3))
    lines = [
        jlog(
            c.tick(),
            "warn",
            "media-store",
            "PutObject retry",
            code="SlowDown",
            http_status=503,
            bucket=f"media-{envtag(imp)}",
            attempt=a,
        )
        for a in range(1, rng.randint(3, 6))
    ]
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "media-store",
            "GetObject ok",
            bucket=f"media-{envtag(imp)}",
            latency_ms=rng.randint(20, 120),
        ),
        rng,
    )
    body = {
        2: f"{imp.err_pct:g}% of image uploads failing after retries; provider status page shows elevated error rates.",
        3: "Upload latency p99 1.4s (baseline 800ms); uploads succeed; provider incident acknowledged.",
        4: f"Environment: {imp.env}. Sporadic SlowDown responses from object storage provider.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Object storage provider returning SlowDown (503)",
        body=body,
        logs=lines,
        summary="Object storage provider throttling/erroring (SlowDown 503) on media uploads.",
        component="storage",
        category="dependency",
    )


@scenario("fraud_api_latency", "dependency", (2, 3))
def _fraud(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 2))
    lines = [
        jlog(
            c.tick(),
            "warn",
            "payments",
            "fraud score call slow",
            vendor="riskscore",
            latency_ms=rng.randint(2500, 9000),
            timeout_ms=3000,
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines.append(
        jlog(
            c.tick(),
            "error",
            "payments",
            "checkout aborted: fraud score timeout",
            order_id=f"ord_{hexs(rng, 8)}",
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "payments",
            "fraud score ok",
            vendor="riskscore",
            latency_ms=rng.randint(80, 300),
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] riskscore API latency above timeout",
        body=error_line(imp, "checkouts")
        if imp.sev == 2
        else "Fraud-score fallback rule engaged; checkouts succeed with manual review flag.",
        logs=lines,
        summary="External fraud-scoring vendor latency exceeding 3s timeout; checkouts aborted or degraded.",
        component="payments",
        category="dependency",
    )


@scenario("cdn_provider_incident", "dependency", (1, 2, 3))
def _cdn_vendor(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 60))
    lines = [
        f"{iso(c.tick())} statuspage: Investigating - elevated 5xx errors in {rng.choice(REGIONS)} edge locations",
        f"{iso(c.tick())} statuspage: Identified - a configuration push on the provider network is being rolled back",
    ]
    lines += [
        json.dumps(
            {
                "ts": iso(c.tick()),
                "pop": f"pop-x{rng.randint(1, 9)}",
                "status": 503,
                "error": "edge_internal",
                "cache": "ERROR",
            }
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines,
        lambda: json.dumps({"ts": iso(c.tick()), "pop": "pop-y1", "status": 200, "cache": "HIT"}),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] CDN edge 503s (provider incident declared)",
        body=error_line(imp, "static asset and page requests")
        + " Origin healthy; errors generated at provider edge.",
        logs=lines,
        summary="CDN provider incident causing edge 503s; origin healthy.",
        component="cdn",
        category="dependency",
    )


@scenario("pkg_registry_outage", "dependency", (3, 4))
def _registry(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 5))
    lines = [
        f"[{iso(c.tick())}] Step 4/12 : RUN pip install -r requirements.txt",
        f"[{iso(c.tick())}] WARNING: Retrying (Retry(total=4)) after connection broken by 'ReadTimeoutError(\"HTTPSConnectionPool(host='pkgs.example.org', port=443): Read timed out.\")'",
        f"[{iso(c.tick())}] ERROR: Could not find a version that satisfies the requirement httpx==0.28.1",
        f"[{iso(c.tick())}] ERROR: Job failed: exit code 1",
    ]
    lines = pad(lines, lambda: f"[{iso(c.tick())}] Using cached layer {hexs(rng, 12)}", rng)
    return Draft(
        title="CI: dependency installs timing out against public package index",
        body=(
            "All Python builds failing; deploys blocked. No production impact."
            if imp.sev == 3
            else "Some nightly builds failed; retried successfully. No production impact."
        ),
        logs=lines,
        summary="Public package registry unreachable/slow; CI dependency installs failing.",
        component="ci-cd",
        category="dependency",
    )


@scenario("dns_provider_outage", "dependency", (1, 2, 3))
def _dns_vendor(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 20))
    lines = [
        f"{iso(c.tick())} probe dig @ns{rng.randint(1, 4)}.dns-vendor.example api.example.com A -> ;; connection timed out; no servers could be reached"
        for _ in range(rng.randint(2, 4))
    ]
    lines.append(
        f"{iso(c.tick())} statuspage dns-vendor: Major outage - authoritative DNS queries failing in multiple regions"
    )
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} probe dig @ns{rng.randint(1, 4)}.dns-vendor.example www.example.com A -> 203.0.113.10 (cached)"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Authoritative DNS provider not answering",
        body=error_line(imp, "new client connections (resolver cache expiring)")
        + " Our zone config unchanged in 30 days.",
        logs=lines,
        summary="Third-party authoritative DNS provider outage; public names failing to resolve.",
        component="dns",
        category="dependency",
    )


# --- config -----------------------------------------------------------------


@scenario("flag_misroute", "config", (1, 2, 3))
def _flag(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 10))
    flag = rng.choice(["route_v2_orders", "edge_canary_pct", "new_pricing_backend"])
    pct = {1: 100, 2: rng.randint(10, 30), 3: 1}[imp.sev]
    lines = [
        audit(
            c.tick(),
            actor=f"user:{rng.choice(['ops-a', 'dev-b', 'sre-c'])}",
            action="flag.update",
            flag=flag,
            old="0",
            new=str(pct),
            env="production",
        )
    ]
    lines += [
        f'{nginx_err_ts(c.tick())} [error] {rng.randint(100, 9000)}#0: *{rng.randint(10**5, 10**7)} no live upstreams while connecting to upstream, upstream: "orders_v2"'
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines,
        lambda: (
            f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /v2/orders HTTP/1.1" 502 157 "-" "app/5.{rng.randint(0, 9)}"'
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] 502s on /v2/orders after flag change {flag}",
        body=error_line(imp, "order API requests")
        + f" Flag {flag} set to {pct}% routes traffic to the orders_v2 pool, which has 0 instances.",
        logs=lines,
        summary=f"Feature flag {flag} routes {pct}% of traffic to empty orders_v2 upstream; 502s.",
        component="api-gateway",
        category="config",
    )


@scenario("tls_cert_expiry", "config", (1, 3, 4))
def _cert(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 30))
    days = {1: 0, 3: rng.randint(2, 6), 4: rng.randint(20, 40)}[imp.sev]
    if imp.sev == 1:
        lines = [
            f"{iso(c.tick())} probe https://shop.example.com: x509: certificate has expired or is not yet valid: current time {iso(c.t)} is after {iso(c.t - timedelta(minutes=rng.randint(5, 50)))}"
            for _ in range(rng.randint(2, 4))
        ]
        body = "Browsers show certificate error on all storefront pages; checkout unusable."
    else:
        lines = [
            f"{iso(c.tick())} cert-check shop.example.com notAfter={(c.t + timedelta(days=days)).strftime('%b %d %H:%M:%S %Y GMT')} days_left={days}"
        ]
        lines.append(
            f"{iso(c.tick())} cert-renewer: renewal skipped: no matching issuer configured for shop.example.com"
        )
        body = f"Edge certificate expires in {days} days; auto-renewal misconfigured. No current impact."
    lines = pad(
        lines,
        lambda: f"{iso(c.tick())} cert-check www.example.com days_left={rng.randint(45, 80)}",
        rng,
    )
    return Draft(
        title=(
            "Storefront TLS certificate EXPIRED"
            if imp.sev == 1
            else f"TLS certificate for shop.example.com expires in {days} days"
        ),
        body=body,
        logs=lines,
        summary=(
            "Storefront edge TLS cert expired; renewal issuer misconfigured."
            if imp.sev == 1
            else f"Edge TLS cert expires in {days} days; auto-renewal issuer not configured."
        ),
        component="cdn",
        category="config",
    )


@scenario("wrong_db_host_env", "config", (1, 2, 3))
def _envvar(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 10))
    old = host(rng, "pg-legacy")
    lines = [
        audit(
            c.tick(),
            actor="pipeline:config-sync",
            action="configmap.update",
            object="payments-env",
            key="DB_HOST",
            new=old,
        )
    ]
    lines += [
        jlog(
            c.tick(),
            "error",
            "payments",
            "db connect failed",
            host=old,
            error="dial tcp: connect: connection refused",
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines, lambda: jlog(c.tick(), "info", "payments", "healthcheck", status="degraded"), rng
    )
    n = {1: "all", 2: "4 of 12", 3: "1 of 12"}[imp.sev]
    return Draft(
        title=f"[{imp.env}] payments pods failing DB connect after config sync",
        body=error_line(imp, "payment requests")
        + f" {n} pods restarted with DB_HOST pointing to decommissioned host.",
        logs=lines,
        summary="Config sync set payments DB_HOST to decommissioned legacy host; connections refused.",
        component="payments",
        category="config",
    )


@scenario("cache_ttl_zero", "config", (2, 3, 4))
def _ttl(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 10))
    lines = [
        audit(
            c.tick(),
            actor=f"user:{rng.choice(['dev-d', 'ops-e'])}",
            action="config.update",
            object="catalog-cache",
            key="ttl_seconds",
            old="300",
            new="0",
            env=imp.env,
        )
    ]
    lines += [
        jlog(
            c.tick(),
            "warn",
            "catalog-api",
            "cache miss",
            key=f"product:{rng.randint(1000, 99999)}",
            hit_ratio=round(rng.uniform(0, 0.05), 3),
        )
        for _ in range(rng.randint(2, 3))
    ]
    lines = pad(
        lines,
        lambda: jlog(c.tick(), "info", "catalog-api", "db query", ms=rng.randint(40, 900)),
        rng,
    )
    body = {
        2: f"Catalog p99 latency 4.8s (baseline 300ms); {imp.err_pct:g}% of product page loads time out.",
        3: "Cache hit ratio dropped to ~0; DB CPU at 85% but latency within SLO.",
        4: f"Environment: {imp.env}. Cache hit ratio dropped after config edit.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] catalog cache hit ratio collapsed",
        body=body,
        logs=lines,
        summary="catalog-cache TTL changed from 300s to 0; cache bypassed, DB load up.",
        component="cache",
        category="config",
    )


@scenario("dns_record_wrong", "config", (1, 2, 3))
def _dnsrec(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 30))
    wrong = ip(rng)
    name = {1: "api.example.com", 2: "eu.api.example.com", 3: "status-internal.example.com"}[
        imp.sev
    ]
    lines = [
        audit(
            c.tick(),
            actor="user:netops-f",
            action="dns.record.update",
            zone="example.com",
            name=name,
            type="A",
            old=ip(rng),
            new=wrong,
            ttl=300,
        )
    ]
    lines += [
        f"{iso(c.tick())} probe GET https://{name}/healthz -> dial tcp {wrong}:443: connect: no route to host"
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} probe GET https://www.example.com/ -> 200 in {rng.randint(40, 200)}ms"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] {name} resolving to unreachable address",
        body=error_line(imp, f"requests to {name}")
        + " Record edited manually 10 minutes before first failure.",
        logs=lines,
        summary=f"Manual DNS change pointed {name} A record to an unreachable IP.",
        component="dns",
        category="config",
    )


@scenario("rate_limit_too_strict", "config", (2, 3, 4))
def _ratelimit(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 1))
    lines = [
        audit(
            c.tick(),
            actor="pipeline:gateway-config",
            action="ratelimit.update",
            route="/v1/search",
            old="500r/s",
            new="5r/s",
            env=imp.env,
        )
    ]
    lines += [
        f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /v1/search?q={hexs(rng, 4)} HTTP/1.1" 429 0 "-" "app/5.{rng.randint(0, 9)}"'
        for _ in range(rng.randint(3, 5))
    ]
    lines = pad(
        lines,
        lambda: (
            f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /v1/catalog HTTP/1.1" 200 {rng.randint(500, 5000)} "-" "app/5.1"'
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Spike in HTTP 429 on /v1/search",
        body=(
            error_line(imp, "search requests")
            if imp.sev <= 3
            else f"Environment: {imp.env}. Load tests hitting 429."
        )
        + " Rate limit lowered from 500r/s to 5r/s in latest gateway config.",
        logs=lines,
        summary="Gateway rate limit for /v1/search misconfigured (5r/s instead of 500r/s); 429s.",
        component="api-gateway",
        category="config",
    )


@scenario("log_retention_misconfig", "config", (3, 4))
def _retention(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (60, 600))
    lines = [
        audit(
            c.tick(),
            actor="pipeline:infra-apply",
            action="bucket.lifecycle.update",
            bucket=f"audit-logs-{envtag(imp)}",
            rule="expire",
            old_days=400,
            new_days=4,
        )
    ]
    lines += [
        audit(
            c.tick(),
            actor="lifecycle",
            action="object.expire",
            bucket=f"audit-logs-{envtag(imp)}",
            count=rng.randint(1000, 90000),
        )
        for _ in range(rng.randint(1, 3))
    ]
    lines = pad(
        lines,
        lambda: audit(
            c.tick(),
            actor="svc:log-shipper",
            action="object.put",
            bucket=f"audit-logs-{envtag(imp)}",
            count=rng.randint(10, 400),
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Audit-log bucket lifecycle expiring objects after 4 days",
        body=(
            "Compliance audit logs older than 4 days being deleted in production; no customer impact."
            if imp.sev == 3
            else f"Environment: {imp.env}. Lifecycle rule expiry shortened by infra change."
        ),
        logs=lines,
        summary="Bucket lifecycle rule misconfigured (4d instead of 400d); audit logs expiring early.",
        component="storage",
        category="config",
    )


# --- deploy_regression ------------------------------------------------------


@scenario("payments_npe_release", "deploy_regression")
def _npe(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 5))
    v = f"v2.{rng.randint(10, 40)}.{rng.randint(0, 9)}"
    lines = [
        f"{iso(c.tick())} deploy payments-api {v} rollout complete (canary=0%, replicas={rng.randint(6, 12)})"
        if imp.sev <= 3
        else f"{iso(c.tick())} deploy payments-api {v} to {imp.env} complete",
        f"{iso(c.tick())} ERROR c.e.p.api.ErrorHandler - Unhandled exception on POST /v1/refunds",
        'java.lang.NullPointerException: Cannot invoke "com.example.payments.model.Currency.getCode()" because "refund.currency" is null',
        "\tat com.example.payments.refunds.RefundService.validate(RefundService.java:142)",
        "\tat com.example.payments.refunds.RefundService.create(RefundService.java:77)",
        "\tat com.example.payments.api.RefundController.post(RefundController.java:51)",
    ]
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} INFO  c.e.p.api.AccessLog - POST /v1/refunds 500 {rng.randint(5, 40)}ms"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] payments-api 5xx on /v1/refunds after {v} rollout",
        body=error_line(imp, "refund requests")
        + f" Errors began 2 minutes after {v} rollout; previous version had none.",
        logs=lines,
        summary=f"payments-api {v} NullPointerException in refund validation; refunds returning 500.",
        component="payments",
        category="deploy_regression",
    )


@scenario("gateway_route_404", "deploy_regression", (1, 2, 3))
def _route404(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 2))
    v = f"gw-{rng.randint(100, 400)}"
    path = {1: "/v1/", 2: "/v2/orders", 3: "/internal/reports"}[imp.sev]
    lines = [f"{iso(c.tick())} argo: rollout api-gateway image tag {v} Healthy"]
    lines += [
        f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET {path}{hexs(rng, 3)} HTTP/1.1" 404 548 "-" "app/5.{rng.randint(0, 9)}"'
        for _ in range(rng.randint(3, 5))
    ]
    lines = pad(
        lines,
        lambda: (
            f'{ip(rng)} - - [{nginx_acc_ts(c.tick())}] "GET /healthz HTTP/1.1" 200 2 "-" "kube-probe/1.31"'
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] 404s on {path}* after api-gateway {v}",
        body=error_line(imp, f"requests under {path}")
        + f" Route table in {v} drops the {path} prefix; rollback to previous tag restores.",
        logs=lines,
        summary=f"api-gateway release {v} dropped {path} route prefix; requests 404.",
        component="api-gateway",
        category="deploy_regression",
    )


@scenario("auth_release_token", "deploy_regression", (1, 2, 3, 4))
def _authrel(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 4))
    v = f"auth-service:{rng.randint(3, 6)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
    lines = [f"{iso(c.tick())} deploy {v} to {imp.env} finished"]
    lines += [
        jlog(
            c.tick(),
            "error",
            "auth-service",
            "token issue failed",
            error="KeyError: 'aud_claims'",
            path="/oauth/token",
            status=500,
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines.append("Traceback (most recent call last):")
    lines.append('  File "/app/auth/tokens.py", line 88, in issue')
    lines.append('    claims = build_claims(client, scopes)["aud_claims"]')
    lines.append("KeyError: 'aud_claims'")
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(), "info", "auth-service", "token issued", client=f"cli_{hexs(rng, 6)}"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] auth-service 500s on /oauth/token after {v}",
        body=error_line(imp, "token requests")
        + " Started immediately after deploy; only clients with custom audiences fail.",
        logs=lines,
        summary=f"auth-service release {v} KeyError building claims; /oauth/token returns 500.",
        component="auth-service",
        category="deploy_regression",
    )


@scenario("ingress_controller_upgrade", "deploy_regression", (1, 2, 3))
def _ingress(rng: Rng, imp: Impact) -> Draft:
    rows = [
        (
            "Normal",
            "ScalingReplicaSet",
            "deployment/ingress-nginx-controller",
            f"Scaled up replica set ingress-nginx-controller-{hexs(rng, 9)} to 3",
        ),
        (
            "Warning",
            "BackOff",
            f"pod/{pod(rng, 'ingress-nginx-controller')}",
            "Back-off restarting failed container controller",
        ),
        (
            "Warning",
            "Unhealthy",
            f"pod/{pod(rng, 'ingress-nginx-controller')}",
            "Readiness probe failed: HTTP probe failed with statuscode: 500",
        ),
        (
            "Warning",
            "RELOAD",
            f"pod/{pod(rng, 'ingress-nginx-controller')}",
            'Error reloading NGINX: unknown directive "more_set_headers"',
        ),
    ]
    lines = k8s_events(rows, rng)
    lines = pad(
        lines,
        lambda: k8s_events(
            [
                (
                    "Normal",
                    "Sync",
                    f"ingress/{rng.choice(['shop', 'api', 'admin'])}",
                    "Scheduled for sync",
                )
            ],
            rng,
        )[1],
        rng,
    )
    return Draft(
        title=f"[{imp.env}] ingress-nginx controller crashlooping after chart upgrade",
        body=error_line(imp, "HTTP traffic through the cluster ingress")
        + " Helm chart upgraded 4.x -> 5.0 an hour ago.",
        logs=lines,
        summary="ingress-nginx chart upgrade removed headers module; controller reload fails, pods crashloop.",
        component="kubernetes",
        category="deploy_regression",
    )


@scenario("cdn_worker_release", "deploy_regression", (1, 2, 3, 4))
def _edgeworker(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 3))
    v = f"edge-rules@{rng.randint(50, 300)}"
    lines = [
        f"{iso(c.tick())} edge deploy {v} published to {imp.env} ({rng.randint(30, 200)} PoPs)"
    ]
    lines += [
        json.dumps(
            {
                "ts": iso(c.tick()),
                "status": 200,
                "cache": "BYPASS",
                "path": f"/static/{hexs(rng, 6)}.js",
                "rule": "cache-static",
                "error": "TypeError: url.pathname is undefined",
            }
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines,
        lambda: json.dumps({"ts": iso(c.tick()), "status": 200, "cache": "BYPASS", "path": "/"}),
        rng,
    )
    body = {
        1: f"Cache hit ratio 0%; origin overloaded and returning 503 for {imp.err_pct:g}% of requests.",
        2: f"Cache hit ratio dropped 92% -> 15%; {imp.err_pct:g}% of page loads time out.",
        3: "Cache hit ratio dropped 92% -> 60%; origin absorbing load within SLO.",
        4: f"Environment: {imp.env}. Edge rules test suite shows cache bypass.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] CDN cache bypass after edge rules {v}",
        body=body,
        logs=lines,
        summary=f"Edge rules release {v} throws in cache rule; static assets bypass cache.",
        component="cdn",
        category="deploy_regression",
    )


@scenario("pipeline_template_break", "deploy_regression", (3, 4))
def _pipeline(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 5))
    v = f"ci-templates@v{rng.randint(4, 9)}.{rng.randint(0, 9)}"
    lines = [
        f"[{iso(c.tick())}] Including {v}/deploy.yml",
        f"[{iso(c.tick())}] $ helm upgrade --install app ./chart --set image.tag=$CI_COMMIT_SHA",
        f"[{iso(c.tick())}] Error: UPGRADE FAILED: template: chart/templates/deploy.yaml:12: unknown variable CI_COMMIT_SHA",
        f"[{iso(c.tick())}] ERROR: Job failed: exit code 1",
    ]
    lines = pad(
        lines, lambda: f"[{iso(c.tick())}] Fetching changes with git depth set to 20...", rng
    )
    return Draft(
        title=f"CI: deploy jobs failing after shared template {v}",
        body=(
            "All service deploy jobs failing since template release; no production impact."
            if imp.sev == 3
            else "Deploy jobs failing for one sandbox project after template bump."
        ),
        logs=lines,
        summary=f"Shared CI template {v} renamed a variable; deploy jobs fail.",
        component="ci-cd",
        category="deploy_regression",
    )


# --- auth -------------------------------------------------------------------


@scenario("db_password_rotation", "auth", (1, 2, 3, 4))
def _pwrot(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 5))
    pid = rng.randint(1000, 60000)
    lines = [
        audit(
            c.tick(),
            actor="svc:secret-rotator",
            action="secret.rotate",
            secret="payments-db-rw",
            status="SUCCESS",
        )
    ]
    lines += [
        f'{pg_ts(c.tick())} [{pid + i}] FATAL:  password authentication failed for user "payments_rw"'
        for i in range(rng.randint(2, 4))
    ]
    lines.append(
        f'{pg_ts(c.tick())} [{pid + 9}] DETAIL:  Connection matched pg_hba.conf line 98: "hostssl all all 198.51.100.0/24 scram-sha-256"'
    )
    lines = pad(
        lines,
        lambda: (
            f"{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  connection authorized: user=orders_ro database=orders"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] payments: database password authentication failures",
        body=error_line(imp, "payment requests")
        + " Secret rotated 5 minutes earlier; pods still use the old password.",
        logs=lines,
        summary="DB password rotated but payments pods still use old credential; auth failures.",
        component="payments",
        category="auth",
    )


@scenario("credential_stuffing", "auth", (1, 2, 3, 4))
def _stuffing(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.001, 0.2))
    rate = {
        1: rng.randint(3000, 9000),
        2: rng.randint(800, 3000),
        3: rng.randint(200, 800),
        4: rng.randint(20, 100),
    }[imp.sev]
    lines = [
        jlog(
            c.tick(),
            "warn",
            "auth-service",
            "login failed",
            reason="invalid_password",
            ip=ip(rng),
            user=f"u_{hexs(rng, 6)}",
        )
        for _ in range(rng.randint(3, 5))
    ]
    if imp.sev == 1:
        lines.append(
            jlog(
                c.tick(),
                "warn",
                "auth-service",
                "login succeeded from new device",
                ip=ip(rng),
                user=f"u_{hexs(rng, 6)}",
                risk="high",
                followed_by="payout_method_changed",
            )
        )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(), "warn", "auth-service", "login failed", reason="unknown_user", ip=ip(rng)
        ),
        rng,
    )
    body = {
        1: f"{rate} failed logins/min from rotating IPs; confirmed account takeovers with payout details changed.",
        2: f"{rate} failed logins/min from rotating IPs; WAF rules not matching; login latency degraded for real users.",
        3: f"{rate} failed logins/min, blocked by WAF bot rule; no successful suspicious logins.",
        4: f"{rate} failed logins/min against {imp.env if imp.env != 'production' else 'production'}; normal background noise.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Elevated failed logins (credential stuffing pattern)",
        body=body,
        logs=lines,
        summary=f"Credential-stuffing attack on login (~{rate}/min failures).",
        component="auth-service",
        category="auth",
    )


@scenario("iam_bucket_denied", "auth", (2, 3, 4))
def _iam(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.1, 10))
    lines = [
        audit(
            c.tick(),
            principal=f"svc-media-writer@{envtag(imp)}",
            action="PutObject",
            resource=f"bucket/media-{envtag(imp)}/uploads/{hexs(rng, 8)}",
            sourceIP=ip(rng),
            result="AccessDenied",
            reason="no identity-based policy allows the action",
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines = pad(
        lines,
        lambda: audit(
            c.tick(),
            principal=f"svc-media-reader@{envtag(imp)}",
            action="GetObject",
            result="Success",
        ),
        rng,
    )
    body = {
        2: f"{imp.err_pct:g}% of user uploads failing with AccessDenied in production.",
        3: "Thumbnail backfill job failing with AccessDenied; user uploads unaffected.",
        4: f"Environment: {imp.env}. Writer role lost PutObject permission.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] AccessDenied on media bucket writes",
        body=body + " Writer role policy was replaced by a narrower version this morning.",
        logs=lines,
        summary="Media writer service account lacks PutObject on bucket; writes AccessDenied.",
        component="storage",
        category="auth",
    )


@scenario("jwks_rotation", "auth", (1, 2, 3))
def _jwks(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 2))
    kid = hexs(rng, 8)
    lines = [
        jlog(
            c.tick(),
            "warn",
            "api-gateway",
            "jwt verify failed",
            error=f"no key found for kid {kid}",
            jwks_cache_age_s=rng.randint(3600, 86400),
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines.append(jlog(c.tick(), "info", "auth-service", "signing key rotated", new_kid=kid))
    lines = pad(
        lines,
        lambda: jlog(c.tick(), "info", "api-gateway", "request", status=401, path="/v1/account"),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Spike of 401s at api-gateway: unknown JWT kid",
        body=error_line(imp, "authenticated API calls")
        + " Gateway JWKS cache TTL 24h; new signing key not yet fetched.",
        logs=lines,
        summary=f"Signing key rotated; api-gateway JWKS cache stale, rejects tokens with kid {kid}.",
        component="api-gateway",
        category="auth",
    )


@scenario("ci_rbac_forbidden", "auth", (3, 4))
def _rbac(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 5))
    lines = [
        f"[{iso(c.tick())}] $ kubectl apply -f k8s/",
        f'[{iso(c.tick())}] Error from server (Forbidden): deployments.apps "checkout-web" is forbidden: User "system:serviceaccount:ci:deployer" cannot patch resource "deployments" in API group "apps" in the namespace "shop"',
        f"[{iso(c.tick())}] ERROR: Job failed: exit code 1",
    ]
    lines = pad(
        lines, lambda: f"[{iso(c.tick())}] Checking out {hexs(rng, 8)} as detached HEAD", rng
    )
    return Draft(
        title="CI deploys forbidden: ci:deployer cannot patch deployments",
        body=(
            "Deploys to all production namespaces blocked; running services unaffected."
            if imp.sev == 3
            else "Deploys to the sandbox namespace blocked."
        ),
        logs=lines,
        summary="CI deployer service account lost RBAC patch permission; deploys forbidden.",
        component="ci-cd",
        category="auth",
    )


@scenario("leaked_access_key", "auth", (1, 2))
def _leaked(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 20))
    src = ip(rng)
    lines = [
        audit(
            c.tick(),
            principal="svc-backup",
            accessKeyId="AKEXAMPLE" + hexs(rng, 8).upper(),
            action=a,
            sourceIP=src,
            userAgent="python-requests/2.32",
            geo="unusual",
            result="Success",
        )
        for a in (
            ["ListBuckets", "GetObject", "GetObject", "GetObject"]
            if imp.sev == 1
            else ["ListBuckets", "GetBucketPolicy"]
        )
    ]
    lines.append(
        audit(
            c.tick(),
            principal="secret-scanner",
            action="finding",
            detail="access key for svc-backup found in public paste",
        )
    )
    lines = pad(
        lines,
        lambda: audit(
            c.tick(),
            principal="svc-backup",
            action="PutObject",
            sourceIP=ip(rng),
            geo="expected",
            result="Success",
        ),
        rng,
    )
    body = (
        "Backup service key used from unknown network to download customer export objects (confirmed exfiltration in progress)."
        if imp.sev == 1
        else "Backup service key used from unknown network for listing only; no object reads yet."
    )
    return Draft(
        title=f"[{imp.env}] Backup access key used from unrecognised source {src}",
        body=body,
        logs=lines,
        summary="Leaked backup access key used from unknown IP; key must be revoked.",
        component="storage",
        category="auth",
    )


# --- data -------------------------------------------------------------------


@scenario("pg_replication_lag", "data", (2, 3, 4))
def _replag(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (5, 60))
    lag = {2: rng.randint(600, 3600), 3: rng.randint(60, 300), 4: rng.randint(20, 60)}[imp.sev]
    pid = rng.randint(1000, 60000)
    lines = [
        f"{pg_ts(c.tick())} [{pid}] LOG:  started streaming WAL from primary at {rng.randint(1, 9)}/{hexs(rng, 8).upper()} on timeline 3",
        f"{pg_ts(c.tick())} [{pid}] ERROR:  requested WAL segment 0000000300000{rng.randint(10, 99)}00000{rng.randint(10, 99)} has already been removed",
        f"{pg_ts(c.tick())} [{pid}] FATAL:  could not receive data from WAL stream",
        f"{pg_ts(c.tick())} [{pid + 1}] LOG:  restored log file from archive; replay lag {lag}s",
    ]
    lines = pad(
        lines,
        lambda: f"{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  restartpoint complete",
        rng,
    )
    body = {
        2: f"Read replicas {lag // 60} min behind; customers see stale order status in account pages.",
        3: f"Reporting replica {lag}s behind; dashboards slightly stale. No customer-facing reads.",
        4: f"Environment: {imp.env}. Replica lag {lag}s, catching up.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Postgres replica lag {lag}s, WAL stream broken",
        body=body,
        logs=lines,
        summary=f"Replica lost WAL stream (segment removed), replaying from archive; lag {lag}s.",
        component="database",
        category="data",
    )


@scenario("duplicate_charges", "data", (1, 2))
def _dupes(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 3))
    lines = [
        jlog(
            c.tick(),
            "error",
            "payments-ledger",
            "duplicate capture detected",
            order_id=f"ord_{hexs(rng, 8)}",
            idempotency_key=None,
            captures=2,
            amount_minor=rng.randint(500, 90000),
        )
        for _ in range(rng.randint(3, 5))
    ]
    lines.append(
        jlog(
            c.tick(),
            "warn",
            "payments-ledger",
            "reconciliation mismatch",
            ledger_vs_provider_delta=rng.randint(10, 400),
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(), "info", "payments-ledger", "capture ok", order_id=f"ord_{hexs(rng, 8)}"
        ),
        rng,
    )
    body = (
        "Customers being double-charged right now; duplicate captures continuing at ~"
        f"{rng.randint(20, 90)}/min."
        if imp.sev == 1
        else f"{rng.randint(10, 60)} duplicate captures found in the last hour; capture retries were disabled 20 minutes ago and no new duplicates since. Affected customers see double charges pending refund."
    )
    return Draft(
        title=f"[{imp.env}] Duplicate payment captures detected",
        body=body + " Idempotency keys missing on retried capture calls.",
        logs=lines,
        summary="Payment captures duplicated (missing idempotency keys); customers double-charged.",
        component="payments",
        category="data",
    )


@scenario("backup_failed", "data", (3, 4))
def _backup(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 60))
    last = {3: rng.randint(26, 70), 4: rng.randint(4, 20)}[imp.sev]
    lines = [
        f"{syslog_ts(c.tick())} backup-01 systemd[1]: Starting Nightly database backup...",
        f"{syslog_ts(c.tick())} backup-01 pgbackrest[{rng.randint(1000, 9999)}]: ERROR: [082]: WAL segment was not archived before the 60000ms timeout",
        f"{syslog_ts(c.tick())} backup-01 systemd[1]: pg-backup.service: Main process exited, code=exited, status=82/n/a",
        f"{syslog_ts(c.tick())} backup-01 systemd[1]: pg-backup.service: Failed with result 'exit-code'.",
    ]
    lines = pad(
        lines,
        lambda: (
            f"{syslog_ts(c.tick())} backup-01 CRON[{rng.randint(1000, 9999)}]: (root) CMD (/usr/local/bin/rotate-logs)"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Nightly database backup failed",
        body=(
            f"Last successful backup {last}h ago (RPO target 24h). Production database healthy."
            if imp.sev == 3
            else f"Last successful backup {last}h ago (RPO target 24h). Single failure."
        ),
        logs=lines,
        summary=f"Nightly pgBackRest backup failing (WAL archive timeout); last good backup {last}h ago.",
        component="storage",
        category="data",
    )


@scenario("pg_checksum_corruption", "data", (1, 2, 3))
def _corrupt(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.5, 30))
    pid = rng.randint(1000, 60000)
    rel = rng.choice(["orders", "order_items", "invoices"])
    lines = [
        f"{pg_ts(c.tick())} [{pid}] WARNING:  page verification failed, calculated checksum {rng.randint(1000, 65000)} but expected {rng.randint(1000, 65000)}",
        f"{pg_ts(c.tick())} [{pid}] ERROR:  invalid page in block {rng.randint(1000, 900000)} of relation base/16384/{rng.randint(20000, 90000)}",
        f'{pg_ts(c.tick())} [{pid}] CONTEXT:  SQL statement "SELECT * FROM {rel} WHERE id = $1"',
    ]
    lines = pad(
        lines,
        lambda: (
            f'{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  automatic vacuum of table "orders.public.{rel}"'
        ),
        rng,
    )
    body = {
        1: f"Corrupted pages in {rel} on the primary; writes to affected rows failing and corruption spreading to new blocks.",
        2: f"{rng.randint(5, 40)} corrupted pages in {rel}; affected customers cannot load order history.",
        3: f"1 corrupted page in {rel} on reporting replica only; primary verified clean.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Postgres checksum failures in {rel}",
        body=body,
        logs=lines,
        summary=f"Postgres data page checksum failures (corruption) in {rel}.",
        component="database",
        category="data",
    )


@scenario("kafka_offset_reset", "data", (2, 3, 4))
def _offsets(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.1, 5))
    grp = rng.choice(["email-sender", "invoice-writer", "search-indexer"])
    lines = [
        f"[{kafka_ts(c.tick())}] INFO [GroupCoordinator {rng.randint(1, 6)}]: Member consumer-{grp}-{rng.randint(1, 9)} in group {grp} has failed, removing it (kafka.coordinator.group.GroupCoordinator)",
        f"[{kafka_ts(c.tick())}] WARN [Consumer clientId={grp}-1, groupId={grp}] Committed offset out of range for partition orders-{rng.randint(0, 23)}; resetting offset to EARLIEST (org.apache.kafka.clients.consumer.internals.Fetcher)",
        f"[{kafka_ts(c.tick())}] INFO [Consumer clientId={grp}-1, groupId={grp}] Seeking to EARLIEST offset of partition orders-{rng.randint(0, 23)}",
    ]
    lines = pad(
        lines,
        lambda: (
            f"[{kafka_ts(c.tick())}] INFO [Consumer clientId={grp}-2, groupId={grp}] Fetch position updated"
        ),
        rng,
    )
    body = {
        2: f"{grp} reprocessing 9 days of order events; customers receiving duplicate notification emails.",
        3: f"{grp} reprocessing old events; duplicates suppressed downstream, backlog growing.",
        4: f"Environment: {imp.env}. Consumer group offsets reset to earliest.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Consumer group {grp} offsets reset to EARLIEST",
        body=body,
        logs=lines,
        summary=f"Kafka consumer group {grp} offsets reset to earliest; old events reprocessed.",
        component="message-queue",
        category="data",
    )


@scenario("stale_price_cache", "data", (2, 3))
def _stale(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 30))
    lines = [
        jlog(
            c.tick(),
            "warn",
            "pricing",
            "cache/db mismatch",
            sku=f"sku_{rng.randint(1000, 99999)}",
            cached=round(rng.uniform(5, 90), 2),
            db=round(rng.uniform(5, 90), 2),
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines.append(
        jlog(
            c.tick(),
            "error",
            "pricing",
            "invalidation publish dropped",
            channel="price-updates",
            subscribers=0,
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "pricing",
            "price update committed",
            sku=f"sku_{rng.randint(1000, 99999)}",
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Price cache inconsistent with database",
        body=(
            "Storefront displaying outdated prices for thousands of SKUs; checkout re-prices from the database, so charges are correct."
            if imp.sev == 2
            else "A few dozen SKUs showing outdated prices on internal admin tool only."
        ),
        logs=lines,
        summary="Price cache invalidations not delivered; cached prices inconsistent with DB.",
        component="cache",
        category="data",
    )


# --- unknown ----------------------------------------------------------------


@scenario("synthetic_generic_fail", "unknown", (1, 2, 3))
def _synth(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (10, 60))
    regions = {1: "all", 2: "2 of 5", 3: "1 of 5"}[imp.sev]
    lines = [
        f"{iso(c.tick())} synthetic check 'checkout-journey' FAILED from {rng.choice(REGIONS)}: step 3 'submit order' -> HTTP 502 (no body)"
        for _ in range(rng.randint(3, 5))
    ]
    lines = pad(
        lines,
        lambda: (
            f"{iso(c.tick())} synthetic check 'homepage' PASSED from {rng.choice(REGIONS)} in {rng.randint(200, 900)}ms"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Synthetic checkout journey failing ({regions} regions)",
        body=f"Synthetic monitor failing from {regions} probe regions. Real-user error rate {imp.err_pct:g}%. No correlated alerts yet.",
        logs=lines,
        summary=f"Synthetic checkout failing with bare 502 from {regions} regions; cause not identified.",
        component="unknown",
        category="unknown",
    )


@scenario("traffic_drop_anomaly", "unknown", (2, 3))
def _drop(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (30, 120))
    d = {2: rng.randint(30, 60), 3: rng.randint(10, 20)}[imp.sev]
    lines = [
        f"{iso(c.tick())} anomaly-detector: requests_per_second below forecast by {d + rng.randint(-3, 3)}% (z=-{rng.uniform(3, 7):.1f})"
        for _ in range(rng.randint(3, 5))
    ]
    lines.append(
        f"{iso(c.tick())} anomaly-detector: error_rate within normal band (0.{rng.randint(1, 4)}%)"
    )
    lines = pad(
        lines, lambda: f"{iso(c.tick())} anomaly-detector: latency_p95 within normal band", rng
    )
    return Draft(
        title=f"[{imp.env}] Traffic {d}% below forecast",
        body=f"Request volume {d}% below forecast with no rise in errors or latency. Orders per minute down {d}%. No deploys or config changes recorded.",
        logs=lines,
        summary=f"Traffic and orders {d}% below forecast with no errors; cause unknown.",
        component="unknown",
        category="unknown",
    )


@scenario("silent_crash_worker", "unknown", (2, 3, 4))
def _silent(rng: Rng, imp: Impact) -> Draft:
    p = pod(rng, "payments-worker")
    rows = [
        ("Warning", "BackOff", f"pod/{p}", "Back-off restarting failed container worker")
    ] * rng.randint(1, 3)
    rows += [
        ("Normal", "Pulled", f"pod/{p}", "Container image already present on machine"),
        ("Normal", "Started", f"pod/{p}", "Started container worker"),
    ]
    lines = k8s_events(rows, rng)
    lines.append(f"$ kubectl logs {p} --previous")
    lines.append("(no output)")
    lines.append(
        f"Last State: Terminated  Reason: Error  Exit Code: 1  Restart Count: {rng.randint(5, 60)}"
    )
    body = {
        2: f"Payout batch not processing; {imp.err_pct:g}% of merchant payouts delayed beyond SLA.",
        3: "Payout batch delayed; within SLA so far.",
        4: f"Environment: {imp.env}. Worker crashlooping.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] payments-worker CrashLoopBackOff (exit code 1, no logs)",
        body=body + " Memory and CPU normal; no recent deploy.",
        logs=lines,
        summary="payments-worker crashlooping with exit 1 and no logs; cause unknown.",
        component="payments",
        category="unknown",
    )


@scenario("error_budget_burn", "unknown", (2, 3))
def _burn(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (60, 300))
    lines = [
        f"{iso(c.tick())} slo-alert auth-service availability burn_rate_1h={rng.uniform(6, 20):.1f} burn_rate_6h={rng.uniform(2, 8):.1f}"
    ]
    lines += [
        jlog(
            c.tick(), "error", "auth-service", "request failed", status=500, error="internal error"
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(lines, lambda: jlog(c.tick(), "info", "auth-service", "request", status=200), rng)
    return Draft(
        title=f"[{imp.env}] auth-service error budget burning fast",
        body=error_line(imp, "auth requests")
        + " Errors are generic 500s with no stack traces; dependencies report healthy.",
        logs=lines,
        summary="auth-service error budget burn from generic 500s; root cause not evident.",
        component="auth-service",
        category="unknown",
    )


@scenario("agent_heartbeat_missing", "unknown", (3, 4))
def _heartbeat(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (30, 120))
    h = f"host-{hexs(rng, 6)}"
    lines = [
        f"{iso(c.tick())} monitoring: no heartbeat from agent on {h} for {m}m" for m in (5, 10, 15)
    ]
    lines += [
        f"{iso(c.tick())} monitoring: host {h} last metrics: cpu={rng.randint(5, 40)}% mem={rng.randint(20, 60)}% disk={rng.randint(20, 70)}%",
        f"{iso(c.tick())} inventory: {h} role=unknown owner=unassigned",
    ]
    return Draft(
        title=f"[{imp.env}] Monitoring agent heartbeat missing on {h}",
        body=(
            "Host role unknown; no service alerts correlate."
            if imp.sev == 3
            else f"Environment: {imp.env}. Host role unknown."
        ),
        logs=lines,
        summary=f"Monitoring heartbeat lost from unowned host {h}; impact and cause unknown.",
        component="unknown",
        category="unknown",
    )


# --- ambiguous scenarios (labelled by LABELING.md tie-break rules) ---------


@scenario(
    "amb_oom_after_limit_change",
    "config",
    (1, 2, 3),
    ambiguous=True,
    note="OOM symptom, but an earlier config change lowered the memory limit: earliest cause wins -> config.",
)
def _amb_oom(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 30))
    p = pod(rng, "payments-api")
    lines = [
        audit(
            c.tick(),
            actor="pipeline:helm-values",
            action="values.update",
            release="payments-api",
            key="resources.limits.memory",
            old="4Gi",
            new="1Gi",
        )
    ]
    lines += [
        f"{iso(c.tick())} kubelet: container payments-api in pod {p} OOMKilled (exit code 137)"
        for _ in range(rng.randint(2, 3))
    ]
    lines.append(
        f'{iso(c.tick())} Exception in thread "main" java.lang.OutOfMemoryError: Java heap space'
    )
    lines = pad(lines, lambda: f"{iso(c.tick())} kubelet: Started container payments-api", rng)
    return Draft(
        title=f"[{imp.env}] payments-api OOMKilled repeatedly",
        body=error_line(imp, "payment requests") + " Heap settings unchanged in code.",
        logs=lines,
        summary="Helm values change cut payments-api memory limit 4Gi->1Gi; pods OOMKilled.",
        component="payments",
        category="config",
    )


@scenario(
    "amb_mtls_cert_handshake",
    "config",
    (1, 2, 3),
    ambiguous=True,
    note="Looks like auth (handshake/permission errors) but cause is an expired internal mTLS certificate -> config.",
)
def _amb_mtls(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 2))
    lines = [
        jlog(
            c.tick(),
            "error",
            "payments",
            "call auth-service failed",
            error="remote error: tls: bad certificate",
            peer="auth-service:8443",
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines.append(
        jlog(
            c.tick(),
            "warn",
            "auth-service",
            "tls handshake rejected",
            error="x509: certificate has expired",
            client_cn="payments.internal.example",
        )
    )
    lines = pad(
        lines,
        lambda: jlog(
            c.tick(),
            "info",
            "auth-service",
            "request",
            status=200,
            client_cn="web.internal.example",
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] payments -> auth-service calls failing: tls bad certificate",
        body=error_line(imp, "payment requests needing token introspection")
        + " Looks like a permission problem to the on-call.",
        logs=lines,
        summary="payments mTLS client certificate expired; auth-service rejects handshakes.",
        component="payments",
        category="config",
    )


@scenario(
    "amb_deploy_and_slow_queries",
    "deploy_regression",
    (1, 2, 3),
    ambiguous=True,
    note="Slow DB queries and a release both present; release precedes symptoms -> deploy_regression on the deployed service.",
)
def _amb_deploy_db(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (5, 60))
    v = f"v{rng.randint(1, 9)}.{rng.randint(0, 30)}.0"
    pid = rng.randint(1000, 60000)
    lines = [f"{iso(c.tick())} deploy catalog-api {v} complete"]
    lines += [
        f"{pg_ts(c.tick())} [{pid + i}] LOG:  duration: {rng.randint(4000, 30000)}.{rng.randint(100, 999)} ms  statement: SELECT * FROM products WHERE lower(title) LIKE $1"
        for i in range(rng.randint(2, 4))
    ]
    lines.append(f"{pg_ts(c.tick())} [{pid}] LOG:  checkpoint complete; cpu 96%")
    lines = pad(
        lines,
        lambda: (
            f"{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  connection authorized: user=catalog"
        ),
        rng,
    )
    return Draft(
        title=f"[{imp.env}] Database CPU 96%, catalog latency high",
        body=error_line(imp, "product searches")
        + f" catalog-api {v} shipped 20 minutes before the DB alert.",
        logs=lines,
        summary=f"catalog-api {v} introduced unindexed LIKE query; DB CPU saturated.",
        component="database",
        category="deploy_regression",
    )


@scenario(
    "amb_idp_down_401",
    "dependency",
    (1, 2, 3),
    ambiguous=True,
    note="401s look like auth, but the earliest event is the external IdP failing -> dependency.",
)
def _amb_idp(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.05, 3))
    lines = [
        jlog(
            c.tick(),
            "error",
            "auth-service",
            "SSO token exchange failed",
            idp="sso-vendor",
            http_status=503,
        )
    ]
    lines += [
        jlog(
            c.tick(), "warn", "auth-service", "login rejected", status=401, reason="sso_unavailable"
        )
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(lines, lambda: jlog(c.tick(), "info", "auth-service", "password login ok"), rng)
    return Draft(
        title=f"[{imp.env}] Surge in 401 responses on SSO login",
        body=error_line(imp, "SSO logins") + " Password logins work.",
        logs=lines,
        summary="External SSO identity provider returning 503; SSO logins rejected with 401.",
        component="auth-service",
        category="dependency",
    )


@scenario(
    "amb_pool_and_packet_loss",
    "resource_exhaustion",
    (1, 2, 3),
    ambiguous=True,
    note="Pool exhaustion and packet loss start in the same minute; tie-break order puts resource_exhaustion first.",
)
def _amb_pool(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (0.01, 0.5))
    lines = [
        jlog(c.tick(), "error", "orders-api", "pool exhausted", pool="pgx", in_use=100, max=100)
    ]
    lines.append(
        jlog(
            c.tick(),
            "warn",
            "netmon",
            "packet loss",
            path="app->db",
            loss=round(rng.uniform(0.02, 0.08), 3),
        )
    )
    lines += [
        jlog(c.tick(), "error", "orders-api", "acquire conn timeout", waited_ms=5000)
        for _ in range(rng.randint(2, 3))
    ]
    lines = pad(
        lines, lambda: jlog(c.tick(), "info", "orders-api", "query ok", ms=rng.randint(5, 50)), rng
    )
    return Draft(
        title=f"[{imp.env}] orders DB connection pool exhausted",
        body=error_line(imp, "order requests")
        + " Mild packet loss on app->db path at the same time.",
        logs=lines,
        summary="orders DB connection pool exhausted (100/100) with concurrent mild packet loss.",
        component="database",
        category="resource_exhaustion",
    )


@scenario(
    "amb_replica_lag_io",
    "resource_exhaustion",
    (2, 3, 4),
    ambiguous=True,
    note="Replication lag (data) caused by earlier disk IO saturation on the replica -> resource_exhaustion.",
)
def _amb_lag(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (10, 90))
    lines = [
        f"{syslog_ts(c.tick())} pg-replica-2 node_exporter: disk nvme1n1 io_util=100% await={rng.randint(80, 400)}ms"
    ]
    pid = rng.randint(1000, 60000)
    lines += [
        f"{pg_ts(c.tick())} [{pid}] LOG:  replication lag {rng.randint(120, 2400)}s, replay rate {rng.randint(1, 9)} MB/s"
        for _ in range(rng.randint(2, 4))
    ]
    lines = pad(
        lines,
        lambda: f"{syslog_ts(c.tick())} pg-replica-2 node_exporter: disk nvme1n1 io_util=99%",
        rng,
    )
    body = {
        2: "Customer account pages reading from replica show stale data for ~20 min.",
        3: "Reporting replica behind; no customer reads.",
        4: f"Environment: {imp.env}. Replica behind.",
    }[imp.sev]
    return Draft(
        title=f"[{imp.env}] Replica lag growing on pg-replica-2",
        body=body + " Disk on replica saturated since before lag began.",
        logs=lines,
        summary="Replica disk IO saturated (100% util) causing growing replication lag.",
        component="database",
        category="resource_exhaustion",
    )


@scenario(
    "amb_borderline_err_network",
    "network",
    (1, 2),
    ambiguous=True,
    note="Error rate exactly at a severity threshold (25% -> SEV1, 2% -> SEV2; thresholds inclusive).",
)
def _amb_border_net(rng: Rng, imp: Impact) -> Draft:
    imp = impact(rng, imp.sev, borderline=True)
    d = _packet_loss(rng, imp)
    return d


@scenario(
    "amb_borderline_eta_disk",
    "resource_exhaustion",
    (2, 3),
    ambiguous=True,
    note="Projected exhaustion exactly at a threshold (60 min -> SEV2, 24 h -> SEV3; inclusive).",
)
def _amb_border_disk(rng: Rng, imp: Impact) -> Draft:
    return _kafka_disk(rng, impact(rng, imp.sev, borderline=True))


@scenario(
    "amb_staging_demo",
    "auth",
    (4,),
    ambiguous=True,
    note="Customer-visible demo on staging still counts as non-production -> SEV4.",
)
def _amb_staging(rng: Rng, imp: Impact) -> Draft:
    d = _jwks(rng, Impact(3, "staging", 0.5, 0, "no customer-facing impact"))
    d.title = "[staging] 401s at api-gateway: unknown JWT kid (sales demo environment)"
    d.body = "Environment: staging, currently used for a prospect demo. 100% of authenticated calls failing in staging."
    return d


@scenario(
    "amb_borderline_err_auth",
    "auth",
    (1, 2),
    ambiguous=True,
    note="Error rate exactly at a severity threshold (inclusive).",
)
def _amb_border_auth(rng: Rng, imp: Impact) -> Draft:
    return _jwks(rng, impact(rng, imp.sev, borderline=True))


@scenario(
    "amb_deploy_plus_config",
    "deploy_regression",
    (2, 3),
    ambiguous=True,
    note="A release and a config edit landed in the same minute; tie-break order puts deploy_regression first.",
)
def _amb_deploy_cfg(rng: Rng, imp: Impact) -> Draft:
    d = _route404(rng, imp)
    t = before_first(d.logs, rng, (0.05, 0.5))
    d.logs.insert(
        1,
        audit(
            t,
            actor="user:ops-a",
            action="gateway.config.update",
            key="upstream_keepalive",
            old="32",
            new="64",
        ),
    )
    d.body += " A gateway config edit was applied in the same minute as the rollout."
    return d


@scenario(
    "amb_dup_after_release",
    "deploy_regression",
    (1, 2),
    ambiguous=True,
    note="Data symptom (duplicates) introduced by a release -> deploy_regression (deploy beats data).",
)
def _amb_dup(rng: Rng, imp: Impact) -> Draft:
    d = _dupes(rng, imp)
    v = f"v{rng.randint(3, 9)}.{rng.randint(0, 20)}.0"
    d.logs.insert(
        0,
        f"{iso(before_first(d.logs, rng))} deploy payments-ledger {v} complete (removed legacy idempotency middleware)",
    )
    d.body += f" Duplicates started right after {v}."
    d.summary = f"payments-ledger {v} removed idempotency handling; duplicate captures."
    d.category = "deploy_regression"
    return d


@scenario(
    "amb_data_backup_vs_storage",
    "data",
    (3,),
    ambiguous=True,
    note="Backup failure caused by WAL archive timeout; no resource evidence -> data (backups).",
)
def _amb_backup(rng: Rng, imp: Impact) -> Draft:
    return _backup(rng, imp)


@scenario(
    "amb_unknown_partial_evidence",
    "unknown",
    (2, 3),
    ambiguous=True,
    note="Weak hints in several directions, none conclusive -> unknown.",
)
def _amb_unknown(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (5, 60))
    lines = [
        jlog(c.tick(), "warn", "checkout-web", "slow response", ms=rng.randint(2000, 6000)),
        f"{iso(c.tick())} kubelet: Readiness probe failed for pod {pod(rng, 'checkout-web')}",
        jlog(c.tick(), "info", "netmon", "packet loss", loss=0.004),
        f"{pg_ts(c.tick())} [{rng.randint(1000, 60000)}] LOG:  duration: 1200.4 ms  statement: SELECT ...",
        jlog(c.tick(), "warn", "checkout-web", "upstream 502", upstream="unknown"),
    ]
    return Draft(
        title=f"[{imp.env}] checkout-web intermittent 502s",
        body=error_line(imp, "checkout page loads")
        + " No deploys; minor packet loss and a slow query logged around the same time.",
        logs=lines,
        summary="Intermittent checkout 502s with inconclusive signals (slow query, minor loss).",
        component="unknown",
        category="unknown",
    )


@scenario(
    "amb_dependency_queue",
    "dependency",
    (2, 3),
    ambiguous=True,
    note="Queue backlog symptom, but the earliest event is the external email provider rejecting sends -> dependency.",
)
def _amb_queue(rng: Rng, imp: Impact) -> Draft:
    c = Clock(rng, (1, 20))
    lines = [
        jlog(
            c.tick(),
            "error",
            "notifier",
            "email provider rejected batch",
            provider="mail-vendor",
            http_status=429,
        )
    ]
    lines += [
        f"[{kafka_ts(c.tick())}] WARN consumer lag for group notifier on topic notifications = {rng.randint(10**4, 10**6)}"
        for _ in range(rng.randint(2, 3))
    ]
    lines = pad(lines, lambda: jlog(c.tick(), "info", "notifier", "retry scheduled"), rng)
    return Draft(
        title=f"[{imp.env}] notifications topic consumer lag growing",
        body=(
            "Password-reset emails delayed >30 min for many users."
            if imp.sev == 2
            else "Marketing emails delayed; transactional emails unaffected."
        ),
        logs=lines,
        summary="External email provider throttling (429); notifier backlog growing on queue.",
        component="message-queue",
        category="dependency",
    )


@scenario(
    "amb_network_vs_dns",
    "network",
    (2, 3),
    ambiguous=True,
    note="DNS errors caused by packet loss to the resolver; no config change -> network, component dns.",
)
def _amb_netdns(rng: Rng, imp: Impact) -> Draft:
    return _coredns(rng, imp)


@scenario(
    "amb_data_cache",
    "data",
    (2, 3),
    ambiguous=True,
    note="Stale cache (data) while invalidation subscriber count is 0 without any config change -> data.",
)
def _amb_cache(rng: Rng, imp: Impact) -> Draft:
    return _stale(rng, imp)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def sev_quota(n: int) -> list[int]:
    raw = {s: n * p for s, p in SEV_SHARE.items()}
    out = {s: int(v) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - out[s], reverse=True)[: n - sum(out.values())]:
        out[s] += 1
    return [s for s in (1, 2, 3, 4) for _ in range(out[s])]


def build_items(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    seq = 0
    for cat, n in CATEGORY_TARGETS.items():
        sevs = sev_quota(n)
        rng.shuffle(sevs)
        regular = [s for s in SCENARIOS if s.category == cat and not s.ambiguous]
        ambiguous = [s for s in SCENARIOS if s.category == cat and s.ambiguous]
        n_amb = AMBIGUOUS_TARGETS.get(cat, 0)
        rr: Counter[str] = Counter()
        amb_slots: set[int] = set()
        for i, sev in enumerate(sevs):
            if len(amb_slots) < n_amb and any(sev in s.sevs for s in ambiguous):
                amb_slots.add(i)
        for i, sev in enumerate(sevs):
            pool = ambiguous if i in amb_slots else regular
            fits = [s for s in pool if sev in s.sevs] or [s for s in regular if sev in s.sevs]
            if not fits:  # no scenario at this severity: take the nearest supported one
                fits = regular
                sev = min(fits[0].sevs, key=lambda x: abs(x - sev))
                fits = [s for s in regular if sev in s.sevs]
            sc = min(fits, key=lambda s: (rr[s.key], s.key))
            rr[sc.key] += 1
            item_rng = random.Random(f"{seed}:{sc.key}:{rr[sc.key]}")
            if sev not in sc.sevs:
                sev = min(sc.sevs, key=lambda x: abs(x - sev))
            d = sc.build(item_rng, impact(item_rng, sev, prod_sev4=sc.key in PROD_SEV4_OK))
            d.logs = order_logs(d.logs)
            assert d.category == cat == sc.category, (sc.key, d.category, cat)
            if sc.key == "amb_staging_demo":
                sev = 4
            seq += 1
            items.append(
                {
                    "id": f"tri-{seq:04d}",
                    "input": {
                        "alert_title": d.title,
                        "alert_body": d.body,
                        "log_excerpt": "\n".join(d.logs),
                        **({"metrics_snippet": d.metrics} if d.metrics else {}),
                    },
                    "label": {
                        "severity": f"SEV{sev}",
                        "component": d.component,
                        "category": d.category,
                        "needs_page": sev <= 2,
                        "summary": d.summary[:200],
                    },
                    "review_status": "draft",
                    "meta": {
                        "scenario": sc.key,
                        "ambiguous": sc.ambiguous,
                        "ambiguity_note": sc.note or None,
                        "generator": GENERATOR_VERSION,
                    },
                }
            )
    # Shuffle so ids don't reveal category, then renumber.
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["id"] = f"tri-{i:04d}"
    return items


def split(
    items: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stratified by category: dev gets N_DEV items, allocated by largest remainder."""
    rng = random.Random(seed + 1)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by_cat.setdefault(it["label"]["category"], []).append(it)
    share = {c: len(v) * N_DEV / len(items) for c, v in by_cat.items()}
    alloc = {c: int(s) for c, s in share.items()}
    for c in sorted(share, key=lambda c: (share[c] - alloc[c], c), reverse=True)[
        : N_DEV - sum(alloc.values())
    ]:
        alloc[c] += 1
    dev, test = [], []
    for c in sorted(by_cat):
        group = sorted(by_cat[c], key=lambda it: it["id"])
        rng.shuffle(group)
        dev += group[: alloc[c]]
        test += group[alloc[c] :]
    return sorted(dev, key=lambda it: it["id"]), sorted(test, key=lambda it: it["id"])


REVIEW_COLUMNS = [
    "id",
    "split",
    "alert_title",
    "alert_body",
    "log_excerpt",
    "severity",
    "component",
    "category",
    "needs_page",
    "summary",
    "reviewer_ok",
    "corrected_severity",
    "corrected_component",
    "corrected_category",
    "corrected_needs_page",
    "notes",
]


def write_review_csv(path: Path, splits: dict[str, list[dict[str, Any]]]) -> None:
    rows = []
    for name, items in splits.items():
        for it in items:
            lab, inp = it["label"], it["input"]
            rows.append(
                {
                    "id": it["id"],
                    "split": name,
                    "alert_title": inp["alert_title"],
                    "alert_body": inp["alert_body"],
                    "log_excerpt": inp["log_excerpt"],
                    "severity": lab["severity"],
                    "component": lab["component"],
                    "category": lab["category"],
                    "needs_page": str(lab["needs_page"]).lower(),
                    "summary": lab["summary"],
                    "reviewer_ok": "",
                    "corrected_severity": "",
                    "corrected_component": "",
                    "corrected_category": "",
                    "corrected_needs_page": "",
                    "notes": "",
                }
            )
    rows.sort(key=lambda r: r["id"])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite even if items were reviewed")
    args = ap.parse_args()
    for name in ("dev.jsonl", "test.jsonl"):
        p = OUT_DIR / name
        if p.exists() and not args.force and '"review_status": "reviewed"' in p.read_text():
            print(f"refusing to overwrite reviewed dataset {p} (use --force)", file=sys.stderr)
            return 1
    items = build_items(args.seed)
    assert len(items) == N_ITEMS, len(items)
    dev, test = split(items, args.seed)
    write_jsonl(OUT_DIR / "dev.jsonl", dev)
    write_jsonl(OUT_DIR / "test.jsonl", test)
    write_review_csv(OUT_DIR / "review.csv", {"dev": dev, "test": test})
    m = write_manifest(OUT_DIR, {"generator": GENERATOR_VERSION, "split_seed": args.seed})
    print(json.dumps(m["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
