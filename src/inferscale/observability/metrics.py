"""Prometheus metrics. Each Gateway owns its own registry so tests stay isolated."""

from __future__ import annotations

from collections.abc import Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    30.0,
    45.0,
    60.0,
)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "inferscale_requests", "Chat completion requests by outcome", ["outcome"], registry=r
        )
        self.request_duration = Histogram(
            "inferscale_request_duration_seconds",
            "End-to-end request duration",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "inferscale_ttft_seconds",
            "Time to first streamed token",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "inferscale_prompt_tokens", "Prompt tokens reported by the engine", registry=r
        )
        self.completion_tokens = Counter(
            "inferscale_completion_tokens", "Completion tokens reported by the engine", registry=r
        )
        self.inflight = Gauge("inferscale_inflight", "Requests holding an engine slot", registry=r)
        self.queue_depth = Gauge(
            "inferscale_queue_depth", "Requests waiting for an engine slot", registry=r
        )
        self.rejected = Counter(
            "inferscale_rejected", "Requests rejected by admission", ["reason"], registry=r
        )
        self.engine_up = Gauge(
            "inferscale_engine_up", "1 if the engine health check passes", registry=r
        )
        self.model_info = Gauge(
            "inferscale_model_info",
            "Active model (value is always 1)",
            ["model_version", "engine", "quant"],
            registry=r,
        )

    def bind_admission(
        self, inflight: Callable[[], float], queue_depth: Callable[[], float]
    ) -> None:
        self.inflight.set_function(inflight)
        self.queue_depth.set_function(queue_depth)

    def render(self) -> bytes:
        return generate_latest(self.registry)
