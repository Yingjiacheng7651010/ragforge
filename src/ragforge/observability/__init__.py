"""Observability suite: OpenTelemetry tracing, Prometheus metrics, structured logs.

All observability logic lives here; business code only declares observation
points via :func:`traced` and :func:`span_set`.
"""

from ragforge.observability.logging import (
    configure_logging,
    get_logger,
    get_request_id,
    otel_context_processor,
    set_request_id,
)
from ragforge.observability.metrics import Metrics, get_metrics, start_metrics_server
from ragforge.observability.tracing import (
    get_tracer,
    init_tracing,
    span_set,
    traced,
)

__all__ = [
    "Metrics",
    "configure_logging",
    "get_logger",
    "get_metrics",
    "get_request_id",
    "get_tracer",
    "init_tracing",
    "otel_context_processor",
    "set_request_id",
    "span_set",
    "start_metrics_server",
    "traced",
]


def init_observability(
    *,
    service_name: str = "ragforge",
    otlp_endpoint: str | None = None,
    log_level: str = "INFO",
    metrics_port: int | None = None,
) -> None:
    """One-call setup for tracing, logging and (optionally) the metrics server."""
    init_tracing(service_name=service_name, otlp_endpoint=otlp_endpoint)
    configure_logging(log_level)
    if metrics_port is not None:
        start_metrics_server(metrics_port)
