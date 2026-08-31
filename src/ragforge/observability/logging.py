"""Structured logging: structlog with trace_id/span_id/request_id on every event."""

import contextvars
from collections.abc import MutableMapping
from typing import Any

import structlog
from opentelemetry import trace

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ragforge_request_id", default=None
)


def set_request_id(request_id: str | None) -> None:
    """Set the request id for the current async/sync context (per-request)."""
    if request_id is None:
        _request_id_var.set(None)
    else:
        _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


def otel_context_processor(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Attach trace_id/span_id (from the active OTel span) and request_id to log events."""
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    request_id = _request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to emit JSON with correlation ids (idempotent)."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            otel_context_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level.upper()),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
