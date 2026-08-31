"""OpenTelemetry tracing: initialization, the ``@traced`` decorator and span helpers."""

import functools
import inspect
import sys
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from ragforge.observability.metrics import get_metrics

_TRACER_NAME = "ragforge"
_initialized = False


def init_tracing(
    *,
    service_name: str = "ragforge",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize the global tracer provider once.

    With ``otlp_endpoint`` set (e.g. ``http://localhost:4317``) spans stream
    to Jaeger/OTLP collectors; otherwise they print to the console for local
    development. Idempotent: later calls are no-ops.
    """
    global _initialized
    if _initialized:
        return
    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service_name}),
    )
    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def span_set(**attributes: object) -> None:
    """Set attributes on the current span (values are JSON-safe serialized)."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        span.set_attribute(key, _serialize(value))


def _serialize(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return list(value)
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _record_error(span: trace.Span) -> None:
    span.set_attribute("error", True)
    exc_info = sys.exc_info()
    if exc_info[1] is not None:
        span.record_exception(exc_info[1])


_F = TypeVar("_F", bound=Callable[..., Any])


def traced(name: str) -> Callable[[_F], _F]:
    """Open a span named ``name`` around the decorated callable.

    Works for sync and async functions; records ``latency_ms`` and error
    status on the span and feeds the query/latency/error metrics.
    """

    def decorator(fn: _F) -> _F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                started = time.perf_counter()
                with tracer.start_as_current_span(name) as span:
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception:
                        _record_error(span)
                        _observe(name, started, "error")
                        raise
                    _observe(name, started, "success")
                    return result

            return cast(_F, async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            started = time.perf_counter()
            with tracer.start_as_current_span(name) as span:
                try:
                    result = fn(*args, **kwargs)
                except Exception:
                    _record_error(span)
                    _observe(name, started, "error")
                    raise
                _observe(name, started, "success")
                return result

        return cast(_F, sync_wrapper)

    return decorator


def _observe(name: str, started: float, status: str) -> None:
    latency_ms = (time.perf_counter() - started) * 1000
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("latency_ms", round(latency_ms, 3))
    get_metrics().record_query(stage=name, latency_ms=latency_ms, status=status)
