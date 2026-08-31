"""Unit tests for observability: traced spans, log processors, metrics."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ragforge.observability import (
    get_metrics,
    otel_context_processor,
    set_request_id,
    span_set,
    traced,
)


@pytest.fixture(scope="module")
def exporter() -> InMemorySpanExporter:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace.set_tracer_provider(provider)
    return memory


@traced("rag.parent")
async def parent_fn() -> int:
    await child_fn()
    span_set(parent_attr="yes")
    return 42


@traced("rag.child")
async def child_fn() -> str:
    return "child"


@traced("rag.sync")
def sync_fn(value: int) -> int:
    return value * 2


@traced("rag.failing")
async def failing_fn() -> None:
    raise ValueError("boom")


async def test_traced_decorator_builds_span_tree(exporter: InMemorySpanExporter) -> None:
    exporter.clear()

    assert await parent_fn() == 42

    spans = {span.name: span for span in exporter.get_finished_spans()}
    parent, child = spans["rag.parent"], spans["rag.child"]
    # parent-child linkage: same trace, child points at the parent span
    assert child.context.trace_id == parent.context.trace_id
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
    # decorator records latency and span_set attributes
    assert "latency_ms" in parent.attributes
    assert parent.attributes["parent_attr"] == "yes"


async def test_traced_sync_function(exporter: InMemorySpanExporter) -> None:
    exporter.clear()

    assert sync_fn(21) == 42

    spans = [s for s in exporter.get_finished_spans() if s.name == "rag.sync"]
    assert len(spans) == 1
    assert "latency_ms" in spans[0].attributes


async def test_traced_records_error_and_propagates(exporter: InMemorySpanExporter) -> None:
    exporter.clear()

    with pytest.raises(ValueError, match="boom"):
        await failing_fn()

    span = [s for s in exporter.get_finished_spans() if s.name == "rag.failing"][-1]
    assert span.attributes.get("error") is True
    assert span.events  # recorded exception event


def test_log_processor_adds_trace_id(exporter: InMemorySpanExporter) -> None:
    with trace.get_tracer("test").start_as_current_span("log-span") as span:
        event = otel_context_processor(None, None, {"event": "hello"})

    assert event["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert event["span_id"] == format(span.get_span_context().span_id, "016x")


def test_log_processor_adds_request_id() -> None:
    set_request_id("req-123")

    event = otel_context_processor(None, None, {"event": "hello"})

    assert event["request_id"] == "req-123"
    set_request_id(None)


def test_metrics_record_query_and_cache() -> None:
    metrics = get_metrics()
    # metrics are process-wide singletons; assert deltas from the current values
    before_success = metrics.queries.labels(stage="test.stage", status="success")._value.get()
    before_error = metrics.queries.labels(stage="test.stage", status="error")._value.get()
    before_errors = metrics.errors.labels(stage="test.stage")._value.get()
    before_exact = metrics.cache_lookups.labels(outcome="exact")._value.get()
    before_miss = metrics.cache_lookups.labels(outcome="miss")._value.get()
    before_lookups = metrics.cache_queries._value.get()
    before_cost_sum = metrics.query_cost._sum.get()

    metrics.record_query(stage="test.stage", latency_ms=123.4, status="success")
    metrics.record_query(stage="test.stage", latency_ms=5.0, status="error")
    metrics.record_cache("exact")
    metrics.record_cache("miss")
    metrics.record_cost(0.001)

    queries = metrics.queries.labels
    assert queries(stage="test.stage", status="success")._value.get() == before_success + 1
    assert queries(stage="test.stage", status="error")._value.get() == before_error + 1
    assert metrics.errors.labels(stage="test.stage")._value.get() == before_errors + 1
    assert metrics.cache_lookups.labels(outcome="exact")._value.get() == before_exact + 1
    assert metrics.cache_lookups.labels(outcome="miss")._value.get() == before_miss + 1
    assert metrics.cache_queries._value.get() == before_lookups + 2
    assert metrics.query_cost._sum.get() == pytest.approx(before_cost_sum + 0.001)


@traced("rag.noop")
async def noop_fn() -> str:
    return "fine"


async def test_traced_works_without_provider() -> None:
    # Runs before any provider is installed in this process: spans are no-ops.
    assert await noop_fn() == "fine"
