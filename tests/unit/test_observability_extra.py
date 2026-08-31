"""Additional observability tests: init paths, span attributes, one-call setup."""

from unittest.mock import MagicMock, patch

from opentelemetry import trace

from ragforge.observability import (
    init_observability,
    init_tracing,
    span_set,
    traced,
)


def test_init_tracing_without_endpoint_uses_console() -> None:
    from ragforge.observability import tracing

    created: list[object] = []

    def fake_console(*args: object, **kwargs: object) -> object:
        created.append(True)
        return MagicMock()  # exporter duck-type

    with (
        patch.object(tracing, "_initialized", False),
        patch.object(tracing, "ConsoleSpanExporter", fake_console),
    ):
        init_tracing(service_name="test-service")

    assert created, "ConsoleSpanExporter must be instantiated"


def test_init_tracing_with_otlp_endpoint() -> None:
    from ragforge.observability import tracing

    created: list[object] = []

    def fake_otlp(*args: object, **kwargs: object) -> object:
        created.append(kwargs)
        return MagicMock()  # exporter duck-type with shutdown/export

    with (
        patch.object(tracing, "_initialized", False),
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
            fake_otlp,
        ),
    ):
        init_tracing(service_name="s", otlp_endpoint="http://localhost:4317")

    assert created == [{"endpoint": "http://localhost:4317"}]


def test_init_tracing_is_idempotent() -> None:
    from ragforge.observability import tracing

    calls = {"console": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["console"] += 1
        return MagicMock()

    with (
        patch.object(tracing, "_initialized", False),
        patch.object(tracing, "ConsoleSpanExporter", counting),
    ):
        init_tracing(service_name="s")
        init_tracing(service_name="s")  # second call is a no-op

    assert calls["console"] == 1


def test_span_set_serializes_structured_values() -> None:
    tracer = trace.get_tracer("test-serialization")

    attributes: dict[str, object] = {}
    with tracer.start_as_current_span("attrs") as span:
        span_set(query="q", top_k=5, scores=[0.9, 0.8], nested={"a": 1})
        for key, value in span.attributes.items():
            attributes[key] = value

    assert attributes["query"] == "q"
    assert attributes["top_k"] == 5
    assert list(attributes["scores"]) == [0.9, 0.8]
    assert isinstance(attributes["nested"], str)  # dicts become JSON


def test_init_observability_wires_everything() -> None:
    with (
        patch("ragforge.observability.init_tracing") as init_tracing_mock,
        patch("ragforge.observability.configure_logging") as configure_mock,
        patch("ragforge.observability.start_metrics_server") as metrics_mock,
    ):
        init_observability(
            service_name="svc",
            otlp_endpoint="http://localhost:4317",
            log_level="DEBUG",
            metrics_port=8001,
        )

    init_tracing_mock.assert_called_once_with(
        service_name="svc",
        otlp_endpoint="http://localhost:4317",
    )
    configure_mock.assert_called_once_with("DEBUG")
    metrics_mock.assert_called_once_with(8001)


@traced("rag.extra")
async def extra_fn() -> str:
    return "ok"


async def test_traced_still_works_after_init() -> None:
    assert await extra_fn() == "ok"
