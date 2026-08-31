"""Prometheus metrics for the query lifecycle.

Counter/Histogram primitives; derived rates are computed in PromQL, e.g.:
- QPS:            rate(rag_queries_total[1m])
- P50/P95/P99:    histogram_quantile(0.50/0.95/0.99, rag_query_latency_seconds_bucket)
- error rate:     rag_errors_total / rag_queries_total
- cache hit rate: rag_cache_hits_total / rag_cache_queries_total
"""

from prometheus_client import Counter, Histogram, start_http_server


class Metrics:
    """Process-wide metrics registry (use :func:`get_metrics`)."""

    def __init__(self, prefix: str = "rag") -> None:
        self.queries = Counter(
            f"{prefix}_queries_total",
            "Query-stage executions",
            ["stage", "status"],
        )
        self.errors = Counter(
            f"{prefix}_errors_total",
            "Query-stage errors",
            ["stage"],
        )
        self.latency = Histogram(
            f"{prefix}_query_latency_seconds",
            "Query-stage latency",
            ["stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )
        self.cache_lookups = Counter(
            f"{prefix}_cache_lookups_total",
            "Answer-cache lookups by outcome",
            ["outcome"],
        )
        self.cache_queries = Counter(
            f"{prefix}_cache_queries_total",
            "All answer-cache lookups",
        )
        self.query_cost = Histogram(
            f"{prefix}_query_cost_usd",
            "USD cost per generated answer",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
        )

    def record_query(self, *, stage: str, latency_ms: float, status: str) -> None:
        self.queries.labels(stage=stage, status=status).inc()
        self.latency.labels(stage=stage).observe(latency_ms / 1000)
        if status == "error":
            self.errors.labels(stage=stage).inc()

    def record_cache(self, outcome: str) -> None:
        self.cache_queries.inc()
        self.cache_lookups.labels(outcome=outcome).inc()

    def record_cost(self, usd: float) -> None:
        self.query_cost.observe(usd)


_instance: Metrics | None = None


def get_metrics() -> Metrics:
    """Return the process-wide metrics singleton."""
    global _instance
    if _instance is None:
        _instance = Metrics()
    return _instance


def start_metrics_server(port: int) -> None:
    """Expose ``/metrics`` on ``port`` via a background HTTP thread."""
    start_http_server(port)
