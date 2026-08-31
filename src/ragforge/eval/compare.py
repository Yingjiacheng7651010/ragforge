"""Report diffing: compare a new evaluation report against a baseline."""

from collections.abc import Mapping
from typing import Any


def compare_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Per-metric before/after/delta for every metric present in both reports."""
    diff: dict[str, dict[str, float]] = {}
    for section in ("retrieval", "generation"):
        before = baseline.get(section, {})
        after = current.get(section, {})
        for metric, after_value in after.items():
            before_value = before.get(metric)
            if before_value is not None:
                diff[metric] = {
                    "before": round(float(before_value), 4),
                    "after": round(float(after_value), 4),
                    "delta": round(float(after_value) - float(before_value), 4),
                }
    return diff
