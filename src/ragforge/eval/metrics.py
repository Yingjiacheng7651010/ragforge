"""Retrieval metric primitives (pure functions).

Conventions: ``relevant`` is the set of golden chunk ids; ``retrieved`` is
the ranked list of chunk ids returned by the retriever. All functions are
deterministic and side-effect free.
"""

from collections.abc import Sequence


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of relevant chunks present in the top-k results (0 when none)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of top-k results that are relevant (0 when k is 0)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(relevant & set(top)) / len(top)


def mrr(relevant: set[str], retrieved: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result (0 when nothing relevant)."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """1.0 when at least one relevant chunk appears in the top-k, else 0.0."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return 1.0 if relevant & set(retrieved[:k]) else 0.0
