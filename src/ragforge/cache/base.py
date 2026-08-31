"""Cache data model, sensitive-query detection and similarity helpers."""

import math
import re
from dataclasses import dataclass
from typing import Literal, cast

from ragforge.generation import Citation

HitType = Literal["exact", "semantic"]


@dataclass(frozen=True)
class CachedAnswer:
    """A cache hit: the answer plus the citations it references."""

    answer: str
    citations: list[Citation]
    source_query: str
    hit_type: HitType


#: Privacy guards: queries matching any of these are never cached.
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email address
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),  # mainland China mobile number
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),  # Chinese national ID
)


def is_sensitive(query: str, extra_patterns: list[re.Pattern[str]] | None = None) -> bool:
    """True when the query contains personal data that must not be cached."""
    patterns = _SENSITIVE_PATTERNS + tuple(extra_patterns or [])
    return any(pattern.search(query) for pattern in patterns)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 on empty/mismatched input)."""
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def citation_to_dict(citation: Citation) -> dict[str, object]:
    return {
        "chunk_id": citation.chunk_id,
        "page": citation.page,
        "text": citation.text,
        "score": citation.score,
        "doc_id": citation.doc_id,
    }


def citation_from_dict(data: dict[str, object]) -> Citation:
    page_raw = data.get("page")
    score_raw = data.get("score")
    return Citation(
        chunk_id=str(data.get("chunk_id", "")),
        page=cast(int, page_raw) if page_raw is not None else None,
        text=str(data.get("text", "")),
        score=cast(float, score_raw) if score_raw is not None else 0.0,
        doc_id=str(data["doc_id"]) if data.get("doc_id") else None,
    )


def as_str(raw: bytes | str) -> str:
    """Normalize a redis value to str regardless of decode_responses."""
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw
