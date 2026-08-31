"""Chunking data model, token counting and shared text-splitting utilities.

The text utilities here are shared by every chunker so budget packing,
overlap and fallback splitting behave identically across strategies.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ragforge.ingestion.parsers import ParsedDocument

_T = TypeVar("_T")


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of a document."""

    chunk_id: str
    doc_id: str
    text: str
    parent_id: str | None = None
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


#: Counts tokens in a text; inject a real tokenizer for production.
TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Rough default estimate (~4 chars per token) when no counter is configured."""
    return max(1, len(text) // 4)


class Chunker(ABC):
    """Split a parsed document into chunks (deterministic: same input -> same output)."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        chars_per_token: int = 4,
    ) -> None:
        self._count: TokenCounter = token_counter or estimate_tokens
        self._chars_per_token = chars_per_token

    @abstractmethod
    def split(self, doc: ParsedDocument) -> list[Chunk]:
        """Split ``doc`` into chunks."""
        raise NotImplementedError

    @staticmethod
    def _chunk_id(doc_id: str, heading_path: list[str], index: int) -> str:
        """Idempotent chunk id: sha1 over doc_id + heading path + sequence index."""
        payload = f"{doc_id}|{'/'.join(heading_path)}|{index}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _make_chunk(
        self,
        doc_id: str,
        text: str,
        heading_path: list[str],
        page: int | None,
        index: int,
        *,
        parent_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=self._chunk_id(doc_id, heading_path, index),
            doc_id=doc_id,
            text=text,
            parent_id=parent_id,
            heading_path=list(heading_path),
            page=page,
            metadata=dict(metadata or {}),
        )


# --- shared text splitting utilities ---

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation/line breaks (keeps the punctuation)."""
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_text_by_structure(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
    chars_per_token: int = 4,
) -> list[str]:
    """Split text by paragraphs, packed into ``max_tokens``-sized pieces."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    return pack_units(
        paragraphs,
        max_tokens,
        overlap_tokens,
        count,
        chars_per_token=chars_per_token,
    )


def pack_units(
    units: list[str],
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
    *,
    join_sep: str = "\n\n",
    chars_per_token: int = 4,
) -> list[str]:
    """Pack pre-split units into pieces of at most ``max_tokens`` tokens.

    Units longer than the budget are split via :func:`split_long_unit`; when
    ``overlap_tokens`` > 0 the tail of each piece is re-attached to the head
    of the next one.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    separator_tokens = count(join_sep) if join_sep else 0
    for unit in units:
        unit_tokens = count(unit)
        if unit_tokens > max_tokens:
            if current:
                pieces.append(join_sep.join(current))
                current, current_tokens = [], 0
            pieces.extend(
                split_long_unit(
                    unit,
                    max_tokens,
                    overlap_tokens,
                    count,
                    chars_per_token=chars_per_token,
                )
            )
            continue
        if current and current_tokens + unit_tokens + separator_tokens > max_tokens:
            pieces.append(join_sep.join(current))
            current, current_tokens = overlap_head(
                pieces[-1],
                overlap_tokens,
                chars_per_token,
                count,
            )
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        pieces.append(join_sep.join(current))
    return pieces


def split_long_unit(
    unit: str,
    max_tokens: int,
    overlap_tokens: int,
    count: TokenCounter,
    *,
    chars_per_token: int = 4,
) -> list[str]:
    """Split a unit that alone exceeds the budget: by sentence, then by char window."""
    sentences = split_sentences(unit)
    if len(sentences) > 1:
        return pack_units(
            sentences,
            max_tokens,
            overlap_tokens,
            count,
            join_sep=" ",
            chars_per_token=chars_per_token,
        )
    return split_by_chars(unit, max_tokens, overlap_tokens, chars_per_token)


def split_by_chars(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    chars_per_token: int,
) -> list[str]:
    """Fallback: fixed character windows with optional overlap between windows."""
    window = max(1, max_tokens * chars_per_token)
    step = max(1, window - overlap_tokens * chars_per_token)
    return [text[i : i + window] for i in range(0, len(text), step)]


def overlap_head(
    previous: str,
    overlap_tokens: int,
    chars_per_token: int,
    count: TokenCounter,
) -> tuple[list[str], int]:
    """Return the tail of ``previous`` (approx ``overlap_tokens``) as the next piece's head."""
    if overlap_tokens <= 0:
        return [], 0
    tail = previous[-overlap_tokens * chars_per_token :]
    if not tail:
        return [], 0
    return [tail], count(tail)
