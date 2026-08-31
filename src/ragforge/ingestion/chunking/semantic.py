"""Semantic chunker: split at low neighbor-sentence similarity (embedding cosine)."""

import math
from collections.abc import Callable

from ragforge.ingestion.chunking.base import (
    Chunk,
    Chunker,
    TokenCounter,
    split_long_unit,
    split_sentences,
)
from ragforge.ingestion.parsers import ParsedDocument


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _slice_at(items: list[str], boundaries: list[int]) -> list[list[str]]:
    groups: list[list[str]] = []
    start = 0
    for boundary in boundaries:
        groups.append(items[start : boundary + 1])
        start = boundary + 1
    groups.append(items[start:])
    return [group for group in groups if group]


class SemanticChunker(Chunker):
    """Split sections at sentence boundaries with low neighbor cosine similarity.

    ``embed_fn(texts) -> vectors`` must be a synchronous embedding function
    (adapt an async provider in the caller, e.g. via ``asyncio.run``).
    Boundaries between similar sentences are kept together; the token budget
    is a backstop that further cuts oversized semantic groups by sentence.
    """

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]],
        max_tokens: int,
        threshold: float = 0.75,
        overlap_tokens: int = 0,
        token_counter: TokenCounter | None = None,
        chars_per_token: int = 4,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        super().__init__(token_counter=token_counter, chars_per_token=chars_per_token)
        self._embed_fn = embed_fn
        self._max_tokens = max_tokens
        self._threshold = threshold
        self._overlap_tokens = overlap_tokens

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in doc.sections:
            sentences = split_sentences(section.text)
            if not sentences:
                continue
            embeddings = self._embed_fn(sentences)
            boundaries = [
                i
                for i in range(len(sentences) - 1)
                if _cosine(embeddings[i], embeddings[i + 1]) < self._threshold
            ]
            for group in _slice_at(sentences, boundaries):
                group_text = " ".join(group)
                for piece in split_long_unit(
                    group_text,
                    self._max_tokens,
                    self._overlap_tokens,
                    self._count,
                    chars_per_token=self._chars_per_token,
                ):
                    chunks.append(
                        self._make_chunk(
                            doc.doc_id,
                            piece,
                            section.heading_path,
                            section.page,
                            index,
                            metadata={"strategy": "semantic"},
                        )
                    )
                    index += 1
        return chunks
