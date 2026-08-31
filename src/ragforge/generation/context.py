"""Context assembly: budget-aware selection of chunks with citations."""

from collections.abc import Sequence

from ragforge.core.vector_store import SearchHit
from ragforge.generation.base import Citation
from ragforge.ingestion import Chunk
from ragforge.ingestion.chunking import TokenCounter, estimate_tokens


class ContextAssembler:
    """Build the prompt context from retrieval hits, never exceeding the token budget.

    Selection pipeline:
    1. parent-child restoration: a hit whose chunk is a child is replaced by
       its parent when the parent is also among the hits;
    2. deduplication: by restored chunk_id, then by identical text (multiple
       children of one parent collapse into a single entry);
    3. sort by hit score descending, then greedily fill until the token
       budget is exhausted (the "[n] " prefix counts towards the budget);
    4. number the selected chunks [1][2]... and return matching citations.
    """

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
        self._count: TokenCounter = token_counter or estimate_tokens

    def assemble(
        self,
        query: str,
        hits: Sequence[SearchHit],
        max_tokens: int,
    ) -> tuple[str, list[Citation]]:
        """Return ``(context_text, citations)``; citation ``i`` maps to marker ``[i+1]``."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        selected = self._select(hits, max_tokens)
        context = "\n\n".join(
            f"[{index + 1}] {chunk.text}" for index, (chunk, _) in enumerate(selected)
        )
        citations = [
            Citation(chunk_id=chunk.chunk_id, page=chunk.page, text=chunk.text, score=score)
            for chunk, score in selected
        ]
        return context, citations

    def _select(self, hits: Sequence[SearchHit], max_tokens: int) -> list[tuple[Chunk, float]]:
        candidates = self._dedupe(hits)
        selected: list[tuple[Chunk, float]] = []
        used_tokens = 0
        for chunk, score in candidates:
            numbered = f"[{len(selected) + 1}] {chunk.text}"
            tokens = self._count(numbered)
            if tokens > max_tokens:
                continue  # a single chunk that does not fit is skipped
            if used_tokens + tokens > max_tokens:
                break  # budget exhausted; candidates are score-sorted, so stop
            selected.append((chunk, score))
            used_tokens += tokens
        return selected

    def _dedupe(self, hits: Sequence[SearchHit]) -> list[tuple[Chunk, float]]:
        """Restore parents, dedupe by chunk_id then by identical text, keep best score."""
        chunks_by_id = {hit.chunk.chunk_id: hit.chunk for hit in hits if hit.chunk is not None}
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        candidates: list[tuple[Chunk, float]] = []
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            chunk = hit.chunk
            if chunk is None:
                continue
            if chunk.parent_id is not None and chunk.parent_id in chunks_by_id:
                chunk = chunks_by_id[chunk.parent_id]
            if chunk.chunk_id in seen_ids or chunk.text in seen_texts:
                continue
            seen_ids.add(chunk.chunk_id)
            seen_texts.add(chunk.text)
            candidates.append((chunk, hit.score))
        return candidates
