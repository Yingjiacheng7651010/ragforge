"""Corrective RAG: rewrite-and-retry on partial relevance, declare insufficiency otherwise."""

from collections.abc import Sequence

import structlog

from ragforge.core.vector_store import Filter, SearchHit
from ragforge.ingestion import Chunk
from ragforge.retrieval.retrievers import Retriever
from ragforge.retrieval.self_rag.base import SelfRagAssessment, SelfRagEvaluator

logger = structlog.get_logger(__name__)


class CorrectiveRagRetriever(Retriever):
    """Self-assess retrieval and correct the query before giving up.

    Flow per attempt: retrieve -> evaluate. ``sufficient`` returns the hits
    filtered to the relevant chunks; ``retry`` re-retrieves with the
    refined query (bounded by ``max_retries``, at most 2, to avoid loops);
    ``insufficient`` (or retries exhausted) returns an empty list and marks
    the outcome in :attr:`last_assessment` so callers can tell the user the
    knowledge base lacks the material.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        evaluator: SelfRagEvaluator,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0 or max_retries > 2:
            raise ValueError("max_retries must be between 0 and 2 (avoid retry loops)")
        self._retriever = retriever
        self._evaluator = evaluator
        self._max_retries = max_retries
        self.last_assessment: SelfRagAssessment | None = None

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        current_query = query
        self.last_assessment = None

        for _ in range(self._max_retries + 1):
            hits = await self._retriever.retrieve(current_query, top_k, filters)
            chunks = [hit.chunk for hit in hits if hit.chunk is not None]
            assessment = await self._evaluator.evaluate(current_query, chunks)
            self.last_assessment = assessment

            if assessment.verdict == "sufficient":
                return self._filter_relevant(hits, chunks, assessment.relevance)

            if assessment.verdict == "retry" and assessment.refined_query:
                logger.warning(
                    "retrieval partially relevant; retrying with refined query",
                    refined_query=assessment.refined_query,
                )
                current_query = assessment.refined_query
                continue

            break  # insufficient, or retry without a refined query

        logger.warning(
            "retrieval deemed insufficient after attempts; returning empty",
            query=query,
            attempts=self._max_retries + 1,
        )
        return []

    def _filter_relevant(
        self,
        hits: Sequence[SearchHit],
        chunks: Sequence[Chunk],
        relevance: Sequence[bool],
    ) -> list[SearchHit]:
        """Keep only hits whose chunk was marked relevant (order preserved)."""
        relevant_ids = {
            chunk.chunk_id
            for chunk, is_relevant in zip(chunks, relevance, strict=True)
            if is_relevant
        }
        return [hit for hit in hits if hit.chunk is not None and hit.chunk.chunk_id in relevant_ids]
