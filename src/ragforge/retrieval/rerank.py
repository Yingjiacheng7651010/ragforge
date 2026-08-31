"""Rerankers: precision stage after recall (cross-encoder scoring)."""

import asyncio
import importlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ragforge.core.errors import RAGForgeError
from ragforge.core.vector_store import SearchHit
from ragforge.observability import span_set, traced


class Reranker(ABC):
    """Score query-chunk pairs and return hits reordered by descending score."""

    @abstractmethod
    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Return the same hits (possibly a subset) reordered by relevance."""
        raise NotImplementedError


class BGEReranker(Reranker):
    """Cross-encoder reranker via sentence-transformers (requires the ``local`` extra).

    Pairs are scored with the model's ``predict`` (logits); scores replace
    the original hit scores. Ranking by raw logits equals ranking by
    sigmoid probabilities, so no calibration is needed.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = self._load_model()

    def _load_model(self) -> Any:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as err:
            raise RAGForgeError(
                "BGEReranker requires the 'local' extra: run `uv sync --extra local`",
                code="E_RERANKER_DEPS_MISSING",
            ) from err
        return module.CrossEncoder(self._model_name, device=self._device)

    @traced("rag.rerank")
    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.text if hit.chunk else "") for hit in hits]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        scored = sorted(
            zip(scores, hits, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        reordered = [
            SearchHit(chunk_id=hit.chunk_id, score=float(score), chunk=hit.chunk)
            for score, hit in scored
        ]
        span_set(query=query, reranked=len(reordered))
        return reordered
