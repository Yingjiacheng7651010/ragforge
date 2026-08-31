"""Vector store abstraction: search hits, filters, and the store interface.

The base class owns embedding (via an injected ``EmbeddingProvider``) and
the RRF fusion used by ``search_hybrid``; concrete stores only implement
storage-level operations, so callers never import provider clients.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from ragforge.core.embeddings import EmbeddingProvider
from ragforge.ingestion import Chunk


@dataclass(frozen=True)
class SearchHit:
    """A retrieval result; ``chunk`` is populated when the store returns entities."""

    chunk_id: str
    score: float
    chunk: Chunk | None = None


@dataclass(frozen=True)
class Filter:
    """Filter for retrieval: by ``doc_id`` and/or arbitrary metadata fields.

    Used for document-level permission isolation: combine ``doc_id`` with
    tenant/owner metadata keys to scope every query.
    """

    doc_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.doc_id is None and not self.metadata


class VectorStore(ABC):
    """Common interface for vector stores (Milvus, Elasticsearch, ...)."""

    def __init__(self, *, embedder: EmbeddingProvider, rrf_k: int = 60) -> None:
        self._embedder = embedder
        self._rrf_k = rrf_k

    async def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and upsert chunks (idempotent: re-adding overwrites by chunk_id)."""
        if not chunks:
            return
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        await self._upsert(list(chunks), vectors)

    @abstractmethod
    async def _upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Persist chunks with their vectors, overwriting by chunk_id."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        """Vector search; results are sorted by descending score."""
        raise NotImplementedError

    @abstractmethod
    async def search_text(
        self,
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        """Full-text (BM25) search; sorted by descending score."""
        raise NotImplementedError

    async def search_hybrid(
        self,
        embedding: Sequence[float],
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        """Fuse vector and text results with reciprocal rank fusion."""
        vector_hits = await self.search(embedding, top_k, filters)
        text_hits = await self.search_text(text, top_k, filters)
        return rrf_fuse(vector_hits, text_hits, self._rrf_k)

    @abstractmethod
    async def delete(self, doc_id: str) -> None:
        """Delete every chunk belonging to ``doc_id``."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Release client resources."""
        raise NotImplementedError


def rrf_fuse(
    vector_hits: Sequence[SearchHit],
    text_hits: Sequence[SearchHit],
    k: int,
) -> list[SearchHit]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk | None] = {}
    for rank, hit in enumerate(vector_hits):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunks.setdefault(hit.chunk_id, hit.chunk)
    for rank, hit in enumerate(text_hits):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunks.setdefault(hit.chunk_id, hit.chunk)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        SearchHit(chunk_id=chunk_id, score=score, chunk=chunks[chunk_id])
        for chunk_id, score in ranked
    ]


def chunk_from_entity(entity: Mapping[str, object]) -> Chunk:
    """Rebuild a Chunk from a store entity (Milvus output_fields / ES _source)."""
    page_raw = entity.get("page")
    metadata_raw = entity.get("metadata")
    return Chunk(
        chunk_id=str(entity["chunk_id"]),
        doc_id=str(entity["doc_id"]),
        text=str(entity["text"]),
        heading_path=cast(list[str], entity.get("heading_path") or []),
        page=cast(int, page_raw) if page_raw is not None else None,
        metadata=dict(cast(dict[str, object], metadata_raw)) if metadata_raw else {},
    )
