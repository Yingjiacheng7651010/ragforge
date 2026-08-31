"""Retrieval strategies over the vector store abstraction."""

import asyncio
from abc import ABC, abstractmethod

from ragforge.core.embeddings import EmbeddingProvider
from ragforge.core.vector_store import Filter, SearchHit, VectorStore, rrf_fuse


class Retriever(ABC):
    """Common interface: turn a query string into ranked hits."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        """Retrieve up to ``top_k`` hits for ``query``, sorted by descending score."""
        raise NotImplementedError


class DenseRetriever(Retriever):
    """Vector (dense) retrieval: embed the query, then search the store."""

    def __init__(self, *, store: VectorStore, embedder: EmbeddingProvider) -> None:
        self._store = store
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        embedding = await self._embedder.embed_query(query)
        return await self._store.search(embedding, top_k, filters)


class SparseRetriever(Retriever):
    """Lexical (sparse) retrieval: BM25-style text search on the store."""

    def __init__(self, *, store: VectorStore) -> None:
        self._store = store

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        return await self._store.search_text(query, top_k, filters)


class HybridRetriever(Retriever):
    """Fuse dense and sparse results with RRF (both legs run concurrently)."""

    def __init__(self, *, dense: Retriever, sparse: Retriever, rrf_k: int = 60) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense.retrieve(query, top_k, filters),
            self._sparse.retrieve(query, top_k, filters),
        )
        return rrf_fuse([dense_hits, sparse_hits], k=self._rrf_k)
