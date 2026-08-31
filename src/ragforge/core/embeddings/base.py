"""Unified embedding interface: batching, dimension validation, optional caching.

The base class owns batching and validation so every provider gets the same
guarantees: vectors are produced in input order, in bounded batches, and a
mismatch between the declared ``dimensions`` and what the model actually
returns surfaces as ``E_EMBEDDING_DIM_MISMATCH`` instead of corrupting the
index downstream.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable

from ragforge.core.embeddings.cache import EmbeddingCache
from ragforge.core.errors import RAGForgeError


class EmbeddingProvider(ABC):
    """Common interface implemented by every embedding provider."""

    def __init__(self, cache: EmbeddingCache | None = None) -> None:
        self._cache = cache

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Expected number of dimensions per vector."""

    @property
    def cache_key_prefix(self) -> str:
        """Namespace for cache keys so different providers never collide."""
        return type(self).__name__

    @abstractmethod
    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of already-prepared texts, in order."""

    def _prepare_docs(self, texts: list[str]) -> list[str]:
        """Hook for provider-specific document prefixes (default: none)."""
        return texts

    def _prepare_query(self, text: str) -> str:
        """Hook for provider-specific query prefixes (default: none)."""
        return text

    async def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed documents in batches; returns one vector per input text.

        When a cache is configured, document vectors are cached keyed by a
        hash of the raw text (queries are never cached).
        """
        if not texts:
            return []
        if self._cache is None:
            vectors = await self._embed_texts(texts, batch_size, self._prepare_docs)
        else:
            vectors = await self._embed_cached(texts, batch_size)
        self._check_dimensions(vectors)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query (never cached)."""
        vectors = await self._embed_raw([self._prepare_query(text)])
        self._check_dimensions(vectors)
        return vectors[0]

    async def _embed_texts(
        self,
        texts: list[str],
        batch_size: int,
        prepare: Callable[[list[str]], list[str]],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_raw(prepare(batch)))
        return vectors

    async def _embed_cached(self, texts: list[str], batch_size: int) -> list[list[float]]:
        cache = self._cache
        assert cache is not None  # embed() only routes here when a cache is set
        keys = [self._cache_key(text) for text in texts]
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, key in enumerate(keys):
            cached = await cache.get(key)
            if cached is not None:
                results[index] = cached
            else:
                missing.append(index)
        if missing:
            vectors = await self._embed_texts(
                [texts[index] for index in missing],
                batch_size,
                self._prepare_docs,
            )
            for index, vector in zip(missing, vectors, strict=True):
                results[index] = vector
                await cache.set(keys[index], vector)
        return [result for result in results if result is not None]

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.cache_key_prefix}:{digest}"

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        expected = self.dimensions
        for vector in vectors:
            if len(vector) != expected:
                raise RAGForgeError(
                    f"embedding dimension mismatch: expected {expected}, got {len(vector)}",
                    code="E_EMBEDDING_DIM_MISMATCH",
                )
