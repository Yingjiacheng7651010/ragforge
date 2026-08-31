"""Redis-backed answer cache: exact (md5) plus semantic (embedding similarity) lookups.

Layout (all keys under ``key_prefix``):
- ``exact:<md5(query)>``        -> JSON payload (O(1) exact hit)
- ``semantic:<md5(query)>``     -> same payload incl. the query embedding
- ``doc:<doc_id>``              -> Redis set of chunk_ids (for invalidation)

Semantic matching scans the ``semantic:*`` keys and compares cosine
similarity of the query embedding against the stored ones; fine for cache
sizes up to ~10^4 entries, beyond that switch to a vector index (e.g.
RediSearch VSS). Queries flagged as sensitive are never written.
"""

import hashlib
import json
import re
from collections.abc import Sequence

import redis.asyncio as redis_async
import structlog

from ragforge.cache.base import (
    CachedAnswer,
    HitType,
    as_str,
    citation_from_dict,
    citation_to_dict,
    cosine,
    is_sensitive,
)
from ragforge.core.embeddings import EmbeddingProvider
from ragforge.generation import Citation

logger = structlog.get_logger(__name__)


class CacheService:
    """Exact + semantic answer cache with per-document invalidation."""

    def __init__(
        self,
        *,
        redis: redis_async.Redis,
        embedder: EmbeddingProvider,
        threshold: float = 0.92,
        key_prefix: str = "ragforge:cache:",
        ttl_seconds: int | None = None,
        sensitive_patterns: Sequence[re.Pattern[str]] | None = None,
    ) -> None:
        self._redis = redis
        self._embedder = embedder
        self._threshold = threshold
        self._prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._extra_sensitive = list(sensitive_patterns or [])

    async def get(self, query: str) -> CachedAnswer | None:
        """Exact lookup first, then semantic matching; None when nothing hits."""
        raw = await self._redis.get(self._exact_key(query))
        if raw is not None:
            return self._deserialize(as_str(raw), "exact")

        query_embedding = await self._embedder.embed_query(query)
        best_similarity = -1.0
        best_raw: str | None = None
        async for key in self._redis.scan_iter(match=f"{self._semantic_prefix()}*"):
            raw = await self._redis.get(key)
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            stored_embedding = entry.get("embedding") or []
            similarity = cosine(query_embedding, stored_embedding)
            if similarity >= self._threshold and similarity > best_similarity:
                best_similarity = similarity
                best_raw = as_str(raw)
        if best_raw is None:
            return None
        return self._deserialize(best_raw, "semantic")

    async def set(self, query: str, answer: str, citations: Sequence[Citation]) -> None:
        """Cache an answer; sensitive queries are skipped entirely."""
        if is_sensitive(query, self._extra_sensitive):
            logger.warning("not caching sensitive query")
            return

        embedding = await self._embedder.embed_query(query)
        payload = json.dumps(
            {
                "answer": answer,
                "citations": [citation_to_dict(citation) for citation in citations],
                "query": query,
                "embedding": embedding,
            },
            ensure_ascii=False,
        )
        await self._redis.set(self._exact_key(query), payload, ex=self._ttl_seconds)
        await self._redis.set(self._semantic_key(query), payload, ex=self._ttl_seconds)

        # Maintain the doc -> chunks index so updates can invalidate entries.
        for citation in citations:
            if citation.doc_id:
                await self._redis.sadd(self._doc_key(citation.doc_id), citation.chunk_id)

    async def invalidate(self, doc_id: str) -> int:
        """Delete every cached entry whose citations reference chunks of ``doc_id``.

        Returns the number of entries removed.
        """
        chunk_ids = {
            as_str(chunk_id) for chunk_id in await self._redis.smembers(self._doc_key(doc_id))
        }
        if not chunk_ids:
            return 0
        removed = 0
        deleted_entries: set[str] = set()
        for prefix in (self._exact_prefix(), self._semantic_prefix()):
            async for key in self._redis.scan_iter(match=f"{prefix}*"):
                raw = await self._redis.get(key)
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                cited = {citation.get("chunk_id") for citation in entry.get("citations", [])}
                if cited & chunk_ids:
                    await self._redis.delete(key)
                    digest = key.rsplit(":", 1)[-1]
                    if digest not in deleted_entries:
                        deleted_entries.add(digest)
                        removed += 1  # count entries, not keys
        await self._redis.delete(self._doc_key(doc_id))
        return removed

    def _deserialize(self, raw: str, hit_type: HitType) -> CachedAnswer:
        data = json.loads(raw)
        citations = [
            citation_from_dict(dict(item))
            for item in data.get("citations", [])
            if isinstance(item, dict)
        ]
        return CachedAnswer(
            answer=str(data["answer"]),
            citations=citations,
            source_query=str(data["query"]),
            hit_type=hit_type,
        )

    def _exact_key(self, query: str) -> str:
        return f"{self._exact_prefix()}{self._digest(query)}"

    def _semantic_key(self, query: str) -> str:
        return f"{self._semantic_prefix()}{self._digest(query)}"

    def _doc_key(self, doc_id: str) -> str:
        return f"{self._prefix}doc:{doc_id}"

    def _exact_prefix(self) -> str:
        return f"{self._prefix}exact:"

    def _semantic_prefix(self) -> str:
        return f"{self._prefix}semantic:"

    @staticmethod
    def _digest(query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()
