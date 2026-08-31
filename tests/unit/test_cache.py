"""Unit tests for the Redis answer cache using fakeredis."""

import math

import fakeredis.aioredis
import pytest

from ragforge.cache import CacheService
from ragforge.core.embeddings import EmbeddingProvider
from ragforge.generation import Citation


class DictEmbedding(EmbeddingProvider):
    """Scripted embedder: per-text vectors from a dict (records every call)."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = vectors or {}
        self.calls = 0

    @property
    def dimensions(self) -> int:
        return 2

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vectors.get(text, [0.0, 0.0]) for text in texts]


Q_ALPHA = "alpha 问题"
Q_ALPHA_VARIANT = "alpha 的另一个说法"
Q_BETA = "beta 问题"
#: unit vector with cosine([1,0], v) == 0.92 (the threshold boundary)
Q_EDGE_92 = "边界 0.92"
#: unit vector with cosine([1,0], v) == 0.91 (just below the threshold)
Q_EDGE_91 = "边界 0.91"

VECTORS = {
    Q_ALPHA: [1.0, 0.0],
    Q_ALPHA_VARIANT: [1.0, 0.0],
    Q_BETA: [0.0, 1.0],
    Q_EDGE_92: [0.92, math.sqrt(1.0 - 0.92**2)],
    Q_EDGE_91: [0.91, math.sqrt(1.0 - 0.91**2)],
}

CITATION_A = Citation(chunk_id="c1", page=2, text="chunk one", score=0.9, doc_id="doc1")
CITATION_B = Citation(chunk_id="c2", page=None, text="chunk two", score=0.5, doc_id="doc2")


@pytest.fixture
async def service() -> tuple[CacheService, DictEmbedding]:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    embedder = DictEmbedding(VECTORS)
    svc = CacheService(redis=redis, embedder=embedder, threshold=0.92)
    return svc, embedder


async def test_exact_hit_returns_cached_answer(service: tuple[CacheService, DictEmbedding]) -> None:
    svc, embedder = service

    await svc.set(Q_ALPHA, "缓存答案", [CITATION_A])
    cached = await svc.get(Q_ALPHA)

    assert cached is not None
    assert cached.answer == "缓存答案"
    assert cached.hit_type == "exact"
    assert cached.source_query == Q_ALPHA
    assert cached.citations == [CITATION_A]  # round-trips the full citation
    assert embedder.calls == 1  # exact hit short-circuits before embedding


async def test_semantic_hit_matches_similar_query(
    service: tuple[CacheService, DictEmbedding],
) -> None:
    svc, embedder = service

    await svc.set(Q_ALPHA, "缓存答案", [CITATION_A])
    cached = await svc.get(Q_ALPHA_VARIANT)

    assert cached is not None
    assert cached.hit_type == "semantic"
    assert cached.answer == "缓存答案"
    assert cached.source_query == Q_ALPHA  # the originally cached query
    assert embedder.calls == 2  # one embed on set, one on the semantic-miss path


async def test_miss_returns_none(service: tuple[CacheService, DictEmbedding]) -> None:
    svc, _ = service

    await svc.set(Q_ALPHA, "答案", [CITATION_A])
    cached = await svc.get(Q_BETA)

    assert cached is None


async def test_threshold_boundary(service: tuple[CacheService, DictEmbedding]) -> None:
    svc, _ = service

    await svc.set(Q_ALPHA, "答案", [CITATION_A])

    assert (await svc.get(Q_EDGE_92)) is not None  # similarity == 0.92 hits
    assert (await svc.get(Q_EDGE_91)) is None  # similarity == 0.91 misses


async def test_sensitive_query_is_never_cached(service: tuple[CacheService, DictEmbedding]) -> None:
    svc, _ = service
    private_query = "我的邮箱 test@example.com 的密码是什么"

    await svc.set(private_query, "隐私答案", [CITATION_A])

    assert await svc.get(private_query) is None
    assert await svc._redis.keys("*") == []  # nothing written at all


async def test_invalidate_removes_only_affected_entries(
    service: tuple[CacheService, DictEmbedding],
) -> None:
    svc, _ = service

    await svc.set(Q_ALPHA, "答案一", [CITATION_A])  # cites doc1
    await svc.set(Q_BETA, "答案二", [CITATION_B])  # cites doc2

    removed = await svc.invalidate("doc1")

    assert removed == 1  # one cached entry (both its keys) removed
    assert await svc.get(Q_ALPHA) is None
    assert await svc.get(Q_BETA) is not None  # doc2 entry untouched
    assert (await svc.get(Q_BETA)).answer == "答案二"  # type: ignore[union-attr]


async def test_invalidate_without_index_returns_zero(
    service: tuple[CacheService, DictEmbedding],
) -> None:
    svc, _ = service

    assert await svc.invalidate("unknown-doc") == 0


async def test_citations_roundtrip_with_missing_doc_id(
    service: tuple[CacheService, DictEmbedding],
) -> None:
    svc, _ = service
    no_doc = Citation(chunk_id="c9", text="no doc", score=0.1)

    await svc.set(Q_ALPHA, "答案", [CITATION_A, no_doc])
    cached = await svc.get(Q_ALPHA)

    assert cached is not None
    assert cached.citations == [CITATION_A, no_doc]
