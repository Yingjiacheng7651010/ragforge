"""Unit tests for retrieval: RRF fusion, retrievers, reranking and the funnel."""

from unittest.mock import AsyncMock

import pytest

from ragforge.core.errors import RAGForgeError
from ragforge.core.vector_store import Filter, SearchHit
from ragforge.ingestion import Chunk
from ragforge.retrieval import (
    BGEReranker,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrievalPipeline,
    SparseRetriever,
    rrf_fuse,
)
from tests.unit.fakes import FakeEmbedding, FakeRetriever


def hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, score=score)


def chunked_hit(chunk_id: str, text: str, score: float = 1.0) -> SearchHit:
    chunk = Chunk(chunk_id=chunk_id, doc_id="d", text=text)
    return SearchHit(chunk_id=chunk_id, score=score, chunk=chunk)


# --- rrf fusion ---


def test_rrf_fuse_ranks_both_list_winner_first() -> None:
    dense = [hit("b", 0.9), hit("a", 0.8), hit("c", 0.7)]
    sparse = [hit("b", 9.0), hit("d", 8.0)]

    fused = rrf_fuse([dense, sparse], k=60)

    # b: 1/61 + 1/61; a/d: 1/62; c: 1/63
    assert [h.chunk_id for h in fused] == ["b", "a", "d", "c"]
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_fuse_deduplicates_by_chunk_id() -> None:
    dense = [hit("a", 1.0), hit("b", 0.9), hit("b", 0.8)]
    sparse = [hit("b", 5.0), hit("a", 4.0)]

    fused = rrf_fuse([dense, sparse], k=60)

    ids = [h.chunk_id for h in fused]
    assert len(ids) == len(set(ids)) == 2  # one entry per chunk, no duplicates
    assert set(ids) == {"a", "b"}


def test_rrf_fuse_single_list_is_just_ranked() -> None:
    fused = rrf_fuse([[hit("x", 1.0), hit("y", 0.5)]], k=60)

    assert [h.chunk_id for h in fused] == ["x", "y"]


# --- retrievers ---


async def test_dense_retriever_embeds_query_then_searches() -> None:
    store = AsyncMock()
    store.search.return_value = [hit("c1", 0.9)]
    retriever = DenseRetriever(store=store, embedder=FakeEmbedding(dimensions=3))

    results = await retriever.retrieve("what is rag?", top_k=5, filters=Filter(doc_id="d1"))

    assert results == [hit("c1", 0.9)]
    store.search.assert_awaited_once()
    args = store.search.call_args.args
    assert args[0] == [0.1, 0.2, 0.3]  # embedding from FakeEmbedding (positional)
    assert args[1] == 5
    assert args[2] == Filter(doc_id="d1")


async def test_sparse_retriever_forwards_text_search() -> None:
    store = AsyncMock()
    store.search_text.return_value = [hit("c2", 9.0)]
    retriever = SparseRetriever(store=store)

    results = await retriever.retrieve("bm25 query", top_k=5)

    assert results == [hit("c2", 9.0)]
    store.search_text.assert_awaited_once_with("bm25 query", 5, None)


async def test_hybrid_retriever_fuses_both_legs() -> None:
    dense = FakeRetriever([hit("a", 0.9), hit("b", 0.8)])
    sparse = FakeRetriever([hit("b", 9.0), hit("c", 8.0)])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = await hybrid.retrieve("query", top_k=10, filters=Filter(doc_id="d1"))

    assert [h.chunk_id for h in results] == ["b", "a", "c"]
    assert len(dense.calls) == 1 and len(sparse.calls) == 1
    assert dense.calls[0][1] == 10  # top_k forwarded
    assert dense.calls[0][2] == Filter(doc_id="d1")  # filters forwarded


# --- reranking ---


class FakeCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = list(pairs)
        return list(self.scores)


def make_bge_reranker(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> BGEReranker:
    model = FakeCrossEncoder(scores)
    monkeypatch.setattr(BGEReranker, "_load_model", lambda self: model)
    return BGEReranker(model_name="fake-reranker"), model  # type: ignore[return-value]


async def test_bge_reranker_scores_pairs_and_reorders(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker, model = make_bge_reranker(monkeypatch, [0.3, 0.9, 0.5])
    hits = [
        chunked_hit("c1", "first text"),
        chunked_hit("c2", "second text"),
        chunked_hit("c3", "third text"),
    ]

    reordered = await reranker.rerank("query", hits)

    assert [h.chunk_id for h in reordered] == ["c2", "c3", "c1"]  # sorted by score desc
    assert [h.score for h in reordered] == [0.9, 0.5, 0.3]  # scores replaced
    assert model.pairs == [
        ("query", "first text"),
        ("query", "second text"),
        ("query", "third text"),
    ]


async def test_bge_reranker_handles_empty_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker, _ = make_bge_reranker(monkeypatch, [])

    assert await reranker.rerank("query", []) == []


async def test_bge_reranker_missing_deps_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_fails(name: str) -> None:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr("ragforge.retrieval.rerank.importlib.import_module", import_fails)

    with pytest.raises(RAGForgeError) as exc_info:
        BGEReranker(model_name="fake")

    assert exc_info.value.code == "E_RERANKER_DEPS_MISSING"


# --- two-stage funnel ---


class FakeReranker(Reranker):
    """Reverses the input order (a deterministic fake precision stage)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        self.calls.append((query, len(hits)))
        return list(reversed(hits))


async def test_pipeline_recalls_then_reranks_and_truncates() -> None:
    recalls = [hit(f"c{i}", float(i)) for i in range(50)]
    retriever = FakeRetriever(recalls)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(retriever=retriever, reranker=reranker, recall_k=50, rerank_n=8)

    results = await pipeline.retrieve("query", top_k=10)

    assert len(results) == 8
    assert results[0].chunk_id == "c49"  # reranker reversed the order
    assert retriever.calls[0][1] == 50  # recall width is the configured recall_k
    assert reranker.calls == [("query", 50)]  # reranked the full recall set


async def test_pipeline_without_reranker_truncates_only() -> None:
    recalls = [hit(f"c{i}", float(i)) for i in range(12)]
    pipeline = RetrievalPipeline(retriever=FakeRetriever(recalls), recall_k=50, rerank_n=8)

    results = await pipeline.retrieve("query", top_k=10)

    assert [h.chunk_id for h in results] == [f"c{i}" for i in range(8)]


async def test_pipeline_skips_rerank_on_empty_recall() -> None:
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(retriever=FakeRetriever([]), reranker=reranker)

    results = await pipeline.retrieve("query", top_k=10)

    assert results == []
    assert reranker.calls == []


def test_pipeline_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        RetrievalPipeline(retriever=FakeRetriever(), recall_k=0)
    with pytest.raises(ValueError):
        RetrievalPipeline(retriever=FakeRetriever(), rerank_n=0)
