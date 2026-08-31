"""Targeted tests for small uncovered branches across modules."""

from unittest.mock import patch

import pytest

from ragforge.ingestion import Chunk
from tests.unit.fakes import FakeEmbedding, InMemoryVectorStore


async def test_store_add_with_empty_chunks_is_noop() -> None:
    store = InMemoryVectorStore(embedder=FakeEmbedding(dimensions=3))

    await store.add([])

    assert store._chunks == {}  # type: ignore[attr-defined]


async def test_store_delete_by_doc_id() -> None:
    store = InMemoryVectorStore(embedder=FakeEmbedding(dimensions=3))
    await store.add(
        [
            Chunk(chunk_id="a1", doc_id="d1", text="one"),
            Chunk(chunk_id="b1", doc_id="d2", text="two"),
        ]
    )

    await store.delete("d1")

    assert set(store._chunks) == {"b1"}  # type: ignore[attr-defined]


async def test_store_search_text_substring() -> None:
    store = InMemoryVectorStore(embedder=FakeEmbedding(dimensions=3))
    await store.add(
        [
            Chunk(chunk_id="a1", doc_id="d1", text="RAG 检索增强"),
            Chunk(chunk_id="b1", doc_id="d2", text="无关内容"),
        ]
    )

    hits = await store.search_text("RAG", top_k=5)

    assert [hit.chunk_id for hit in hits] == ["a1"]


async def test_abstract_vector_store_contract() -> None:
    from ragforge.core.vector_store import VectorStore

    class MissingTextSearch(VectorStore):
        async def _upsert(self, chunks: object, vectors: object) -> None: ...

        async def search(self, embedding: object, top_k: int, filters: object = None) -> object:
            return []

        async def search_text(self, text: str, top_k: int, filters: object = None) -> object:
            raise NotImplementedError("search_text not implemented")

        async def delete(self, doc_id: str) -> None: ...

        async def close(self) -> None: ...

    store = MissingTextSearch(embedder=FakeEmbedding(dimensions=3))
    with pytest.raises(NotImplementedError):
        await store.search_text("q", 5)  # contract: missing text search surfaces clearly

    class EmptyStore(VectorStore):
        async def _upsert(self, chunks: object, vectors: object) -> None: ...

        async def search(self, embedding: object, top_k: int, filters: object = None) -> object:
            return []

        async def search_text(self, text: str, top_k: int, filters: object = None) -> object:
            return []

        async def delete(self, doc_id: str) -> None: ...

        async def close(self) -> None: ...

    empty = EmptyStore(embedder=FakeEmbedding(dimensions=3))
    # search_hybrid runs the base RRF fusion over empty legs
    assert await empty.search_hybrid([0.1], "q", 5) == []


def test_fallback_rejects_invalid_breaker_threshold() -> None:
    from ragforge.core.llm.fallback import FallbackLLM
    from tests.unit.fakes import FakeLLM

    with pytest.raises(ValueError):
        FallbackLLM([FakeLLM()], failure_threshold=0)


def test_start_metrics_server_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragforge.observability import metrics as metrics_module

    started: list[int] = []
    monkeypatch.setattr(metrics_module, "start_http_server", lambda port: started.append(port))

    metrics_module.start_metrics_server(9091)

    assert started == [9091]


def test_build_services_enables_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from ragforge.api import services as services_module

    class FakeReranker:
        pass

    monkeypatch.setattr(services_module, "BGEReranker", FakeReranker)
    monkeypatch.setattr(
        services_module,
        "get_settings",
        lambda: SimpleNamespace(
            llm_api_key=SimpleNamespace(get_secret_value=lambda: "sk-test"),
            llm_model="m",
            embedding_model="e",
            embedding_dim=3,
        ),
    )
    monkeypatch.setenv("RAGFORGE_RERANKER", "1")
    monkeypatch.setenv("ES_URL", "http://127.0.0.1:1")

    container = services_module.build_services()

    assert isinstance(container.reranker, FakeReranker)


def test_ingest_task_requires_es_store() -> None:
    from types import SimpleNamespace

    from ragforge.api import tasks

    with patch.object(
        tasks,
        "build_services",
        return_value=SimpleNamespace(es_store=None),
    ), pytest.raises(RuntimeError, match="elasticsearch"):
        tasks.ingest_document("doc-1", "a.md", b"x")


async def test_set_request_id_none_clears_context() -> None:
    from ragforge.observability import get_request_id, set_request_id

    set_request_id("req-1")
    set_request_id(None)

    assert get_request_id() is None


def test_otel_processor_without_span() -> None:
    from ragforge.observability import otel_context_processor

    event = otel_context_processor(None, None, {"event": "no span"})

    assert "trace_id" not in event  # no active span, no correlation ids


def test_retrieval_rerank_unknown_model_path() -> None:
    from ragforge.core.errors import RAGForgeError
    from ragforge.retrieval import BGEReranker

    def import_fails(name: str) -> None:
        raise ImportError(name)

    with (
        patch("ragforge.retrieval.rerank.importlib.import_module", import_fails),
        pytest.raises(RAGForgeError) as exc_info,
    ):
        BGEReranker(model_name="x")

    assert exc_info.value.code == "E_RERANKER_DEPS_MISSING"
