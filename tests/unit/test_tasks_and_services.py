"""Unit tests for Celery tasks and the production service assembly."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ragforge.api.services import AppServices, build_services, new_doc_id
from ragforge.api.tasks import CeleryIngestor, ingest_document
from ragforge.core.errors import RAGForgeError
from tests.unit.fakes import FakeEmbedding, InMemoryVectorStore

# --- ingest task ---


def test_ingest_document_parses_chunks_and_indexes() -> None:
    from ragforge.api import tasks

    embedder = FakeEmbedding(dimensions=3)
    store = InMemoryVectorStore(embedder=embedder)
    fake_services = SimpleNamespace(es_store=store)
    with patch.object(tasks, "build_services", return_value=fake_services):
        result = ingest_document(
            "doc-1",
            "guide.md",
            "# 标题\n\n这是正文内容，包含 RAG 相关信息。".encode(),
        )

    assert result["status"] == "indexed"
    assert result["chunks"] >= 1
    assert len(store._chunks) >= 1  # type: ignore[attr-defined]


def test_new_doc_id_is_unique() -> None:
    assert new_doc_id() != new_doc_id()


# --- CeleryIngestor ---


def test_celery_ingestor_submits_task() -> None:
    from ragforge.api import tasks

    ingestor = CeleryIngestor()
    with patch.object(tasks.celery_app, "send_task") as send:
        ingestor.submit("doc-1", "a.md", b"content")

    send.assert_called_once_with(
        "ragforge.ingest_document",
        args=["doc-1", "a.md", b"content"],
    )


async def test_celery_ingestor_reads_status() -> None:
    from ragforge.api import tasks

    class FakeAsyncResult:
        state = "SUCCESS"
        info = {"chunks": 3}

    ingestor = CeleryIngestor()
    with patch.object(tasks.celery_app, "AsyncResult", return_value=FakeAsyncResult()):
        state, info = await ingestor.status("doc-1")

    assert state == "SUCCESS"
    assert info == {"chunks": 3}


# --- build_services ---


def fake_settings() -> SimpleNamespace:
    from pydantic import SecretStr

    return SimpleNamespace(
        llm_api_key=SecretStr("sk-test"),
        llm_model="test-model",
        embedding_model="test-embedding",
        embedding_dim=3,
    )


def test_build_services_assembles_full_container() -> None:
    from ragforge.api import services as services_module

    with (
        patch.object(services_module, "get_settings", return_value=fake_settings()),
        patch.dict("os.environ", {"ES_URL": "http://127.0.0.1:1", "REDIS_URL": "redis://127.0.0.1:1/0"}),
    ):
        container = build_services()

    assert isinstance(container, AppServices)
    assert container.input_guard is not None
    assert container.output_guard is not None
    assert container.cache is not None
    assert container.understanding is not None
    assert container.pipeline is not None
    assert container.generator is not None
    assert container.ingestor is not None


def test_build_services_requires_api_key() -> None:
    from ragforge.api import services as services_module

    with patch.object(
        services_module,
        "get_settings",
        return_value=SimpleNamespace(llm_api_key=SimpleNamespace(get_secret_value=lambda: "")),
    ), pytest.raises(RAGForgeError) as exc_info:
        build_services()

    assert exc_info.value.code == "E_API_CONFIG"


async def test_check_health_includes_elasticsearch() -> None:
    container = AppServices(
        input_guard=None,  # type: ignore[arg-type]
        output_guard=None,  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        understanding=None,  # type: ignore[arg-type]
        pipeline=None,  # type: ignore[arg-type]
        generator=None,  # type: ignore[arg-type]
        ingestor=None,  # type: ignore[arg-type]
        redis=AsyncMock(),
        es_store=AsyncMock(),
    )
    container.redis.ping = AsyncMock(return_value=True)
    container.es_store.ping = AsyncMock(return_value=True)

    checks = await container.check_health()

    assert checks == {"api": "ok", "redis": "ok", "elasticsearch": "ok"}


async def test_check_health_marks_downstream_errors() -> None:
    container = AppServices(
        input_guard=None,  # type: ignore[arg-type]
        output_guard=None,  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        understanding=None,  # type: ignore[arg-type]
        pipeline=None,  # type: ignore[arg-type]
        generator=None,  # type: ignore[arg-type]
        ingestor=None,  # type: ignore[arg-type]
        redis=AsyncMock(),
        es_store=AsyncMock(),
    )
    container.redis.ping = AsyncMock(side_effect=ConnectionError("down"))
    container.es_store.ping = AsyncMock(side_effect=ConnectionError("down"))

    checks = await container.check_health()

    assert checks == {"api": "ok", "redis": "error", "elasticsearch": "error"}
