"""Unit tests for vector stores with mocked clients (no services required)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragforge.core.vector_store import Filter, SearchHit, rrf_fuse
from ragforge.ingestion import Chunk
from ragforge.providers import ElasticsearchStore, MilvusVectorStore
from tests.unit.fakes import FakeEmbedding

VECTOR = [0.1, 0.2, 0.3]


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="alpha text",
            heading_path=["H1"],
            page=2,
            metadata={"tenant": "a"},
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d2",
            text="beta text",
            heading_path=["H2"],
            metadata={"tenant": "b"},
        ),
    ]


def milvus_entity() -> dict[str, object]:
    return {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "alpha text",
        "heading_path": ["H1"],
        "page": 2,
        "metadata": {"tenant": "a"},
    }


# --- rrf fusion ---


def test_rrf_fuse_ranks_present_in_both_lists_first() -> None:
    vector_hits = [SearchHit(chunk_id="c1", score=0.9), SearchHit(chunk_id="c2", score=0.8)]
    text_hits = [SearchHit(chunk_id="c2", score=9.0), SearchHit(chunk_id="c3", score=8.0)]

    fused = rrf_fuse(vector_hits, text_hits, k=60)

    assert [hit.chunk_id for hit in fused] == ["c2", "c1", "c3"]


# --- milvus store (client class monkeypatched) ---


def make_milvus(monkeypatch: pytest.MonkeyPatch) -> tuple[MilvusVectorStore, MagicMock]:
    client = MagicMock()
    monkeypatch.setattr("ragforge.providers.milvus_store.MilvusClient", lambda **kwargs: client)
    store = MilvusVectorStore(
        uri="http://fake:19530",
        collection_name="chunks",
        dimension=3,
        embedder=FakeEmbedding(),
    )
    return store, client


async def test_milvus_add_upserts_with_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)

    await store.add(make_chunks())

    client.upsert.assert_called_once()
    _, data = client.upsert.call_args.args
    assert [row["chunk_id"] for row in data] == ["c1", "c2"]
    assert data[0]["vector"] == VECTOR
    assert data[0]["heading_path"] == ["H1"]
    assert data[0]["page"] == 2
    assert data[0]["metadata"] == {"tenant": "a"}
    client.insert.assert_not_called()  # idempotent writes use upsert, never insert


async def test_milvus_prepare_creates_collection_once(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)
    client.has_collection.return_value = False

    await store.prepare()
    await store.prepare()

    assert client.create_collection.call_count == 1
    client.load_collection.assert_called_once()


async def test_milvus_search_translates_filters_to_expr(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)
    client.search.return_value = [[{"id": "c1", "distance": 0.9, "entity": {}}]]

    hits = await store.search(
        VECTOR,
        top_k=5,
        filters=Filter(doc_id="d1", metadata={"tenant": "a"}),
    )

    kwargs = client.search.call_args.kwargs
    assert kwargs["filter"] == 'doc_id == "d1" and metadata["tenant"] == "a"'
    assert kwargs["limit"] == 5
    assert kwargs["consistency_level"] == "Strong"
    assert hits == [SearchHit(chunk_id="c1", score=0.9)]


async def test_milvus_search_maps_entity_to_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)
    client.search.return_value = [[{"id": "c1", "distance": 0.9, "entity": milvus_entity()}]]

    hits = await store.search(VECTOR, top_k=5)

    assert hits[0].chunk is not None
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].chunk.text == "alpha text"
    assert hits[0].chunk.heading_path == ["H1"]
    assert hits[0].chunk.page == 2
    assert hits[0].chunk.metadata == {"tenant": "a"}


async def test_milvus_delete_by_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)

    await store.delete("d1")

    assert client.delete.call_args.kwargs["filter"] == 'doc_id == "d1"'


async def test_milvus_search_text_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = make_milvus(monkeypatch)

    with pytest.raises(NotImplementedError):
        await store.search_text("query", 5)


async def test_milvus_hybrid_degrades_to_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = make_milvus(monkeypatch)
    client.search.return_value = [[{"id": "c1", "distance": 0.9, "entity": {}}]]

    hits = await store.search_hybrid(VECTOR, "query", top_k=5)

    assert client.search.call_count == 1
    assert hits[0].chunk_id == "c1"


# --- elasticsearch store (client instance mocked) ---


def make_es() -> tuple[ElasticsearchStore, AsyncMock]:
    store = ElasticsearchStore(
        hosts="http://fake:9200",
        index_name="chunks",
        dimension=3,
        embedder=FakeEmbedding(),
    )
    client = AsyncMock()
    store._client = client  # type: ignore[attr-defined]
    return store, client


async def test_es_add_bulks_with_chunk_id_as_doc_id() -> None:
    store, client = make_es()

    await store.add(make_chunks())

    ops = client.bulk.call_args.kwargs["operations"]
    assert ops[0] == {"index": {"_index": "chunks", "_id": "c1"}}
    assert ops[1]["vector"] == VECTOR
    assert ops[1]["heading_path"] == ["H1"]
    assert ops[1]["metadata"] == {"tenant": "a"}


async def test_es_search_uses_knn_with_filters() -> None:
    store, client = make_es()
    client.search.return_value = {
        "hits": {"hits": [{"_id": "c1", "_score": 0.8, "_source": {}}]}
    }

    await store.search(VECTOR, top_k=5, filters=Filter(doc_id="d1", metadata={"tenant": "a"}))

    kwargs = client.search.call_args.kwargs
    knn = kwargs["knn"]
    assert knn["field"] == "vector"
    assert knn["query_vector"] == VECTOR
    assert knn["k"] == 5
    assert knn["filter"] == {
        "bool": {"filter": [{"term": {"doc_id": "d1"}}, {"term": {"metadata.tenant": "a"}}]}
    }
    assert kwargs["size"] == 5


async def test_es_search_maps_hits_and_chunks() -> None:
    store, client = make_es()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "c1",
                    "_score": 0.8,
                    "_source": {
                        "chunk_id": "c1",
                        "doc_id": "d1",
                        "text": "alpha text",
                        "heading_path": ["H1"],
                        "page": 2,
                        "metadata": {"tenant": "a"},
                    },
                }
            ]
        }
    }

    hits = await store.search(VECTOR, top_k=5)

    assert hits[0].chunk_id == "c1"
    assert hits[0].score == pytest.approx(0.8)
    assert hits[0].chunk is not None
    assert hits[0].chunk.heading_path == ["H1"]


async def test_es_search_text_uses_match_query() -> None:
    store, client = make_es()
    client.search.return_value = {"hits": {"hits": []}}

    await store.search_text("alpha", top_k=5, filters=Filter(doc_id="d1"))

    kwargs = client.search.call_args.kwargs
    assert kwargs["query"] == {
        "bool": {
            "must": [{"match": {"text": "alpha"}}],
            "filter": {"term": {"doc_id": "d1"}},
        }
    }


async def test_es_delete_by_query() -> None:
    store, client = make_es()

    await store.delete("d1")

    kwargs = client.delete_by_query.call_args.kwargs
    assert kwargs["query"] == {"term": {"doc_id": "d1"}}


async def test_es_hybrid_fuses_vector_and_text() -> None:
    store, client = make_es()
    vector_hits = {
        "hits": {
            "hits": [
                {"_id": "c1", "_score": 1.0, "_source": {}},
                {"_id": "c2", "_score": 0.9, "_source": {}},
            ]
        }
    }
    text_hits = {
        "hits": {
            "hits": [
                {"_id": "c2", "_score": 5.0, "_source": {}},
                {"_id": "c3", "_score": 4.0, "_source": {}},
            ]
        }
    }
    client.search.side_effect = [vector_hits, text_hits]

    hits = await store.search_hybrid(VECTOR, "alpha", top_k=5)

    assert [hit.chunk_id for hit in hits] == ["c2", "c1", "c3"]
