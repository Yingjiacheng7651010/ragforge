"""Integration tests against real Milvus / Elasticsearch.

Skipped automatically when no service is reachable. Point the tests at your
instances via ``MILVUS_URI`` and ``ES_URL`` (defaults match local docker
ports), e.g.:

    docker run -d --name milvus -p 19530:19530 milvusdb/milvus:v2.5.4
    docker run -d --name es -p 9200:9200 -e discovery.type=single-node \
        -e xpack.security.enabled=false docker.elastic.co/elasticsearch/elasticsearch:8.17.0
"""

import asyncio
import os
import time
import uuid
from collections.abc import Awaitable, Callable

import pytest

from ragforge.core.vector_store import Filter
from ragforge.ingestion import Chunk
from ragforge.providers import ElasticsearchStore, MilvusVectorStore
from tests.unit.fakes import FakeEmbedding

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
DIMENSION = 3


async def wait_ready(check: Callable[[], Awaitable[object]], timeout: float = 60.0) -> None:
    """Wait until ``check()`` completes without raising (i.e. the service answers)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            await check()
            return
        except Exception:
            await asyncio.sleep(1.0)
    raise RuntimeError("service did not become ready in time")


@pytest.fixture
async def milvus_store():
    store = None
    try:
        store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name=f"test_{uuid.uuid4().hex[:12]}",
            dimension=DIMENSION,
            embedder=FakeEmbedding(dimensions=DIMENSION),
            timeout=10.0,
        )
        await wait_ready(lambda: asyncio.to_thread(store._client.list_collections))
        await store.prepare()
    except Exception as exc:
        if store is not None:
            await store.close()
        pytest.skip(f"Milvus unavailable at {MILVUS_URI}: {exc}")
        return
    yield store
    await store.close()


@pytest.fixture
async def es_store():
    store = ElasticsearchStore(
        hosts=ES_URL,
        index_name=f"test_{uuid.uuid4().hex[:12]}",
        dimension=DIMENSION,
        embedder=FakeEmbedding(dimensions=DIMENSION),
    )
    try:
        await wait_ready(lambda: store._client.info())
        await store.prepare()
    except Exception as exc:
        await store.close()
        pytest.skip(f"Elasticsearch unavailable at {ES_URL}: {exc}")
        return
    yield store
    await store.close()


def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            doc_id=f"doc{i % 2}",
            text=f"chunk {i} contains alpha and beta words",
            heading_path=["Manual", "Intro"],
            page=1,
            metadata={"tenant": f"t{i % 2}"},
        )
        for i in range(4)
    ]


async def exercise_roundtrip(store: object) -> None:
    chunks = sample_chunks()
    await store.add(chunks)  # type: ignore[attr-defined]
    await store.add(chunks)  # type: ignore[attr-defined]  # idempotent upsert

    hits = await store.search([0.1, 0.2, 0.3], top_k=10)  # type: ignore[attr-defined]
    assert {hit.chunk_id for hit in hits} == {chunk.chunk_id for chunk in chunks}

    hits = await store.search([0.1, 0.2, 0.3], top_k=10, filters=Filter(doc_id="doc0"))  # type: ignore[attr-defined]
    assert {hit.chunk_id for hit in hits} == {"c0", "c2"}

    hits = await store.search(  # type: ignore[attr-defined]
        [0.1, 0.2, 0.3],
        top_k=10,
        filters=Filter(metadata={"tenant": "t1"}),
    )
    assert {hit.chunk_id for hit in hits} == {"c1", "c3"}

    await store.delete("doc0")  # type: ignore[attr-defined]
    hits = await store.search([0.1, 0.2, 0.3], top_k=10)  # type: ignore[attr-defined]
    assert {"c0", "c2"} & {hit.chunk_id for hit in hits} == set()


async def test_milvus_roundtrip_filters_and_delete(milvus_store: object) -> None:
    await exercise_roundtrip(milvus_store)


async def test_elasticsearch_roundtrip_filters_and_delete(es_store: object) -> None:
    await exercise_roundtrip(es_store)

    hits = await es_store.search_text("alpha", top_k=10)  # type: ignore[attr-defined]
    assert hits  # BM25 finds the chunks containing "alpha"

    fused = await es_store.search_hybrid([0.1, 0.2, 0.3], "alpha", top_k=10)  # type: ignore[attr-defined]
    assert fused
