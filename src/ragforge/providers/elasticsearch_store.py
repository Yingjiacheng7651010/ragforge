"""Elasticsearch store: BM25 full-text plus dense kNN, RRF hybrid from the base class."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from elasticsearch import AsyncElasticsearch

from ragforge.core.embeddings import EmbeddingProvider
from ragforge.core.vector_store.base import Filter, SearchHit, VectorStore, chunk_from_entity
from ragforge.ingestion import Chunk


class ElasticsearchStore(VectorStore):
    """Elasticsearch-backed store.

    Every chunk is indexed with a dense vector (kNN search) and a ``text``
    field (BM25); ``search_hybrid`` fuses both via RRF (base class). Writes
    use ``_id = chunk_id`` so re-indexing is idempotent. Filters become
    ``term`` clauses over ``doc_id`` and ``metadata.<key>``.
    """

    def __init__(
        self,
        *,
        hosts: str | list[str],
        index_name: str,
        dimension: int,
        embedder: EmbeddingProvider,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(embedder=embedder, rrf_k=rrf_k)
        self._index_name = index_name
        self._dimension = dimension
        self._client = AsyncElasticsearch(hosts=hosts)

    async def prepare(self) -> None:
        """Create the index with dense-vector + text mappings (idempotent)."""
        if await self._client.indices.exists(index=self._index_name):
            return
        await self._client.indices.create(
            index=self._index_name,
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "heading_path": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "metadata": {"type": "object"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": self._dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        )

    async def _upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        operations: list[dict[str, object]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            operations.append({"index": {"_index": self._index_name, "_id": chunk.chunk_id}})
            operations.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "heading_path": chunk.heading_path,
                    "page": chunk.page,
                    "metadata": chunk.metadata,
                    "vector": list(vector),
                }
            )
        # refresh="wait_for" makes writes visible before add() returns
        # (idempotent upsert semantics); tune for production throughput.
        await self._client.bulk(operations=operations, refresh="wait_for")

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        knn: dict[str, object] = {
            "field": "vector",
            "query_vector": list(embedding),
            "k": top_k,
            "num_candidates": max(top_k * 10, 100),
        }
        es_filter = self._build_filter(filters)
        if es_filter is not None:
            knn["filter"] = es_filter
        response = await self._client.search(index=self._index_name, knn=knn, size=top_k)
        return self._map_hits(response)

    async def search_text(
        self,
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        query: dict[str, object] = {"bool": {"must": [{"match": {"text": text}}]}}
        es_filter = self._build_filter(filters)
        if es_filter is not None:
            cast(dict[str, object], query["bool"])["filter"] = es_filter
        response = await self._client.search(index=self._index_name, query=query, size=top_k)
        return self._map_hits(response)

    async def delete(self, doc_id: str) -> None:
        await self._client.delete_by_query(
            index=self._index_name,
            query={"term": {"doc_id": doc_id}},
            refresh=True,
        )

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _build_filter(filters: Filter | None) -> dict[str, object] | None:
        """Translate filters into ES term clauses (None means no filtering)."""
        if filters is None or filters.is_empty:
            return None
        clauses: list[dict[str, object]] = []
        if filters.doc_id:
            clauses.append({"term": {"doc_id": filters.doc_id}})
        for key, value in filters.metadata.items():
            clauses.append({"term": {f"metadata.{key}": value}})
        if len(clauses) == 1:
            return clauses[0]
        return {"bool": {"filter": clauses}}

    @staticmethod
    def _map_hits(response: object) -> list[SearchHit]:
        """Map an ES search response (dict or ObjectApiResponse) to hits."""
        if hasattr(response, "body"):  # ObjectApiResponse wrapper from the real client
            body = cast(Any, response).body
        else:
            body = cast(Mapping[str, Any], response)
        hits = body["hits"]["hits"]
        results: list[SearchHit] = []
        for hit in hits:
            source = dict(hit.get("_source") or {})
            results.append(
                SearchHit(
                    chunk_id=str(source.get("chunk_id") or hit["_id"]),
                    score=float(hit.get("_score") or 0.0),
                    chunk=chunk_from_entity(source) if source.get("chunk_id") else None,
                )
            )
        return results
