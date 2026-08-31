"""Milvus vector store (pymilvus): HNSW index, scalar doc_id, JSON metadata filters."""

import asyncio
from collections.abc import Sequence

from pymilvus import DataType, MilvusClient

from ragforge.core.embeddings import EmbeddingProvider
from ragforge.core.vector_store.base import Filter, SearchHit, VectorStore, chunk_from_entity
from ragforge.ingestion import Chunk

_CHUNK_ID_MAX_LEN = 64
_DOC_ID_MAX_LEN = 255
_TEXT_MAX_LEN = 65535


class MilvusVectorStore(VectorStore):
    """Milvus-backed store with a HNSW/COSINE index.

    Schema: ``chunk_id`` (primary key), ``doc_id``, ``text``, ``heading_path``
    (JSON), ``page`` (nullable), ``metadata`` (JSON) and the float vector.
    Writes are upserts keyed by ``chunk_id`` (idempotent). Filters translate
    to Milvus expressions over the scalar fields and JSON metadata keys.
    Milvus has no built-in BM25 in this setup, so ``search_hybrid`` degrades
    to the vector search and ``search_text`` raises ``NotImplementedError``.
    """

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        dimension: int,
        embedder: EmbeddingProvider,
        rrf_k: int = 60,
        timeout: float = 30.0,
        consistency_level: str = "Strong",
    ) -> None:
        super().__init__(embedder=embedder, rrf_k=rrf_k)
        self._uri = uri
        self._collection_name = collection_name
        self._dimension = dimension
        # "Strong" makes writes visible to search immediately (Milvus' default
        # "Bounded" can lag by the flush interval, ~1s); tune for throughput.
        self._consistency_level = consistency_level
        self._client = MilvusClient(uri=uri, timeout=timeout)
        self._prepared = False

    async def prepare(self) -> None:
        """Create the collection, HNSW index and load it (idempotent)."""
        if self._prepared:
            return
        if not await asyncio.to_thread(self._client.has_collection, self._collection_name):
            schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field(
                "chunk_id",
                DataType.VARCHAR,
                max_length=_CHUNK_ID_MAX_LEN,
                is_primary=True,
            )
            schema.add_field("doc_id", DataType.VARCHAR, max_length=_DOC_ID_MAX_LEN)
            schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
            schema.add_field("heading_path", DataType.JSON)
            schema.add_field("page", DataType.INT64, nullable=True)
            schema.add_field("metadata", DataType.JSON)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dimension)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            await asyncio.to_thread(
                self._client.create_collection,
                self._collection_name,
                schema=schema,
                index_params=index_params,
            )
        await asyncio.to_thread(self._client.load_collection, self._collection_name)
        self._prepared = True

    async def _upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        data = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "page": chunk.page,
                "metadata": chunk.metadata,
                "vector": list(vector),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await asyncio.to_thread(self._client.upsert, self._collection_name, data)

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        expr = self._build_expr(filters)
        results = await asyncio.to_thread(
            self._client.search,
            self._collection_name,
            data=[list(embedding)],
            limit=top_k,
            filter=expr,
            output_fields=["chunk_id", "doc_id", "text", "heading_path", "page", "metadata"],
            consistency_level=self._consistency_level,
        )
        hits: list[SearchHit] = []
        for row in results[0]:
            entity = dict(row.get("entity") or {})
            hits.append(
                SearchHit(
                    chunk_id=str(entity.get("chunk_id") or row.get("id")),
                    score=float(row.get("distance") or 0.0),
                    chunk=chunk_from_entity(entity) if entity.get("chunk_id") else None,
                )
            )
        return hits

    async def search_text(
        self,
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError(
            "MilvusVectorStore has no BM25 support; use an ElasticsearchStore "
            "or a Milvus full-text-search setup for text retrieval"
        )

    async def search_hybrid(
        self,
        embedding: Sequence[float],
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        # No BM25 index: the hybrid call degrades to pure vector search.
        return await self.search(embedding, top_k, filters)

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(
            self._client.delete,
            self._collection_name,
            filter=f'doc_id == "{doc_id}"',
        )

    async def close(self) -> None:
        await asyncio.to_thread(self._client.close)

    def _build_expr(self, filters: Filter | None) -> str:
        """Translate filters into a Milvus boolean expression."""
        if filters is None:
            return ""
        clauses: list[str] = []
        if filters.doc_id:
            clauses.append(f'doc_id == "{filters.doc_id}"')
        for key, value in filters.metadata.items():
            if isinstance(value, str):
                clauses.append(f'metadata["{key}"] == "{value}"')
            else:
                clauses.append(f'metadata["{key}"] == {value}')
        return " and ".join(clauses)
