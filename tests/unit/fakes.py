"""Test doubles shared across unit tests (never imported by src)."""

from collections import deque
from collections.abc import AsyncIterator, Sequence

from ragforge.core.embeddings import EmbeddingCache, EmbeddingProvider
from ragforge.core.llm import BaseLLM, LLMResult, Message
from ragforge.core.vector_store import Filter, SearchHit, VectorStore
from ragforge.ingestion import Chunk
from ragforge.retrieval import Retriever


class FakeLLM(BaseLLM):
    """Scripted LLM double: each call pops the next queued item.

    Queues hold either values (``LLMResult`` for complete, ``list[str]`` for
    stream) or exceptions that should be raised. When a queue is empty a
    default reply is returned, so scripts can rely on ordering alone.
    """

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._completions: deque[LLMResult | Exception] = deque()
        self._streams: deque[list[str] | Exception] = deque()
        self.complete_calls: list[tuple[Sequence[Message], float, int | None]] = []
        self.stream_calls: list[Sequence[Message]] = []

    def enqueue_completion(self, *items: LLMResult | Exception) -> None:
        self._completions.extend(items)

    def enqueue_stream(self, *items: list[str] | Exception) -> None:
        self._streams.extend(items)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.complete_calls.append((messages, temperature, max_tokens))
        item = self._completions.popleft() if self._completions else LLMResult(text="fake-reply")
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, messages: Sequence[Message], **kwargs: object) -> AsyncIterator[str]:
        self.stream_calls.append(messages)
        item = self._streams.popleft() if self._streams else ["fake-chunk"]
        if isinstance(item, Exception):
            raise item

        async def gen() -> AsyncIterator[str]:
            for chunk in item:
                yield chunk

        return gen()


class FakeEmbedding(EmbeddingProvider):
    """Scripted embedding provider: fixed-width vectors, records every batch."""

    def __init__(self, dimensions: int = 3, *, cache: EmbeddingCache | None = None) -> None:
        super().__init__(cache=cache)
        self._dimensions = dimensions
        self.batches: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeRetriever(Retriever):
    """Scripted retriever: pops the next preset hit list per call, records every call."""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._queue: deque[list[SearchHit]] = deque()
        if hits is not None:
            self._queue.append(hits)
        self.calls: list[tuple[str, int, Filter | None]] = []

    def enqueue_hits(self, *hit_lists: list[SearchHit]) -> None:
        self._queue.extend(hit_lists)

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        self.calls.append((query, top_k, filters))
        return list(self._queue.popleft()) if self._queue else []


class InMemoryVectorStore(VectorStore):
    """Minimal in-memory store for e2e tests (no external services)."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        super().__init__(embedder=embedder)
        self._chunks: dict[str, Chunk] = {}

    async def _upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        hits = [
            SearchHit(chunk_id=chunk.chunk_id, score=1.0, chunk=chunk)
            for chunk in self._chunks.values()
        ]
        if filters is not None and filters.doc_id:
            hits = [
                hit
                for hit in hits
                if hit.chunk is not None and hit.chunk.doc_id == filters.doc_id
            ]
        return hits[:top_k]

    async def search_text(
        self,
        text: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        hits = [
            SearchHit(chunk_id=chunk.chunk_id, score=1.0, chunk=chunk)
            for chunk in self._chunks.values()
            if text in chunk.text
        ]
        return hits[:top_k]

    async def delete(self, doc_id: str) -> None:
        self._chunks = {key: chunk for key, chunk in self._chunks.items() if chunk.doc_id != doc_id}

    async def close(self) -> None:
        return None
