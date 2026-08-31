"""Application services: the assembled dependency container for the API."""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import redis.asyncio as redis_async

from ragforge.cache import CacheService
from ragforge.config import get_settings
from ragforge.core.errors import RAGForgeError
from ragforge.generation import Generator
from ragforge.guardrails import InputGuard, OutputGuard
from ragforge.providers import ElasticsearchStore, OpenAIEmbedding, OpenAILLM
from ragforge.query import QueryUnderstandingService
from ragforge.retrieval import (
    BGEReranker,
    CorrectiveRagRetriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrievalPipeline,
    SelfRagEvaluator,
    SparseRetriever,
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
_INDEX_NAME = os.environ.get("RAGFORGE_INDEX", "chunks")


class DocumentIngestor(Protocol):
    """Submit documents for async ingestion and query their status."""

    def submit(self, doc_id: str, filename: str, content: bytes) -> None: ...

    async def status(self, doc_id: str) -> tuple[str, dict[str, Any]]: ...


class InMemoryIngestor:
    """Test double: records submissions and reports scripted statuses."""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, bytes]] = []
        self._statuses: dict[str, tuple[str, dict[str, Any]]] = {}

    def set_status(self, doc_id: str, state: str, info: dict[str, Any] | None = None) -> None:
        self._statuses[doc_id] = (state, info or {})

    def submit(self, doc_id: str, filename: str, content: bytes) -> None:
        self.submissions.append((doc_id, filename, content))
        self._statuses.setdefault(doc_id, ("PENDING", {}))

    async def status(self, doc_id: str) -> tuple[str, dict[str, Any]]:
        return self._statuses.get(doc_id, ("UNKNOWN", {"message": "unknown doc_id"}))


@dataclass
class AppServices:
    """Everything the API routes need; tests inject fakes, production builds real."""

    input_guard: InputGuard
    output_guard: OutputGuard
    cache: CacheService
    understanding: QueryUnderstandingService
    pipeline: RetrievalPipeline
    generator: Generator
    ingestor: DocumentIngestor
    redis: redis_async.Redis
    reranker: Reranker | None = None
    es_store: ElasticsearchStore | None = None

    async def check_health(self) -> dict[str, str]:
        """Probe this service and its downstream dependencies."""
        checks: dict[str, str] = {"api": "ok"}
        try:
            await self.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "error"
        if self.es_store is not None:
            try:
                await self.es_store.ping()
                checks["elasticsearch"] = "ok"
            except Exception:
                checks["elasticsearch"] = "error"
        return checks


def build_services() -> AppServices:
    """Assemble the production service container from settings/environment."""
    settings = get_settings()
    if not settings.llm_api_key.get_secret_value():
        raise RAGForgeError(
            "RAGFORGE_LLM_API_KEY is required to build the API services",
            code="E_API_CONFIG",
        )
    llm = OpenAILLM(model=settings.llm_model, api_key=settings.llm_api_key)
    embedder = OpenAIEmbedding(
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        api_key=settings.llm_api_key,
    )
    es_store = ElasticsearchStore(
        hosts=ES_URL,
        index_name=_INDEX_NAME,
        dimension=embedder.dimensions,
        embedder=embedder,
    )
    redis = redis_async.from_url(REDIS_URL)

    cache = CacheService(redis=redis, embedder=embedder)
    hybrid = HybridRetriever(
        dense=DenseRetriever(store=es_store, embedder=embedder),
        sparse=SparseRetriever(store=es_store),
    )
    corrective = CorrectiveRagRetriever(
        retriever=hybrid,
        evaluator=SelfRagEvaluator(llm),
    )
    reranker: Reranker | None = None
    if os.environ.get("RAGFORGE_RERANKER", "0") == "1":
        reranker = BGEReranker()
    pipeline = RetrievalPipeline(retriever=corrective, reranker=reranker, recall_k=50, rerank_n=8)

    from ragforge.api.tasks import CeleryIngestor

    return AppServices(
        input_guard=InputGuard(llm),
        output_guard=OutputGuard(llm),
        cache=cache,
        understanding=QueryUnderstandingService(llm),
        pipeline=pipeline,
        generator=Generator(llm=llm),
        ingestor=CeleryIngestor(),
        redis=redis,
        reranker=reranker,
        es_store=es_store,
    )


def new_doc_id() -> str:
    return uuid.uuid4().hex
