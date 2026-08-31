"""Two-stage retrieval funnel: recall -> (RRF inside hybrid) -> rerank -> top_n."""

from ragforge.core.vector_store import Filter, SearchHit
from ragforge.retrieval.rerank import Reranker
from ragforge.retrieval.retrievers import Retriever


class RetrievalPipeline(Retriever):
    """Recall with the wrapped retriever, then rerank and truncate to ``rerank_n``.

    The funnel matches production RAG: recall a generous ``recall_k``
    (default 50; a :class:`HybridRetriever` fuses dense+sparse with RRF at
    this stage), then rerank with a cross-encoder and keep ``rerank_n``
    (default 8). With no reranker configured the pipeline degrades to
    recall + truncation.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker | None = None,
        recall_k: int = 50,
        rerank_n: int = 8,
    ) -> None:
        if recall_k < 1 or rerank_n < 1:
            raise ValueError("recall_k and rerank_n must be >= 1")
        self._retriever = retriever
        self._reranker = reranker
        self._recall_k = recall_k
        self._rerank_n = rerank_n

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Filter | None = None,
    ) -> list[SearchHit]:
        """``top_k`` is ignored; the funnel uses the configured recall_k/rerank_n."""
        hits = await self._retriever.retrieve(query, self._recall_k, filters)
        if self._reranker is not None and hits:
            hits = await self._reranker.rerank(query, hits)
        return hits[: self._rerank_n]
