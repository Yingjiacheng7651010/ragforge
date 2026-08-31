"""Local BGE embeddings via sentence-transformers (requires the ``local`` extra)."""

import asyncio
import importlib
from typing import Any

from ragforge.core.embeddings import EmbeddingCache, EmbeddingProvider
from ragforge.core.errors import RAGForgeError

_QUERY_PREFIX = "query:"
_DOC_PREFIX = "passage:"


class BGEEmbedding(EmbeddingProvider):
    """BGE models (e.g. ``BAAI/bge-m3``) running locally via sentence-transformers.

    Query and document prefixes follow the BGE training convention; the
    vector dimension is read from the loaded model. This provider needs
    ``sentence-transformers`` (install with ``uv sync --extra local``); the
    model runs in a worker thread so the event loop stays responsive.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self._model_name = model_name
        self._device = device
        self._model: Any = self._load_model()
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_key_prefix(self) -> str:
        return f"bge:{self._model_name}"

    def _load_model(self) -> Any:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as err:
            raise RAGForgeError(
                "BGEEmbedding requires the 'local' extra: run `uv sync --extra local`",
                code="E_EMBEDDING_DEPS_MISSING",
            ) from err
        return module.SentenceTransformer(self._model_name, device=self._device)

    def _prepare_docs(self, texts: list[str]) -> list[str]:
        return [f"{_DOC_PREFIX}{text}" for text in texts]

    def _prepare_query(self, text: str) -> str:
        return f"{_QUERY_PREFIX}{text}"

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
        )
        return [list(embedding) for embedding in embeddings]
