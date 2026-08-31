"""OpenAI-compatible embeddings provider (any endpoint exposing ``/embeddings``)."""

import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from ragforge.core.embeddings import EmbeddingCache, EmbeddingProvider
from ragforge.core.errors import RAGForgeError


class OpenAIEmbedding(EmbeddingProvider):
    """Embeddings over an OpenAI-compatible endpoint.

    ``dimensions`` is the expected vector width (e.g. from
    ``Settings.embedding_dim``); every returned vector is validated against
    it, so a misconfigured model fails fast with ``E_EMBEDDING_DIM_MISMATCH``.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | str,
        dimensions: int,
        base_url: str | None = None,
        timeout: float = 60.0,
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self._model = model
        self._dimensions = dimensions
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cache_key_prefix(self) -> str:
        return f"openai:{self._model}"

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except openai.APIError as err:
            raise RAGForgeError(
                f"OpenAI embeddings API error: {err}",
                code="E_EMBEDDING_API",
            ) from err
        # The SDK may return data out of order; sort by index to keep input order.
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
