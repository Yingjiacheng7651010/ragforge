"""Application settings for ragforge, loaded from env vars and ``.env``.

Every setting is read from an environment variable prefixed with
``RAGFORGE_`` (e.g. ``RAGFORGE_LLM_MODEL``) or from the ``.env`` file
in the working directory. Empty values are treated as unset.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for ragforge."""

    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str
    llm_model: str
    llm_api_key: SecretStr
    llm_fallback_chain: list[str] = Field(default_factory=list)

    # --- Embedding / reranking ---
    embedding_model: str
    embedding_dim: int
    reranker_model: str

    # --- Retrieval ---
    retrieval_top_k: int = 50
    rerank_top_n: int = 8
    rrf_k: int = 60

    # --- Semantic cache ---
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92

    # --- Observability ---
    otel_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    # Required fields without defaults are populated from env vars / .env
    # at runtime; mypy cannot see pydantic-settings magic here.
    return Settings()  # type: ignore[call-arg]


def reset_settings() -> None:
    """Clear the cached singleton (mainly for tests)."""
    get_settings.cache_clear()
