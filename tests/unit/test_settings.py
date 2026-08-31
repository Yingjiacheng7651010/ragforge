"""Unit tests for ``ragforge.config``: env/.env loading and secrets masking."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ragforge.config import Settings, get_settings, reset_settings

#: Values for required fields that have no default; applied via env or kwargs.
REQUIRED = {
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
    "llm_api_key": "sk-test",
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "reranker_model": "bge-reranker-v2-m3",
}

REQUIRED_ENV = {f"RAGFORGE_{key.upper()}": str(value) for key, value in REQUIRED.items()}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove any RAGFORGE_* variables and clear the cached singleton."""
    for key in list(os.environ):
        if key.startswith("RAGFORGE_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings()
    yield
    reset_settings()


def set_env(monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))


def test_defaults_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED_ENV)

    settings = Settings()

    assert settings.retrieval_top_k == 50
    assert settings.rerank_top_n == 8
    assert settings.rrf_k == 60
    assert settings.semantic_cache_enabled is True
    assert settings.semantic_cache_threshold == 0.92
    assert settings.otel_endpoint is None
    assert settings.llm_fallback_chain == []


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(
        monkeypatch,
        **REQUIRED_ENV,
        RAGFORGE_RETRIEVAL_TOP_K="10",
        RAGFORGE_RERANK_TOP_N="5",
        RAGFORGE_RRF_K="3",
        RAGFORGE_SEMANTIC_CACHE_ENABLED="false",
        RAGFORGE_SEMANTIC_CACHE_THRESHOLD="0.80",
        RAGFORGE_OTEL_ENDPOINT="http://otel:4317",
        RAGFORGE_LLM_FALLBACK_CHAIN='["anthropic", "deepseek"]',
    )

    settings = Settings()

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.retrieval_top_k == 10
    assert settings.rerank_top_n == 5
    assert settings.rrf_k == 3
    assert settings.semantic_cache_enabled is False
    assert settings.semantic_cache_threshold == 0.80
    assert settings.otel_endpoint == "http://otel:4317"
    assert settings.llm_fallback_chain == ["anthropic", "deepseek"]


def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "RAGFORGE_LLM_PROVIDER=openai",
                "RAGFORGE_LLM_MODEL=gpt-4o-from-file",
                "RAGFORGE_LLM_API_KEY=sk-from-file",
                "RAGFORGE_EMBEDDING_MODEL=file-embed",
                "RAGFORGE_EMBEDDING_DIM=768",
                "RAGFORGE_RERANKER_MODEL=file-rerank",
                "RAGFORGE_RETRIEVAL_TOP_K=12",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.llm_model == "gpt-4o-from-file"
    assert settings.llm_api_key.get_secret_value() == "sk-from-file"
    assert settings.retrieval_top_k == 12


def test_empty_env_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED_ENV, RAGFORGE_OTEL_ENDPOINT="")

    settings = Settings()

    assert settings.otel_endpoint is None


def test_secret_str_is_masked() -> None:
    settings = Settings(**REQUIRED)

    assert settings.llm_api_key.get_secret_value() == "sk-test"
    assert str(settings.llm_api_key) != "sk-test"
    assert repr(settings.llm_api_key) != "sk-test"
    assert "sk-test" not in repr(settings)
    assert "sk-test" not in str(settings.model_dump())


def test_get_settings_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED_ENV)

    first = get_settings()
    second = get_settings()

    assert first is second

    reset_settings()
    assert get_settings() is not first
