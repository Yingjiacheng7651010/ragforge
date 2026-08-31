"""Unit tests for the embedding layer: batching, dimension checks, prefixes, caches."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx2
import openai
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from ragforge.core.embeddings import LocalEmbeddingCache, RedisEmbeddingCache
from ragforge.core.errors import RAGForgeError
from ragforge.providers import BGEEmbedding, OpenAIEmbedding
from tests.unit.fakes import FakeEmbedding


def texts(n: int) -> list[str]:
    return [f"text-{i}" for i in range(n)]


# --- base batching / validation ---


async def test_embed_empty_returns_no_vectors() -> None:
    provider = FakeEmbedding()

    assert await provider.embed([]) == []
    assert provider.batches == []


async def test_embed_splits_into_batches() -> None:
    provider = FakeEmbedding()

    vectors = await provider.embed(texts(10), batch_size=4)

    assert len(vectors) == 10
    assert all(len(vector) == 3 for vector in vectors)
    assert [len(batch) for batch in provider.batches] == [4, 4, 2]


async def test_embed_query_returns_single_vector() -> None:
    provider = FakeEmbedding()

    vector = await provider.embed_query("hello")

    assert vector == [0.1, 0.2, 0.3]
    assert provider.batches == [["hello"]]


async def test_dimension_mismatch_raises_on_embed() -> None:
    provider = FakeEmbedding(dimensions=4)  # fake returns 3 dims

    with pytest.raises(RAGForgeError) as exc_info:
        await provider.embed(texts(2))

    assert exc_info.value.code == "E_EMBEDDING_DIM_MISMATCH"


async def test_dimension_mismatch_raises_on_embed_query() -> None:
    provider = FakeEmbedding(dimensions=4)

    with pytest.raises(RAGForgeError) as exc_info:
        await provider.embed_query("hello")

    assert exc_info.value.code == "E_EMBEDDING_DIM_MISMATCH"


# --- local cache ---


async def test_local_cache_skips_second_model_call() -> None:
    cache = LocalEmbeddingCache()
    provider = FakeEmbedding(cache=cache)

    first = await provider.embed(texts(2))
    second = await provider.embed(texts(2))

    assert first == second
    assert len(provider.batches) == 1  # model called exactly once


async def test_cache_keys_are_namespaced_by_provider() -> None:
    cache = LocalEmbeddingCache()
    provider = FakeEmbedding(cache=cache)

    await provider.embed(["hello"])

    key = next(iter(cache._store))  # type: ignore[attr-defined]
    assert key.startswith("FakeEmbedding:")


async def test_local_cache_persists_to_file(tmp_path: Path) -> None:
    persist = tmp_path / "embeddings.json"
    provider = FakeEmbedding(cache=LocalEmbeddingCache(persist_path=persist))
    await provider.embed(["hello"])

    reloaded = FakeEmbedding(cache=LocalEmbeddingCache(persist_path=persist))
    await reloaded.embed(["hello"])

    assert len(reloaded.batches) == 0  # served from disk


# --- redis cache ---


async def test_redis_cache_roundtrip() -> None:
    cache = RedisEmbeddingCache("redis://localhost:6379/0", key_prefix="ragforge:embed:")
    store: dict[str, str] = {}
    fake_redis = AsyncMock()
    fake_redis.get.side_effect = lambda key: store.get(key)
    fake_redis.set.side_effect = lambda key, value, **kwargs: store.__setitem__(key, value)
    cache._redis = fake_redis  # type: ignore[attr-defined]

    assert await cache.get("abc") is None

    await cache.set("abc", [1.0, 2.0])

    assert await cache.get("abc") == [1.0, 2.0]
    fake_redis.set.assert_called_once_with("ragforge:embed:abc", json.dumps([1.0, 2.0]))


# --- OpenAI embedding ---


def make_openai_embedding(
    *,
    dimensions: int = 3,
    **overrides: object,
) -> tuple[OpenAIEmbedding, MagicMock]:
    provider = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="sk-test",
        dimensions=dimensions,
        **overrides,
    )
    client = MagicMock()
    client.embeddings.create = AsyncMock()
    provider._client = client  # type: ignore[attr-defined]
    return provider, client


def embedding_response(items: list[tuple[int, list[float]]]) -> CreateEmbeddingResponse:
    usage = Usage(prompt_tokens=2, total_tokens=2)
    data = [Embedding(index=index, embedding=vector, object="embedding") for index, vector in items]
    return CreateEmbeddingResponse(data=data, model="m", object="list", usage=usage)


async def test_openai_embedding_orders_by_index() -> None:
    provider, client = make_openai_embedding()
    client.embeddings.create.return_value = embedding_response(
        [(2, [0.2, 0.2, 0.2]), (0, [0.0, 0.0, 0.0]), (1, [0.1, 0.1, 0.1])]
    )

    vectors = await provider.embed(["a", "b", "c"])

    assert vectors == [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
    assert client.embeddings.create.call_args.kwargs["input"] == ["a", "b", "c"]
    assert client.embeddings.create.call_args.kwargs["model"] == "text-embedding-3-small"


async def test_openai_embedding_dimension_mismatch() -> None:
    provider, client = make_openai_embedding(dimensions=4)
    client.embeddings.create.return_value = embedding_response([(0, [0.1, 0.2, 0.3])])

    with pytest.raises(RAGForgeError) as exc_info:
        await provider.embed(["a"])

    assert exc_info.value.code == "E_EMBEDDING_DIM_MISMATCH"


async def test_openai_embedding_query_sends_single_text() -> None:
    provider, client = make_openai_embedding()
    client.embeddings.create.return_value = embedding_response([(0, [0.5, 0.5, 0.5])])

    vector = await provider.embed_query("question?")

    assert vector == [0.5, 0.5, 0.5]
    assert client.embeddings.create.call_args.kwargs["input"] == ["question?"]


async def test_openai_embedding_maps_api_errors() -> None:
    provider, client = make_openai_embedding()
    request = httpx2.Request("POST", "http://127.0.0.1:1")
    client.embeddings.create.side_effect = openai.APITimeoutError(request=request)

    with pytest.raises(RAGForgeError) as exc_info:
        await provider.embed(["a"])

    assert exc_info.value.code == "E_EMBEDDING_API"


# --- BGE embedding (fake local model) ---


class FakeSentenceTransformer:
    """Minimal stand-in for sentence_transformers.SentenceTransformer."""

    def __init__(self, dim: int = 2) -> None:
        self.dim = dim
        self.encode_calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], *, normalize_embeddings: bool = True) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


def make_bge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dim: int = 2,
) -> tuple[BGEEmbedding, FakeSentenceTransformer]:
    model = FakeSentenceTransformer(dim)
    monkeypatch.setattr(BGEEmbedding, "_load_model", lambda self: model)
    provider = BGEEmbedding(model_name="fake-bge")
    return provider, model


async def test_bge_doc_prefix_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = make_bge(monkeypatch)

    await provider.embed(["hello", "world"])

    assert model.encode_calls[-1] == ["passage:hello", "passage:world"]


async def test_bge_query_prefix_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = make_bge(monkeypatch)

    await provider.embed_query("hello")

    assert model.encode_calls[-1] == ["query:hello"]


async def test_bge_batches_with_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, model = make_bge(monkeypatch)

    await provider.embed(texts(5), batch_size=2)

    assert model.encode_calls == [
        ["passage:text-0", "passage:text-1"],
        ["passage:text-2", "passage:text-3"],
        ["passage:text-4"],
    ]


async def test_bge_dimensions_read_from_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = make_bge(monkeypatch, dim=7)

    assert provider.dimensions == 7


async def test_bge_missing_deps_raise_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def import_fails(name: str) -> None:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr("ragforge.providers.bge_embedding.importlib.import_module", import_fails)

    with pytest.raises(RAGForgeError) as exc_info:
        BGEEmbedding(model_name="fake")

    assert exc_info.value.code == "E_EMBEDDING_DEPS_MISSING"
