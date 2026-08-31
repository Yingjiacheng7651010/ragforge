"""Optional embedding caches: in-process (with optional file persistence) and Redis."""

import json
from pathlib import Path
from typing import Protocol, cast

import redis.asyncio as redis_async


class EmbeddingCache(Protocol):
    """Async key/value store for document vectors keyed by text hash."""

    async def get(self, key: str) -> list[float] | None: ...

    async def set(self, key: str, value: list[float]) -> None: ...


class LocalEmbeddingCache:
    """In-process cache, optionally persisted to a JSON file on every write."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._path = Path(persist_path) if persist_path is not None else None
        self._store: dict[str, list[float]] = {}
        if self._path is not None and self._path.exists():
            self._store = json.loads(self._path.read_text(encoding="utf-8"))

    async def get(self, key: str) -> list[float] | None:
        return self._store.get(key)

    async def set(self, key: str, value: list[float]) -> None:
        self._store[key] = list(value)
        if self._path is not None:
            self._path.write_text(json.dumps(self._store), encoding="utf-8")


class RedisEmbeddingCache:
    """Redis-backed cache; vectors are stored as JSON strings."""

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "ragforge:embed:",
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis = redis_async.from_url(redis_url)
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    async def get(self, key: str) -> list[float] | None:
        raw = await self._redis.get(self._key_prefix + key)
        if raw is None:
            return None
        return cast(list[float], json.loads(raw))

    async def set(self, key: str, value: list[float]) -> None:
        payload = json.dumps(value)
        full_key = self._key_prefix + key
        if self._ttl_seconds is None:
            await self._redis.set(full_key, payload)
        else:
            await self._redis.set(full_key, payload, ex=self._ttl_seconds)
