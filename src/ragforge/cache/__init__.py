"""Redis-backed answer caching: exact and semantic."""

from ragforge.cache.base import CachedAnswer, HitType, cosine, is_sensitive
from ragforge.cache.service import CacheService

__all__ = ["CacheService", "CachedAnswer", "HitType", "cosine", "is_sensitive"]
