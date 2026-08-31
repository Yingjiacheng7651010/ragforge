"""Token-bucket rate limiting for API endpoints."""

import asyncio
import time


class TokenBucketLimiter:
    """A thread-safe token bucket used as a FastAPI dependency."""

    def __init__(self, capacity: float = 10.0, refill_rate: float = 1.0) -> None:
        if capacity < 1 or refill_rate <= 0:
            raise ValueError("capacity >= 1 and refill_rate > 0 required")
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Consume one token; False when the bucket is empty."""
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._updated) * self._refill_rate,
            )
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False
