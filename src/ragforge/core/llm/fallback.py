"""Fallback orchestration: provider failover, retry with backoff, circuit breaker."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TypeVar

import structlog

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm.base import BaseLLM, LLMError, LLMResult, Message

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")


def _exponential_backoff(attempt: int, base_delay: float) -> float:
    """Compute the retry delay: ``base_delay * 2 ** attempt`` (attempt 0-based)."""
    return base_delay * (1 << attempt)


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures, for ``open_seconds``."""

    def __init__(self, failure_threshold: int = 5, open_seconds: float = 60.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return time.monotonic() - self._opened_at < self.open_seconds

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class FallbackLLM(BaseLLM):
    """Try providers in order, retrying idempotent errors with exponential backoff.

    Only retryable errors (timeout / connection / 5xx) trigger failover;
    deterministic client errors propagate immediately. When every provider
    fails, the breaker counts a failure and ``E_LLM_DOWN`` is raised; after
    ``failure_threshold`` consecutive overall failures the breaker opens for
    ``open_seconds`` and requests fail fast without touching any provider.

    Note: for ``stream`` the failover happens when the stream is *created*;
    errors raised mid-stream are propagated as-is.
    """

    def __init__(
        self,
        providers: Sequence[BaseLLM],
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        failure_threshold: int = 5,
        open_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = list(providers)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self.breaker = CircuitBreaker(failure_threshold, open_seconds)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return await self._run(
            lambda provider: provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    async def stream(self, messages: Sequence[Message], **kwargs: object) -> AsyncIterator[str]:
        return await self._run(lambda provider: provider.stream(messages, **kwargs))

    async def _run(self, call: Callable[[BaseLLM], Awaitable[_T]]) -> _T:
        if self.breaker.is_open:
            raise RAGForgeError(
                "LLM circuit breaker is open; skipping all providers",
                code="E_LLM_DOWN",
            )
        last_error: LLMError | None = None
        for provider in self._providers:
            try:
                result = await self._call_with_retry(provider, call)
            except LLMError as err:
                if not err.retryable:
                    raise
                last_error = err
                logger.warning(
                    "llm provider failed; trying next",
                    provider=type(provider).__name__,
                    error=err.message,
                    code=err.code,
                )
                continue
            self.breaker.record_success()
            return result
        self.breaker.record_failure()
        raise RAGForgeError(
            f"all LLM providers failed: {last_error}",
            code="E_LLM_DOWN",
        ) from last_error

    async def _call_with_retry(
        self,
        provider: BaseLLM,
        call: Callable[[BaseLLM], Awaitable[_T]],
    ) -> _T:
        for attempt in range(self._max_retries + 1):
            try:
                return await call(provider)
            except LLMError as err:
                if not err.retryable or attempt >= self._max_retries:
                    raise
                await asyncio.sleep(_exponential_backoff(attempt, self._base_delay))
        raise AssertionError("unreachable")
