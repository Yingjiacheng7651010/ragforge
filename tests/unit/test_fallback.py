"""Unit tests for FallbackLLM: failover, exponential-backoff retries, circuit breaker."""

import time

import pytest

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import LLMResult, Message
from ragforge.core.llm.base import LLMConnectionError, LLMStatusError, LLMTimeoutError
from ragforge.core.llm.fallback import FallbackLLM, _exponential_backoff
from tests.unit.fakes import FakeLLM


def reply(text: str) -> LLMResult:
    return LLMResult(text=text)


def chat() -> list[Message]:
    return [Message(role="user", content="hi")]


def test_exponential_backoff_values() -> None:
    assert _exponential_backoff(0, 1.0) == 1.0
    assert _exponential_backoff(1, 1.0) == 2.0
    assert _exponential_backoff(2, 1.0) == 4.0
    assert _exponential_backoff(2, 0.5) == 2.0


def test_empty_providers_rejected() -> None:
    with pytest.raises(ValueError):
        FallbackLLM([])


async def test_single_provider_success() -> None:
    provider = FakeLLM("ok")
    provider.enqueue_completion(reply("hi there"))
    fallback = FallbackLLM([provider], max_retries=0, base_delay=0.0)

    result = await fallback.complete(chat())

    assert result.text == "hi there"


async def test_fallback_switches_to_second_provider() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMConnectionError("connection refused"))
    second = FakeLLM("second")
    second.enqueue_completion(reply("from second"))
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0)

    result = await fallback.complete(chat())

    assert result.text == "from second"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 1


async def test_all_providers_fail_raises_e_llm_down() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMTimeoutError("timeout"))
    second = FakeLLM("second")
    second.enqueue_completion(LLMConnectionError("refused"))
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0)

    with pytest.raises(RAGForgeError) as exc_info:
        await fallback.complete(chat())

    assert exc_info.value.code == "E_LLM_DOWN"
    assert len(first.complete_calls) == 1
    assert len(second.complete_calls) == 1


async def test_non_retryable_error_propagates_without_failover() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMStatusError("bad request", status_code=400, retryable=False))
    second = FakeLLM("second")
    second.enqueue_completion(reply("should never be reached"))
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0)

    with pytest.raises(LLMStatusError) as exc_info:
        await fallback.complete(chat())

    assert exc_info.value.status_code == 400
    assert len(second.complete_calls) == 0


async def test_retry_succeeds_after_transient_errors() -> None:
    provider = FakeLLM("flaky")
    provider.enqueue_completion(LLMTimeoutError("t1"), LLMConnectionError("c2"), reply("ok"))
    fallback = FallbackLLM([provider], max_retries=2, base_delay=0.0)

    result = await fallback.complete(chat())

    assert result.text == "ok"
    assert len(provider.complete_calls) == 3


async def test_retry_uses_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("ragforge.core.llm.fallback.asyncio.sleep", fake_sleep)
    provider = FakeLLM("flaky")
    provider.enqueue_completion(LLMTimeoutError("t1"), LLMTimeoutError("t2"), reply("ok"))
    fallback = FallbackLLM([provider], max_retries=2, base_delay=1.0)

    await fallback.complete(chat())

    assert sleeps == [1.0, 2.0]


async def test_retry_exhausted_then_fails_over() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMTimeoutError("t1"), LLMTimeoutError("t2"))
    second = FakeLLM("second")
    second.enqueue_completion(reply("from second"))
    fallback = FallbackLLM([first, second], max_retries=1, base_delay=0.0)

    result = await fallback.complete(chat())

    assert result.text == "from second"
    assert len(first.complete_calls) == 2  # original + 1 retry


async def test_circuit_breaker_opens_after_consecutive_failures() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(*([LLMConnectionError("down")] * 5))
    second = FakeLLM("second")
    second.enqueue_completion(*([LLMConnectionError("down")] * 5))
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0, failure_threshold=5)

    for _ in range(5):
        with pytest.raises(RAGForgeError) as exc_info:
            await fallback.complete(chat())
        assert exc_info.value.code == "E_LLM_DOWN"

    # The 6th call must fail fast without touching any provider.
    with pytest.raises(RAGForgeError) as exc_info:
        await fallback.complete(chat())
    assert exc_info.value.code == "E_LLM_DOWN"
    assert len(first.complete_calls) == 5
    assert len(second.complete_calls) == 5


async def test_circuit_breaker_recovers_after_open_period() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMConnectionError("down"))
    fallback = FallbackLLM(
        [first],
        max_retries=0,
        base_delay=0.0,
        failure_threshold=1,
        open_seconds=60.0,
    )

    with pytest.raises(RAGForgeError):
        await fallback.complete(chat())
    assert fallback.breaker.is_open

    # Simulate 61s passing: breaker half-opens and providers are tried again.
    fallback.breaker._opened_at = time.monotonic() - 61.0  # type: ignore[attr-defined]
    assert not fallback.breaker.is_open

    first.enqueue_completion(LLMConnectionError("down"))
    with pytest.raises(RAGForgeError):
        await fallback.complete(chat())
    assert len(first.complete_calls) == 2


async def test_success_resets_consecutive_failure_count() -> None:
    flaky = FakeLLM("flaky")
    fallback = FallbackLLM([flaky], max_retries=0, base_delay=0.0, failure_threshold=3)

    flaky.enqueue_completion(LLMConnectionError("down"), LLMConnectionError("down"))
    for _ in range(2):
        with pytest.raises(RAGForgeError):
            await fallback.complete(chat())

    flaky.enqueue_completion(reply("recovered"))
    result = await fallback.complete(chat())
    assert result.text == "recovered"

    # Failures before the success must not count towards the breaker.
    flaky.enqueue_completion(LLMConnectionError("down"), LLMConnectionError("down"))
    for _ in range(2):
        with pytest.raises(RAGForgeError):
            await fallback.complete(chat())
    assert not fallback.breaker.is_open


async def test_fallback_stream_switches_provider() -> None:
    first = FakeLLM("first")
    first.enqueue_stream(LLMConnectionError("stream failed"))
    second = FakeLLM("second")
    second.enqueue_stream(["chunk-1", "chunk-2"])
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0)

    stream = await fallback.stream(chat())

    assert [chunk async for chunk in stream] == ["chunk-1", "chunk-2"]
    assert len(first.stream_calls) == 1
    assert len(second.stream_calls) == 1


async def test_fallback_complete_structured_uses_failover() -> None:
    first = FakeLLM("first")
    first.enqueue_completion(LLMTimeoutError("timeout"))
    second = FakeLLM("second")
    second.enqueue_completion(reply('{"answer": 7}'))
    fallback = FallbackLLM([first, second], max_retries=0, base_delay=0.0)

    result = await fallback.complete_structured(chat(), {"type": "object"})

    assert result == {"answer": 7}
    assert len(second.complete_calls) == 1
