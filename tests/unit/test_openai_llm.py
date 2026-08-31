"""Unit tests for OpenAILLM against a mocked SDK client (no real API calls)."""

from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from ragforge.core.llm import LLMResult, Message
from ragforge.core.llm.base import LLMConnectionError, LLMStatusError, LLMTimeoutError
from ragforge.providers import OpenAILLM

REQUEST = httpx2.Request("POST", "http://127.0.0.1:1")


def make_llm(**overrides: object) -> tuple[OpenAILLM, MagicMock]:
    llm = OpenAILLM(model="test-model", api_key="sk-test", **overrides)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    llm._client = client  # type: ignore[attr-defined]
    return llm, client


def completion_response(
    text: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> ChatCompletion:
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = ChatCompletionMessage(role="assistant", content=text)
    choice = Choice(index=0, message=message, finish_reason="stop")
    return ChatCompletion(
        id="c1",
        model="test-model",
        choices=[choice],
        usage=usage,
        created=1,
        object="chat.completion",
    )


def chunk_response(content: str | None) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=content, role="assistant")
    choice = ChunkChoice(index=0, delta=delta, finish_reason=None)
    return ChatCompletionChunk(
        id="c2",
        model="test-model",
        choices=[choice],
        created=1,
        object="chat.completion.chunk",
    )


async def test_complete_returns_normalized_result() -> None:
    llm, client = make_llm()
    client.chat.completions.create.return_value = completion_response("hello", 10, 5)

    result = await llm.complete([Message(role="user", content="hi")], max_tokens=16)

    assert result == LLMResult(text="hello", prompt_tokens=10, completion_tokens=5, cost=0.0)
    assert result.latency_ms >= 0.0
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 16


async def test_complete_estimates_cost_from_pricing() -> None:
    llm, client = make_llm(pricing={"test-model": (1.0, 2.0)})
    client.chat.completions.create.return_value = completion_response("hello", 10, 5)

    result = await llm.complete([Message(role="user", content="hi")])

    # (10 input * 1.0 + 5 output * 2.0) USD per 1M tokens
    assert result.cost == pytest.approx(20.0 / 1_000_000)


async def test_complete_handles_missing_usage() -> None:
    llm, client = make_llm()
    # usage=None exercises the fallback path
    no_usage = completion_response("hello").model_copy(update={"usage": None})
    client.chat.completions.create.return_value = no_usage

    result = await llm.complete([Message(role="user", content="hi")])

    assert result.text == "hello"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


async def test_complete_maps_timeout_to_retryable_error() -> None:
    llm, client = make_llm()
    client.chat.completions.create.side_effect = APITimeoutError(request=REQUEST)

    with pytest.raises(LLMTimeoutError) as exc_info:
        await llm.complete([Message(role="user", content="hi")])

    assert exc_info.value.retryable is True


async def test_complete_maps_5xx_to_retryable_status_error() -> None:
    llm, client = make_llm()
    client.chat.completions.create.side_effect = _status_error(500)

    with pytest.raises(LLMStatusError) as exc_info:
        await llm.complete([Message(role="user", content="hi")])

    assert exc_info.value.status_code == 500
    assert exc_info.value.retryable is True


async def test_complete_maps_4xx_to_non_retryable_error() -> None:
    llm, client = make_llm()
    client.chat.completions.create.side_effect = _status_error(400)

    with pytest.raises(LLMStatusError) as exc_info:
        await llm.complete([Message(role="user", content="hi")])

    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


async def test_complete_maps_connection_error() -> None:
    llm, client = make_llm()
    client.chat.completions.create.side_effect = APIConnectionError(request=REQUEST)

    with pytest.raises(LLMConnectionError) as exc_info:
        await llm.complete([Message(role="user", content="hi")])

    assert exc_info.value.retryable is True


async def test_stream_yields_text_deltas() -> None:
    llm, client = make_llm()

    async def fake_stream() -> object:
        yield chunk_response("Hel")
        yield chunk_response(None)
        yield chunk_response("lo")

    client.chat.completions.create.return_value = fake_stream()

    stream = await llm.stream([Message(role="user", content="hi")])

    assert [chunk async for chunk in stream] == ["Hel", "lo"]
    assert client.chat.completions.create.call_args.kwargs["stream"] is True


async def test_stream_maps_timeout_error() -> None:
    llm, client = make_llm()
    client.chat.completions.create.side_effect = APITimeoutError(request=REQUEST)

    with pytest.raises(LLMTimeoutError):
        await llm.stream([Message(role="user", content="hi")])


async def test_complete_structured_injects_schema_and_parses() -> None:
    llm, client = make_llm()
    client.chat.completions.create.return_value = completion_response('{"ok": 1}')

    result = await llm.complete_structured([Message(role="user", content="go")], {"type": "object"})

    assert result == {"ok": 1}
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "system"
    assert "JSON Schema" in messages[-1]["content"]


def _status_error(status_code: int) -> APIStatusError:
    response = httpx2.Response(status_code, request=REQUEST)
    return APIStatusError("boom", response=response, body=None)
