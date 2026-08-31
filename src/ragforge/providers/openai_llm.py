"""OpenAI-compatible chat-completions provider.

Works against any endpoint speaking the OpenAI protocol, including
DeepSeek (``https://api.deepseek.com``) and Doubao/Volcano Ark
(``https://ark.cn-beijing.volces.com/api/v3``) via ``base_url``.
"""

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from pydantic import SecretStr

from ragforge.core.llm.base import (
    BaseLLM,
    LLMConnectionError,
    LLMError,
    LLMResult,
    LLMStatusError,
    LLMTimeoutError,
    Message,
)

#: Optional pricing table: model -> (USD per 1M input tokens, USD per 1M output tokens).
Pricing = dict[str, tuple[float, float]]


def _to_openai_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    """Convert ragforge messages to the SDK's typed message union."""
    return [cast(ChatCompletionMessageParam, m.to_dict()) for m in messages]


def _translate_error(err: openai.APIError) -> LLMError:
    """Map SDK errors onto the ragforge LLM error taxonomy."""
    if isinstance(err, openai.APITimeoutError):
        return LLMTimeoutError(str(err))
    if isinstance(err, openai.APIConnectionError):
        return LLMConnectionError(str(err))
    if isinstance(err, openai.APIStatusError):
        return LLMStatusError(
            str(err),
            status_code=err.status_code,
            retryable=err.status_code >= 500,
        )
    return LLMError(str(err))


class OpenAILLM(BaseLLM):
    """Chat-completions LLM over any OpenAI-compatible endpoint.

    ``api_key`` is never hardcoded; pass it explicitly (e.g. from
    ``Settings.llm_api_key``). ``pricing`` is optional and only used to
    estimate ``LLMResult.cost``; without it cost stays 0.0.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr | str,
        base_url: str | None = None,
        pricing: Pricing | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._pricing = pricing or {}
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        self._client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        started = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.APIError as err:
            raise _translate_error(err) from err

        if not response.choices:
            raise LLMError("LLM returned no choices", code="E_LLM_EMPTY_RESPONSE")

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=self._estimate_cost(prompt_tokens, completion_tokens),
            latency_ms=(time.monotonic() - started) * 1000,
        )

    async def stream(self, messages: Sequence[Message], **kwargs: object) -> AsyncIterator[str]:
        extra: dict[str, Any] = cast(dict[str, Any], kwargs)
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(messages),
                stream=True,
                **extra,
            )
        except openai.APIError as err:
            raise _translate_error(err) from err
        return self._iter_text(stream)

    async def _iter_text(self, stream: AsyncStream[ChatCompletionChunk]) -> AsyncIterator[str]:
        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prices = self._pricing.get(self._model)
        if prices is None:
            return 0.0
        input_price, output_price = prices
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
