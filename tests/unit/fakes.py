"""Test doubles shared across unit tests (never imported by src)."""

from collections import deque
from collections.abc import AsyncIterator, Sequence

from ragforge.core.llm import BaseLLM, LLMResult, Message


class FakeLLM(BaseLLM):
    """Scripted LLM double: each call pops the next queued item.

    Queues hold either values (``LLMResult`` for complete, ``list[str]`` for
    stream) or exceptions that should be raised. When a queue is empty a
    default reply is returned, so scripts can rely on ordering alone.
    """

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self._completions: deque[LLMResult | Exception] = deque()
        self._streams: deque[list[str] | Exception] = deque()
        self.complete_calls: list[tuple[Sequence[Message], float, int | None]] = []
        self.stream_calls: list[Sequence[Message]] = []

    def enqueue_completion(self, *items: LLMResult | Exception) -> None:
        self._completions.extend(items)

    def enqueue_stream(self, *items: list[str] | Exception) -> None:
        self._streams.extend(items)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.complete_calls.append((messages, temperature, max_tokens))
        item = self._completions.popleft() if self._completions else LLMResult(text="fake-reply")
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, messages: Sequence[Message], **kwargs: object) -> AsyncIterator[str]:
        self.stream_calls.append(messages)
        item = self._streams.popleft() if self._streams else ["fake-chunk"]
        if isinstance(item, Exception):
            raise item

        async def gen() -> AsyncIterator[str]:
            for chunk in item:
                yield chunk

        return gen()
