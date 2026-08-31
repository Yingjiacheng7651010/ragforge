"""Unified LLM interface: message/result types, error taxonomy and the base class."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from ragforge.core.errors import RAGForgeError


@dataclass(frozen=True)
class Message:
    """A single chat message. Never pass bare dicts through the API."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResult:
    """Normalized completion result."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


class LLMError(RAGForgeError):
    """Base error raised by LLM providers.

    ``retryable`` marks idempotent failures (timeout, connection, 5xx) that
    are safe to retry or to fail over to another provider.
    """

    def __init__(self, message: str, *, retryable: bool = False, code: str = "E_LLM_ERROR") -> None:
        super().__init__(message, code=code)
        self.retryable = retryable


class LLMTimeoutError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True, code="E_LLM_TIMEOUT")


class LLMConnectionError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True, code="E_LLM_CONNECTION")


class LLMStatusError(LLMError):
    def __init__(self, message: str, *, status_code: int, retryable: bool) -> None:
        super().__init__(message, retryable=retryable, code=f"E_LLM_STATUS_{status_code}")
        self.status_code = status_code


_STRUCTURED_PROMPT = (
    "Return a single JSON object (no markdown, no commentary) that conforms "
    "to the following JSON Schema:\n{schema}"
)


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Parse the first top-level JSON object out of ``text``.

    Handles plain JSON, markdown code fences and surrounding prose. Returns
    ``None`` when no valid JSON object is found.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class BaseLLM(ABC):
    """Common interface implemented by every LLM provider."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Complete a chat conversation and return a normalized result."""
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: Sequence[Message], **kwargs: object) -> AsyncIterator[str]:
        """Return an async iterator of text chunks for the conversation."""
        raise NotImplementedError

    async def complete_structured(
        self,
        messages: Sequence[Message],
        schema: dict[str, object],
        *,
        max_repairs: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        """Return a JSON object matching ``schema``.

        The schema is injected as a system instruction; when the model
        returns something that is not a JSON object, one repair pass asks it
        to respond again before giving up with ``E_LLM_JSON_INVALID``.
        """
        history = [
            *messages,
            Message(
                role="system",
                content=_STRUCTURED_PROMPT.format(schema=json.dumps(schema, ensure_ascii=False)),
            ),
        ]
        for _ in range(max_repairs + 1):
            result = await self.complete(history, temperature=temperature, max_tokens=max_tokens)
            data = _extract_json_object(result.text)
            if data is not None:
                return data
            history = [
                *history,
                Message(
                    role="user",
                    content="Your previous response was not a valid JSON object. "
                    "Respond again with ONLY the JSON object.",
                ),
            ]
        raise RAGForgeError(
            f"LLM returned invalid JSON after {max_repairs + 1} attempt(s)",
            code="E_LLM_JSON_INVALID",
        )
