"""Guardrails: LLM-based input/output safety checks (prompts P12/P13).

No keyword matching: every check is a semantic LLM judgment. Guard failures
(LLM errors, invalid JSON, unknown categories) degrade fail-closed to
``block`` so unsafe content is never let through silently.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Literal

import structlog

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import BaseLLM, LLMError, Message
from ragforge.core.llm.base import _extract_json_object
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render

logger = structlog.get_logger(__name__)

Verdict = Literal["pass", "block"]


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a guard check."""

    verdict: Verdict
    category: str
    reason: str = ""


class Guard(ABC):
    """Base class for LLM-based guards (fail-closed)."""

    #: prompt template name in the PromptStore
    prompt_name: str
    #: accepted categories; anything else degrades to a block
    categories: frozenset[str]
    #: keyword fields the check() call must provide
    required_fields: tuple[str, ...] = ()

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load(self.prompt_name)

    async def check(self, **fields: str) -> GuardResult:
        """Evaluate the input/output and return the guard verdict (never raises)."""
        missing = [name for name in self.required_fields if name not in fields]
        if missing:
            raise ValueError(f"missing guard fields: {missing}")

        try:
            prompt_text = render(self._template, **fields)
            result = await self._llm.complete(
                [Message(role="user", content=prompt_text)],
                temperature=0.0,
            )
            data = _extract_json_object(result.text)
        except LLMError as err:
            logger.warning("guard evaluation failed; blocking", error=err.message)
            data = None

        if data is None:
            return self._block("guard_error", "guard evaluation failed; blocking by default")

        raw_category = str(data.get("category", "")).strip()
        if raw_category not in self.categories:
            return self._block("guard_error", f"unknown guard category {raw_category!r}")
        return GuardResult(
            verdict="pass" if raw_category == "safe" else "block",
            category=raw_category,
            reason=str(data.get("reason", "")),
        )

    async def enforce(self, **fields: str) -> GuardResult:
        """Check and raise ``E_GUARD_BLOCKED`` when the verdict is ``block``.

        The API layer catches this code and returns a friendly message.
        """
        result = await self.check(**fields)
        if result.verdict == "block":
            raise RAGForgeError(
                f"guard blocked ({result.category}): {result.reason}",
                code="E_GUARD_BLOCKED",
            )
        return result

    @staticmethod
    def _block(category: str, reason: str) -> GuardResult:
        return GuardResult(verdict="block", category=category, reason=reason)
