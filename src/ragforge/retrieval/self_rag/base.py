"""Self-RAG assessment: evaluate retrieval quality before generation (prompt P6)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import structlog

from ragforge.core.llm import BaseLLM, LLMError, Message
from ragforge.core.llm.base import _extract_json_object
from ragforge.ingestion import Chunk
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render

logger = structlog.get_logger(__name__)

Verdict = Literal["sufficient", "retry", "insufficient"]
_VERDICTS: tuple[Verdict, ...] = ("sufficient", "retry", "insufficient")


@dataclass(frozen=True)
class SelfRagAssessment:
    """Result of evaluating retrieval quality for a query."""

    verdict: Verdict
    relevance: list[bool]
    refined_query: str | None = field(default=None)


def _format_chunks(chunks: Sequence[Chunk]) -> str:
    if not chunks:
        return "（无检索结果）"
    return "\n".join(f"{index + 1}. {chunk.text}" for index, chunk in enumerate(chunks))


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return False


def _normalize_relevance(value: object, size: int) -> list[bool]:
    """Coerce a parsed relevance field into a bool list of exactly ``size``."""
    if not isinstance(value, list):
        return [False] * size
    relevance = [_to_bool(item) for item in value]
    if len(relevance) < size:
        relevance.extend([False] * (size - len(relevance)))
    return relevance[:size]


class SelfRagEvaluator:
    """Ask the LLM to judge retrieval quality chunk-by-chunk (prompt P6).

    Any parsing failure or LLM error degrades conservatively to
    ``insufficient`` (with all-relevant=False), so the pipeline never
    proceeds on context it cannot vouch for.
    """

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("self_rag")

    async def evaluate(self, query: str, chunks: Sequence[Chunk]) -> SelfRagAssessment:
        try:
            prompt_text = render(self._template, query=query, chunks=_format_chunks(chunks))
            result = await self._llm.complete(
                [Message(role="user", content=prompt_text)],
                temperature=0.0,
            )
            data = _extract_json_object(result.text)
        except LLMError as err:
            logger.warning(
                "self-rag evaluation failed; treating as insufficient",
                error=err.message,
            )
            data = None

        if data is None:
            return SelfRagAssessment(verdict="insufficient", relevance=[False] * len(chunks))

        raw_verdict = data.get("verdict")
        verdict: Verdict = raw_verdict if raw_verdict in _VERDICTS else "insufficient"
        refined = data.get("refined_query")
        return SelfRagAssessment(
            verdict=verdict,
            relevance=_normalize_relevance(data.get("relevance"), len(chunks)),
            refined_query=refined if isinstance(refined, str) and refined.strip() else None,
        )
