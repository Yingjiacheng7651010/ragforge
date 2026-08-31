"""Orchestration: run the query-understanding steps, each independently toggleable."""

from collections.abc import Awaitable, Sequence
from typing import TypeVar

import structlog

from ragforge.core.llm import BaseLLM, LLMError, Message
from ragforge.query.base import PromptStore, QueryUnderstanding
from ragforge.query.expand import QueryExpander
from ragforge.query.hyde import HydeGenerator
from ragforge.query.intent import IntentRouter
from ragforge.query.rewrite import QueryRewriter

logger = structlog.get_logger(__name__)

_T = TypeVar("_T")


class QueryUnderstandingService:
    """Chain intent routing, rewriting, expansion and HyDE generation.

    Every step can be disabled at construction time; a disabled or failed
    step leaves its field as ``None`` (with a warning log), so the pipeline
    degrades gracefully instead of raising.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        enable_intent: bool = True,
        enable_rewrite: bool = True,
        enable_expand: bool = True,
        enable_hyde: bool = True,
        prompts: PromptStore | None = None,
    ) -> None:
        self._intent = IntentRouter(llm, prompts) if enable_intent else None
        self._rewriter = QueryRewriter(llm, prompts) if enable_rewrite else None
        self._expander = QueryExpander(llm, prompts) if enable_expand else None
        self._hyde = HydeGenerator(llm, prompts) if enable_hyde else None

    async def understand(self, query: str, history: Sequence[Message] = ()) -> QueryUnderstanding:
        """Run every enabled step; rewriting feeds expansion and HyDE."""
        intent = await self._run(self._intent.classify(query, history) if self._intent else None)
        rewritten = await self._run(
            self._rewriter.rewrite(query, history) if self._rewriter else None
        )

        base_query = rewritten or query
        expanded = await self._run(self._expander.expand(base_query) if self._expander else None)
        hyde_doc = await self._run(self._hyde.generate(base_query) if self._hyde else None)

        return QueryUnderstanding(
            raw_query=query,
            intent=intent,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            hyde_doc=hyde_doc,
        )

    async def _run(self, coro: Awaitable[_T] | None) -> _T | None:
        if coro is None:
            return None
        try:
            return await coro
        except LLMError as err:
            logger.warning(
                "query understanding step failed; degrading",
                error=err.message,
                code=err.code,
            )
            return None
