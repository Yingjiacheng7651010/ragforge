"""Query rewriting: resolve anaphora / ellipsis against the conversation history."""

from collections.abc import Sequence

from ragforge.core.llm import BaseLLM, Message
from ragforge.observability import span_set, traced
from ragforge.query.base import (
    DEFAULT_PROMPTS,
    PromptStore,
    extract_json_field,
    format_history,
    nonempty,
    render,
)


class QueryRewriter:
    """Rewrite the latest question into a standalone, retrieval-ready query."""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("rewrite")

    @traced("rag.rewrite")
    async def rewrite(self, query: str, history: Sequence[Message]) -> str:
        prompt_text = render(self._template, query=query, history=format_history(history))
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        rewritten = nonempty(extract_json_field(result.text, "rewritten_query")) or query
        span_set(query=query, rewritten_query=rewritten)
        return rewritten
