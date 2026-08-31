"""Intent routing: classify a query (with history) into a retrieval intent."""

from collections.abc import Sequence

from ragforge.core.llm import BaseLLM, Message
from ragforge.query.base import (
    DEFAULT_PROMPTS,
    PromptStore,
    extract_json_field,
    format_history,
    nonempty,
    render,
)

_DEFAULT_INTENT = "general"


class IntentRouter:
    """Classify the intent of a query using the LLM (prompt from ``data/prompts``)."""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("intent")

    async def classify(self, query: str, history: Sequence[Message]) -> str:
        prompt_text = render(self._template, query=query, history=format_history(history))
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        return nonempty(extract_json_field(result.text, "intent")) or _DEFAULT_INTENT
