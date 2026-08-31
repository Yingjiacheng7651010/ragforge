"""Query expansion: generate multiple queries from different angles (multi-query retrieval)."""

from ragforge.core.llm import BaseLLM, Message
from ragforge.query.base import (
    DEFAULT_PROMPTS,
    PromptStore,
    extract_json_field,
    render,
)


class QueryExpander:
    """Expand a query into ``num`` alternative phrasings (never fewer than the original)."""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("expand")

    async def expand(self, query: str, num: int = 3) -> list[str]:
        prompt_text = render(self._template, query=query, num=num)
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        data = extract_json_field(result.text, "queries")
        if isinstance(data, list):
            queries = [item for item in data if isinstance(item, str) and item.strip()]
            if queries:
                return queries[:num]
        return [query]
