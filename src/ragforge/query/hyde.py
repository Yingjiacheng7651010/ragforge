"""HyDE generation: synthesize a hypothetical document that would answer the query."""

from ragforge.core.llm import BaseLLM, Message
from ragforge.query.base import (
    DEFAULT_PROMPTS,
    PromptStore,
    extract_json_field,
    nonempty,
    render,
)


class HydeGenerator:
    """Generate a hypothetical answer document (HyDE) for the query."""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("hyde")

    async def generate(self, query: str) -> str:
        prompt_text = render(self._template, query=query)
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        return nonempty(extract_json_field(result.text, "document")) or query
