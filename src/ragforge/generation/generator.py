"""Answer generation: assemble context, call the LLM, resolve citations (prompt P7)."""

import re
import time
from collections.abc import Sequence

from ragforge.core.llm import BaseLLM, Message
from ragforge.core.vector_store import SearchHit
from ragforge.generation.base import Citation, GenerationResult
from ragforge.generation.context import ContextAssembler
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render

_CITATION_RE = re.compile(r"\[(\d+)\]")


class Generator:
    """Generate a cited answer for a query given retrieval hits.

    Context is assembled within ``max_context_tokens``; the LLM is prompted
    with P7 to cite sources as [n]. Only citations actually referenced in
    the answer (and within range) are returned, so callers can look the
    chunks back up by ``chunk_id``.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        assembler: ContextAssembler | None = None,
        max_context_tokens: int = 2048,
        prompts: PromptStore | None = None,
    ) -> None:
        self._llm = llm
        self._assembler = assembler or ContextAssembler()
        self._max_context_tokens = max_context_tokens
        self._template = (prompts or DEFAULT_PROMPTS).load("generate")

    async def generate(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        stream: bool = False,
    ) -> GenerationResult:
        """Generate an answer; ``stream`` collects tokens from the streaming API."""
        context, citations = self._assembler.assemble(query, hits, self._max_context_tokens)
        prompt_text = render(self._template, query=query, context=context)

        started = time.monotonic()
        if stream:
            chunks = [
                part
                async for part in await self._llm.stream(
                    [Message(role="user", content=prompt_text)]
                )
            ]
            answer = "".join(chunks)
            return GenerationResult(
                answer=answer,
                citations=self._resolve_citations(answer, citations),
                latency_ms=(time.monotonic() - started) * 1000,
            )

        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        return GenerationResult(
            answer=result.text,
            citations=self._resolve_citations(result.text, citations),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            cost=result.cost,
        )

    def _resolve_citations(self, answer: str, citations: Sequence[Citation]) -> list[Citation]:
        """Keep only the citations the answer actually references, in appearance order."""
        referenced: list[Citation] = []
        seen: set[int] = set()
        for marker in _CITATION_RE.findall(answer):
            index = int(marker)
            if 1 <= index <= len(citations) and index not in seen:
                seen.add(index)
                referenced.append(citations[index - 1])
        return referenced
