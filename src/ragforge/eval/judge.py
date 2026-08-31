"""LLM-as-judge generation metrics: faithfulness (P10) and answer relevance (P11)."""

from abc import ABC, abstractmethod

import structlog

from ragforge.core.llm import BaseLLM, Message
from ragforge.core.llm.base import _extract_json_object
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render

logger = structlog.get_logger(__name__)


class Judge(ABC):
    """LLM-as-judge that scores one aspect of a generated answer in [0, 1]."""

    @abstractmethod
    async def judge(self, **fields: str) -> float:
        """Return a score in [0, 1]; parsing failures degrade to 0.0."""
        raise NotImplementedError


class _JsonScoreJudge(Judge):
    """Shared implementation: render a template, call the LLM, extract the score."""

    def __init__(self, llm: BaseLLM, prompt_name: str, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load(prompt_name)

    async def judge(self, **fields: str) -> float:
        try:
            prompt_text = render(self._template, **fields)
            result = await self._llm.complete(
                [Message(role="user", content=prompt_text)],
                temperature=0.0,
            )
            data = _extract_json_object(result.text)
        except Exception as err:  # LLM failures degrade to the worst score
            logger.warning("judge call failed; scoring 0.0", error=str(err))
            data = None
        if data is None:
            return 0.0
        score = data.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            return max(0.0, min(1.0, float(score)))
        return 0.0


class FaithfulnessJudge(_JsonScoreJudge):
    """P10: is every claim in the answer supported by the context?"""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        super().__init__(llm, "p10_faithfulness", prompts)


class AnswerRelevanceJudge(_JsonScoreJudge):
    """P11: does the answer actually address the question?"""

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        super().__init__(llm, "p11_answer_relevance", prompts)
