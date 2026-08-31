"""Reflector: judge whether the intermediate answer is good enough (prompt P15)."""

from typing import Literal

from ragforge.core.llm import BaseLLM, Message
from ragforge.core.llm.base import _extract_json_object
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render

Verdict = Literal["ok", "revise"]


class Reflector:
    """P15: verdict ``ok`` or ``revise`` (with feedback) on an intermediate answer.

    Parsing/LLM failures degrade to ``revise``: when unsure, try once more
    (the engine caps the retries, so this can never loop forever).
    """

    def __init__(self, llm: BaseLLM, prompts: PromptStore | None = None) -> None:
        self._llm = llm
        self._template = (prompts or DEFAULT_PROMPTS).load("p15_reflector")

    async def reflect(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> tuple[Verdict, str]:
        prompt_text = render(self._template, query=query, context=context, answer=answer)
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        data = _extract_json_object(result.text)
        if data is None:
            return "revise", "反思评估失败，尝试修正后重试。"
        raw_verdict = data.get("verdict")
        verdict: Verdict = "ok" if raw_verdict == "ok" else "revise"
        feedback = str(data.get("feedback", ""))
        return verdict, feedback or ("无修正建议" if verdict == "revise" else "")
