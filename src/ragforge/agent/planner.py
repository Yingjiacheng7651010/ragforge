"""Planner: decompose a question into sub-problem steps with tools (prompt P14)."""

from collections.abc import Mapping

from ragforge.agent.base import _TOOLS, Step
from ragforge.core.llm import BaseLLM, Message
from ragforge.core.llm.base import _extract_json_object
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render


class Planner:
    """Turn a question into an ordered list of tool steps (P14).

    Parsing failures degrade to a single ``retrieve`` step on the original
    question; unknown tools are filtered out and the plan is capped at
    ``max_steps``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_steps: int = 8,
        prompts: PromptStore | None = None,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._template = (prompts or DEFAULT_PROMPTS).load("p14_planner")

    async def plan(self, query: str, feedback: str | None = None) -> list[Step]:
        prompt_text = render(
            self._template,
            query=query,
            feedback=feedback or "（首次规划）",
            max_steps=self._max_steps,
        )
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        return self._parse(result.text, query)

    def _parse(self, text: str, query: str) -> list[Step]:
        data = _extract_json_object(text)
        if data is None:
            return self._fallback(query)
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return self._fallback(query)
        steps: list[Step] = []
        seen_ids: set[str] = set()
        for item in raw_steps[: self._max_steps]:
            if not isinstance(item, Mapping):
                continue
            step_id = str(item.get("id", "")).strip() or str(len(steps) + 1)
            tool = str(item.get("tool", "")).strip()
            step_query = str(item.get("query", "")).strip()
            if tool not in _TOOLS or not step_query or step_id in seen_ids:
                continue
            seen_ids.add(step_id)
            steps.append(Step(id=step_id, tool=tool, query=step_query))
        return steps or self._fallback(query)

    @staticmethod
    def _fallback(query: str) -> list[Step]:
        return [Step(id="1", tool="retrieve", query=query)]
