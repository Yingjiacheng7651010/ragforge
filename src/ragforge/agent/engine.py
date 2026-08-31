"""AgenticRagEngine: plan -> execute -> reflect -> revise loop (bounded)."""

from collections.abc import Mapping

from ragforge.agent.executor import Executor
from ragforge.agent.planner import Planner
from ragforge.agent.reflector import Reflector
from ragforge.core.llm import BaseLLM, Message
from ragforge.query.base import DEFAULT_PROMPTS, PromptStore, render


class AgenticRagEngine:
    """Solve multi-hop questions with a bounded plan/execute/reflect loop.

    Flow per round: plan (with feedback from the previous round when
    revising) -> execute steps -> generate an answer from the step results
    -> reflect. A ``revise`` verdict feeds the reflector's feedback back
    into the next plan; the loop runs at most ``max_rounds + 1`` times, so
    it can never spin forever.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        executor: Executor,
        reflector: Reflector,
        llm: BaseLLM,
        max_rounds: int = 2,
        prompts: PromptStore | None = None,
    ) -> None:
        if max_rounds < 0 or max_rounds > 2:
            raise ValueError("max_rounds must be 0..2 (avoid infinite loops)")
        self._planner = planner
        self._executor = executor
        self._reflector = reflector
        self._llm = llm
        self._max_rounds = max_rounds
        self._generate_template = (prompts or DEFAULT_PROMPTS).load("generate")

    async def run(self, query: str) -> str:
        """Answer ``query``; returns the best answer after at most 3 rounds."""
        feedback: str | None = None
        answer = ""

        for round_index in range(self._max_rounds + 1):
            steps = await self._planner.plan(query, feedback=feedback)
            results = await self._executor.execute(steps)
            context = self._format_context(results)
            answer = await self._generate(query, context)
            verdict, revision = await self._reflector.reflect(query, context, answer)
            if verdict == "ok":
                return answer
            feedback = f"第 {round_index + 1} 轮反思：{revision}"

        return answer  # rounds exhausted; return the last answer

    async def _generate(self, query: str, context: str) -> str:
        prompt_text = render(self._generate_template, query=query, context=context)
        result = await self._llm.complete(
            [Message(role="user", content=prompt_text)],
            temperature=0.0,
        )
        return result.text

    @staticmethod
    def _format_context(results: Mapping[str, str]) -> str:
        return "\n\n".join(f"[{step_id}] {text}" for step_id, text in results.items())
