"""Executor: run planned steps against the available tools."""

from collections.abc import Mapping, Sequence

from ragforge.agent.base import Step, safe_calc
from ragforge.retrieval import Retriever


class Executor:
    """Execute steps with three tools: retrieve / calculator / lookup_table.

    ``tables`` maps table names to row lists (first row = headers). Tool
    failures never raise: they become textual error results so the loop can
    reflect and recover.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        tables: Mapping[str, Sequence[Sequence[str]]] | None = None,
        retrieve_top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._tables = tables or {}
        self._retrieve_top_k = retrieve_top_k

    async def execute(self, steps: Sequence[Step]) -> dict[str, str]:
        """Run each step and return ``{step_id: result_text}``."""
        results: dict[str, str] = {}
        for step in steps:
            results[step.id] = await self._run_tool(step.tool, step.query)
        return results

    async def _run_tool(self, tool: str, query: str) -> str:
        if tool == "retrieve":
            return await self._retrieve(query)
        if tool == "calculator":
            return self._calculate(query)
        if tool == "lookup_table":
            return self._lookup(query)
        return f"[unknown tool: {tool}]"

    async def _retrieve(self, query: str) -> str:
        hits = await self._retriever.retrieve(query, self._retrieve_top_k)
        if not hits:
            return "（检索无结果）"
        return "\n".join(
            f"{index + 1}. {hit.chunk.text if hit.chunk else hit.chunk_id}"
            for index, hit in enumerate(hits)
        )

    def _calculate(self, expression: str) -> str:
        try:
            return f"{safe_calc(expression):g}"
        except (ValueError, SyntaxError, ZeroDivisionError) as err:
            return f"[calc error: {err}]"

    def _lookup(self, table_name: str) -> str:
        rows = self._tables.get(table_name.strip())
        if not rows:
            return f"[table not found: {table_name}]"
        return "\n".join("\t".join(str(cell) for cell in row) for row in rows)
