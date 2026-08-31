"""Output guard: detect hallucinations and unsafe content in answers (prompt P13)."""

from ragforge.guardrails.base import Guard


class OutputGuard(Guard):
    """Block hallucinated or unsafe generated answers (P13)."""

    prompt_name = "p13_output_guard"
    categories = frozenset({"safe", "hallucination", "unsafe"})
    required_fields = ("context", "answer")
