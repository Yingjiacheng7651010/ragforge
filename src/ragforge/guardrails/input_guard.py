"""Input guard: classify user input before it reaches the pipeline (prompt P12)."""

from ragforge.guardrails.base import Guard


class InputGuard(Guard):
    """Block harmful / prompt-injection / out-of-scope user input (P12)."""

    prompt_name = "p12_input_guard"
    categories = frozenset({"safe", "harmful", "injection", "out_of_scope"})
    required_fields = ("user_input",)
