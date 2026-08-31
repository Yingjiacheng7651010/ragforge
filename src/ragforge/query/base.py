"""Query understanding: data model, prompt loading and shared parsing helpers.

Prompts are externalized to ``data/prompts/`` (never hardcoded); the render
helper uses plain placeholder replacement so JSON braces in templates are
safe. Every parsing failure degrades gracefully to a safe default instead
of raising.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import Message
from ragforge.core.llm.base import _extract_json_object


@dataclass(frozen=True)
class QueryUnderstanding:
    """The result of understanding a raw user query for retrieval."""

    raw_query: str
    intent: str | None = None
    rewritten_query: str | None = None
    expanded_queries: list[str] | None = field(default=None)
    hyde_doc: str | None = None


class PromptStore:
    """Loads prompt templates from a directory (one ``<name>.txt`` per service)."""

    def __init__(self, base_dir: str | Path = "data/prompts") -> None:
        self._base = Path(base_dir)

    def load(self, name: str) -> str:
        path = self._base / f"{name}.txt"
        if not path.exists():
            raise RAGForgeError(f"prompt template not found: {path}", code="E_PROMPT_NOT_FOUND")
        return path.read_text(encoding="utf-8")


DEFAULT_PROMPTS = PromptStore()


def render(template: str, **values: object) -> str:
    """Fill ``{name}`` placeholders without touching anything else (JSON-safe)."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def format_history(history: Sequence[Message]) -> str:
    if not history:
        return "（无）"
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def extract_json_field(text: str, field: str) -> object | None:
    """Extract one field from a JSON object embedded in ``text`` (None on failure)."""
    data = _extract_json_object(text)
    if data is None:
        return None
    return data.get(field)


def nonempty(value: object) -> str | None:
    """Return the string value if it is a non-empty string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
