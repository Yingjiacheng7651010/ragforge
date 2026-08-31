"""Generation data model."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """A source the answer cites; ``index + 1`` in the citations list is the [n] marker."""

    chunk_id: str
    page: int | None = None
    text: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    """The generated answer with its citations and usage metrics."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
