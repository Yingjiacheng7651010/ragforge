"""Data model and base class for document parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Section:
    """A run of body text under a heading path."""

    heading_path: list[str]
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Table:
    """A table extracted from a document (first row treated as headers)."""

    headers: list[str]
    rows: list[list[str]]
    page: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized output of a parser."""

    doc_id: str
    title: str
    sections: list[Section]
    tables: list[Table] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class HeadingStack:
    """Tracks the current heading path while walking a document.

    Shared by every parser so heading-tree bookkeeping lives in one place:
    ``push(level, title)`` truncates deeper levels and records the new
    heading, returning the resulting path.
    """

    def __init__(self) -> None:
        self._levels: list[str] = []

    def push(self, level: int, title: str) -> list[str]:
        if level < 1:
            raise ValueError("heading level must be >= 1")
        self._levels = self._levels[: level - 1] + [title]
        return list(self._levels)

    @property
    def path(self) -> list[str]:
        return list(self._levels)


class Parser(ABC):
    """Parse one document file into a structured ``ParsedDocument``."""

    @abstractmethod
    def parse(self, path: str | Path) -> ParsedDocument:
        """Parse the file at ``path`` into a structured document."""
        raise NotImplementedError

    @staticmethod
    def _source_metadata(path: str | Path) -> dict[str, object]:
        return {"source": str(path)}
