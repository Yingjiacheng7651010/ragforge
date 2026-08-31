"""Parser registry: route files to parsers by extension."""

from collections.abc import Mapping
from pathlib import Path

from ragforge.core.errors import RAGForgeError
from ragforge.ingestion.parsers.base import ParsedDocument, Parser
from ragforge.ingestion.parsers.html import HtmlParser
from ragforge.ingestion.parsers.markdown import MarkdownParser
from ragforge.ingestion.parsers.pdf import PDFParser
from ragforge.ingestion.parsers.word import WordParser


class ParserRegistry:
    """Maps file extensions (case-insensitive) to parser classes."""

    def __init__(self, parsers: Mapping[str, type[Parser]] | None = None) -> None:
        self._parsers: dict[str, type[Parser]] = dict(parsers or {})

    def register(self, extension: str, parser_type: type[Parser]) -> None:
        """Register a parser for an extension, with or without the dot."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        self._parsers[ext] = parser_type

    def get(self, path: str | Path) -> Parser:
        """Return a fresh parser instance for the file at ``path``."""
        ext = Path(path).suffix.lower()
        parser_type = self._parsers.get(ext)
        if parser_type is None:
            raise RAGForgeError(
                f"no parser registered for extension {ext!r}",
                code="E_UNSUPPORTED_FORMAT",
            )
        return parser_type()

    def parse(self, path: str | Path) -> ParsedDocument:
        """Parse ``path`` by routing on its extension."""
        return self.get(path).parse(path)


DEFAULT_REGISTRY = ParserRegistry(
    {
        ".pdf": PDFParser,
        ".md": MarkdownParser,
        ".markdown": MarkdownParser,
        ".docx": WordParser,
        ".html": HtmlParser,
        ".htm": HtmlParser,
    }
)
