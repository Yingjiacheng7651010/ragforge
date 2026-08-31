"""Document parsers: PDF, Markdown, Word and HTML -> structured sections/tables."""

from ragforge.ingestion.parsers.base import ParsedDocument, Parser, Section, Table
from ragforge.ingestion.parsers.html import HtmlParser
from ragforge.ingestion.parsers.markdown import MarkdownParser
from ragforge.ingestion.parsers.pdf import PDFParser
from ragforge.ingestion.parsers.registry import DEFAULT_REGISTRY, ParserRegistry
from ragforge.ingestion.parsers.word import WordParser

__all__ = [
    "DEFAULT_REGISTRY",
    "HtmlParser",
    "MarkdownParser",
    "PDFParser",
    "ParsedDocument",
    "Parser",
    "ParserRegistry",
    "Section",
    "Table",
    "WordParser",
]
