"""HTML parser using the stdlib HTMLParser: strips navigation/scripts, keeps content."""

from html.parser import HTMLParser
from pathlib import Path

from ragforge.ingestion.parsers.base import HeadingStack, ParsedDocument, Parser, Section, Table

_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside"}
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCK_TAGS = {"p", "div", "li", "br", "section", "article", "blockquote", "pre"}


class _HtmlExtractor(HTMLParser):
    """Event-driven extractor for titles, heading paths, paragraphs and tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.first_h1: str | None = None
        self.sections: list[Section] = []
        self.tables: list[Table] = []

        self._skip_depth = 0
        self._in_title = False
        self._heading_level: int | None = None
        self._heading_buf: list[str] = []
        self._body_buf: list[str] = []
        self._paragraphs: list[str] = []
        self._headings = HeadingStack()
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_buf: list[str] = []

    # -- helpers --

    def _flush_body_buffer(self) -> None:
        text = " ".join(self._body_buf).strip()
        self._body_buf.clear()
        if text:
            self._paragraphs.append(text)

    def _flush_paragraph(self) -> None:
        self._flush_body_buffer()
        text = "\n\n".join(self._paragraphs).strip()
        self._paragraphs.clear()
        if text:
            self.sections.append(Section(heading_path=self._headings.path, text=text))

    def _flush_heading(self) -> None:
        title = " ".join(self._heading_buf).strip()
        self._heading_buf.clear()
        level = self._heading_level
        self._heading_level = None
        if level is None or not title:
            return
        self._flush_paragraph()
        self._headings.push(level, title)
        if level == 1 and self.first_h1 is None:
            self.first_h1 = title

    def _flush_table(self) -> None:
        if self._table:
            rows = self._table
            if rows:
                self.tables.append(Table(headers=list(rows[0]), rows=rows[1:]))
            self._table = None
        self._row = None

    # -- HTMLParser callbacks --

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "table":
            self._flush_paragraph()
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
            self._table.append(self._row)
        elif tag in ("td", "th") and self._row is not None:
            self._cell_buf = []
        elif tag in _HEADING_LEVELS:
            self._flush_body_buffer()
            self._heading_level = _HEADING_LEVELS[tag]
        elif tag in _BLOCK_TAGS:
            self._flush_body_buffer()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "table":
            self._flush_table()
        elif tag == "tr":
            self._row = None
        elif tag in ("td", "th"):
            self._flush_cell()
        elif tag in _HEADING_LEVELS:
            self._flush_heading()
        elif tag in _BLOCK_TAGS:
            self._flush_body_buffer()

    def _flush_cell(self) -> None:
        if self._row is not None:
            self._row.append(" ".join(self._cell_buf).strip())
        self._cell_buf = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
        elif self._heading_level is not None:
            self._heading_buf.append(data.strip())
        elif self._row is not None:
            self._cell_buf.append(data.strip())
        else:
            self._body_buf.append(data.strip())


class HtmlParser(Parser):
    """Extract title, heading tree, paragraphs and tables from an HTML document."""

    def parse(self, path: str | Path) -> ParsedDocument:
        source = Path(path)
        raw = source.read_text(encoding="utf-8", errors="replace")
        extractor = _HtmlExtractor()
        extractor.feed(raw)
        extractor.close()

        return ParsedDocument(
            doc_id=source.stem,
            title=extractor.title or extractor.first_h1 or source.stem,
            sections=extractor.sections,
            tables=extractor.tables,
            metadata={"format": "html", **self._source_metadata(source)},
        )
