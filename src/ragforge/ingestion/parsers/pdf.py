"""PDF parser based on PyMuPDF: text with page numbers and font-size heading detection."""

from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from ragforge.ingestion.parsers.base import HeadingStack, ParsedDocument, Parser, Section, Table

#: A line is a heading when its largest font size exceeds the body size by this factor.
_HEADING_SIZE_FACTOR = 1.15


class PDFParser(Parser):
    """Extract text, tables and a heading tree from a PDF.

    PDFs carry no structural headings, so heading levels are inferred from
    font sizes: the most frequent size is the body, larger sizes become
    heading levels (largest = level 1).
    """

    def parse(self, path: str | Path) -> ParsedDocument:
        source = Path(path)
        doc: Any = pymupdf.open(source)  # type: ignore[no-untyped-call]  # pymupdf.open is untyped

        line_sizes: list[float] = []
        page_lines: list[list[Any]] = []
        tables: list[Table] = []
        for page in doc:
            page_lines.append(_page_lines(page))
            for found in page.find_tables():
                rows = found.extract() or []
                if rows and rows[0]:
                    tables.append(
                        Table(
                            headers=list(rows[0]),
                            rows=[list(row) for row in rows[1:]],
                            page=page.number + 1,
                        )
                    )
            for line in page_lines[-1]:
                line_sizes.append(max(float(span["size"]) for span in line["spans"]))

        body_size = _body_font_size(line_sizes)
        heading_sizes = sorted(
            {size for size in line_sizes if size > body_size * _HEADING_SIZE_FACTOR},
            reverse=True,
        )
        level_by_size = {size: index + 1 for index, size in enumerate(heading_sizes)}

        headings = HeadingStack()
        sections: list[Section] = []
        title: str | None = None
        paragraph_lines: list[str] = []
        paragraph_page: int | None = None
        current_page: int | None = None

        def flush_paragraph() -> None:
            text = "\n".join(paragraph_lines).strip()
            if text:
                sections.append(
                    Section(heading_path=headings.path, text=text, page=paragraph_page)
                )
            paragraph_lines.clear()

        for page_index, lines in enumerate(page_lines):
            current_page = page_index + 1
            for line in lines:
                size = max(float(span["size"]) for span in line["spans"])
                level = level_by_size.get(size)
                if level is not None:
                    flush_paragraph()
                    heading_text = "".join(span["text"] for span in line["spans"]).strip()
                    headings.push(level, heading_text)
                    if title is None and level == 1:
                        title = heading_text
                    continue
                if not paragraph_lines:
                    paragraph_page = current_page
                paragraph_lines.append("".join(span["text"] for span in line["spans"]).rstrip())
        flush_paragraph()

        metadata: dict[str, Any] = doc.metadata or {}
        page_count = len(page_lines)
        doc.close()

        return ParsedDocument(
            doc_id=source.stem,
            title=title or metadata.get("title") or source.stem,
            sections=sections,
            tables=tables,
            metadata={
                "format": "pdf",
                "page_count": page_count,
                **self._source_metadata(source),
            },
        )


def _body_font_size(line_sizes: list[float]) -> float:
    """Body size: the most frequent size, ties resolve to the smallest."""
    if not line_sizes:
        return 12.0
    counts = Counter(line_sizes)
    max_count = max(counts.values())
    return min(size for size, count in counts.items() if count == max_count)


def _page_lines(page: Any) -> list[Any]:
    """Return one dict per text line: ``{"text": str, "spans": [...]}``."""
    page_dict = page.get_text("dict")
    lines: list[Any] = []
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue  # skip image blocks
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            lines.append(
                {
                    "text": "".join(span.get("text", "") for span in spans),
                    "spans": spans,
                }
            )
    return lines
