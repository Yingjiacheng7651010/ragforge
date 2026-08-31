"""Markdown parser: heading tree (#/##/###), body text, pipe tables, code fences."""

import re
from pathlib import Path

from ragforge.ingestion.parsers.base import HeadingStack, ParsedDocument, Parser, Section, Table

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^```")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


class MarkdownParser(Parser):
    """Parse Markdown into sections with heading paths and pipe tables.

    Only headings split sections: body text, tables and code fences between
    two headings all belong to the current section. Tables are extracted
    separately (first row = headers, separator rows dropped).
    """

    def parse(self, path: str | Path) -> ParsedDocument:
        source = Path(path)
        lines = source.read_text(encoding="utf-8").splitlines()

        headings = HeadingStack()
        sections: list[Section] = []
        tables: list[Table] = []
        paragraphs: list[str] = []
        current_lines: list[str] = []
        title: str | None = None
        in_fence = False
        pending_table: list[str] = []

        def flush_paragraph() -> None:
            text = "\n".join(current_lines).strip()
            current_lines.clear()
            if text:
                paragraphs.append(text)

        def flush_section() -> None:
            flush_paragraph()
            text = "\n\n".join(paragraphs).strip()
            paragraphs.clear()
            if text:
                sections.append(Section(heading_path=headings.path, text=text))

        def flush_table() -> None:
            if len(pending_table) >= 2:
                headers = _split_table_row(pending_table[0])
                rows = [_split_table_row(row) for row in pending_table[1:]]
                if headers:
                    tables.append(Table(headers=headers, rows=rows))
            pending_table.clear()

        for line in lines:
            stripped = line.strip()

            if _FENCE_RE.match(stripped):
                in_fence = not in_fence
                current_lines.append(line)
                continue
            if in_fence:
                current_lines.append(line)
                continue

            if not stripped:
                flush_table()
                flush_paragraph()
                continue

            if _TABLE_ROW_RE.match(stripped):
                if pending_table:
                    if _TABLE_SEPARATOR_RE.match(stripped):
                        continue  # separator row: header marker, not data
                    pending_table.append(stripped)
                    continue
                pending_table.append(stripped)
                flush_paragraph()
                continue

            if pending_table:
                flush_table()

            match = _HEADING_RE.match(stripped)
            if match:
                flush_section()
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                headings.push(level, heading_text)
                if title is None and level == 1:
                    title = heading_text
                continue

            current_lines.append(line)

        flush_table()
        flush_section()

        return ParsedDocument(
            doc_id=source.stem,
            title=title or source.stem,
            sections=sections,
            tables=tables,
            metadata={"format": "markdown", **self._source_metadata(source)},
        )


def _split_table_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]
