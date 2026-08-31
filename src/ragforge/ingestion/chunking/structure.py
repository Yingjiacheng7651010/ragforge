"""Structure-based chunker: one chunk per section, paragraph-level cuts for long sections."""

from ragforge.ingestion.chunking.base import Chunk, Chunker, TokenCounter, split_text_by_structure
from ragforge.ingestion.parsers import ParsedDocument


class StructureChunker(Chunker):
    """Split by section, keeping heading paths and pages.

    A section that exceeds ``max_tokens`` is cut at paragraph level (falling
    back to sentence and character windows for oversized paragraphs).
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        overlap_tokens: int = 0,
        token_counter: TokenCounter | None = None,
        chars_per_token: int = 4,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        super().__init__(token_counter=token_counter, chars_per_token=chars_per_token)
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in doc.sections:
            for piece in split_text_by_structure(
                section.text,
                self._max_tokens,
                self._overlap_tokens,
                self._count,
                chars_per_token=self._chars_per_token,
            ):
                chunks.append(
                    self._make_chunk(
                        doc.doc_id,
                        piece,
                        section.heading_path,
                        section.page,
                        index,
                        metadata={"strategy": "structure"},
                    )
                )
                index += 1
        return chunks
