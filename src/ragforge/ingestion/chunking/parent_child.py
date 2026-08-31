"""Parent-child chunker: coarse parents (~800 tokens) containing linked children (~200 tokens)."""

from ragforge.ingestion.chunking.base import Chunk, Chunker, TokenCounter, split_text_by_structure
from ragforge.ingestion.parsers import ParsedDocument


class ParentChildChunker(Chunker):
    """Two-level chunks linked by ``parent_id``.

    Parents are structure-split at ``parent_max_tokens`` (default ~800);
    each parent is then split again at ``child_max_tokens`` (default ~200).
    Children carry the same heading path/page as their parent and point at
    it via ``parent_id``; both levels use the shared idempotent id sequence.
    """

    def __init__(
        self,
        *,
        parent_max_tokens: int = 800,
        child_max_tokens: int = 200,
        overlap_tokens: int = 0,
        token_counter: TokenCounter | None = None,
        chars_per_token: int = 4,
    ) -> None:
        if parent_max_tokens < 1 or child_max_tokens < 1:
            raise ValueError("parent_max_tokens and child_max_tokens must be >= 1")
        super().__init__(token_counter=token_counter, chars_per_token=chars_per_token)
        self._parent_max_tokens = parent_max_tokens
        self._child_max_tokens = child_max_tokens
        self._overlap_tokens = overlap_tokens

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for section in doc.sections:
            for parent_text in split_text_by_structure(
                section.text,
                self._parent_max_tokens,
                self._overlap_tokens,
                self._count,
                chars_per_token=self._chars_per_token,
            ):
                parent = self._make_chunk(
                    doc.doc_id,
                    parent_text,
                    section.heading_path,
                    section.page,
                    index,
                    metadata={"strategy": "parent-child", "kind": "parent"},
                )
                index += 1
                chunks.append(parent)
                for child_text in split_text_by_structure(
                    parent_text,
                    self._child_max_tokens,
                    0,
                    self._count,
                    chars_per_token=self._chars_per_token,
                ):
                    chunks.append(
                        self._make_chunk(
                            doc.doc_id,
                            child_text,
                            section.heading_path,
                            section.page,
                            index,
                            parent_id=parent.chunk_id,
                            metadata={"strategy": "parent-child", "kind": "child"},
                        )
                    )
                    index += 1
        return chunks
