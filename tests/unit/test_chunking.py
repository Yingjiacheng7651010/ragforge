"""Unit tests for chunking strategies: budgets, headings, links, overlap, idempotency."""

import pytest

from ragforge.ingestion.chunking import (
    ParentChildChunker,
    SemanticChunker,
    StructureChunker,
)
from ragforge.ingestion.parsers import ParsedDocument, Section

#: 1 char == 1 token, so token budgets map exactly onto character counts.
CHAR_COUNTER = len


def make_section(heading_path: list[str], text: str, page: int | None = None) -> Section:
    return Section(heading_path=heading_path, text=text, page=page)


def make_doc(*sections: Section, doc_id: str = "doc-1") -> ParsedDocument:
    return ParsedDocument(doc_id=doc_id, title="title", sections=list(sections))


LONG_TEXT = "\n\n".join(f"Paragraph {i} contains enough words to matter." for i in range(30))


# --- structure chunker ---


def test_structure_keeps_heading_path_and_page() -> None:
    doc = make_doc(
        make_section(["Chapter 1", "Intro"], "Short intro text.", page=1),
        make_section(["Chapter 1", "Usage"], "Usage paragraph.", page=2),
    )
    chunker = StructureChunker(
        max_tokens=1000,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ["Chapter 1", "Intro"]
    assert chunks[0].page == 1
    assert chunks[0].text == "Short intro text."
    assert chunks[1].heading_path == ["Chapter 1", "Usage"]
    assert chunks[0].metadata["strategy"] == "structure"


def test_structure_splits_long_section_within_budget() -> None:
    doc = make_doc(make_section(["Long"], LONG_TEXT))
    chunker = StructureChunker(
        max_tokens=200,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.heading_path == ["Long"] for c in chunks)
    assert "".join(c.text for c in chunks).replace("\n\n", "") == LONG_TEXT.replace("\n\n", "")


def test_structure_overlap_reattaches_tail() -> None:
    doc = make_doc(make_section(["Long"], LONG_TEXT))
    chunker = StructureChunker(
        max_tokens=200,
        overlap_tokens=10,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 + 10 for c in chunks)
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.text.startswith(previous.text[-10:])


def test_structure_split_is_idempotent() -> None:
    doc = make_doc(
        make_section(["Chapter 1", "Intro"], "Short intro text.", page=1),
        make_section(["Chapter 1", "Usage"], LONG_TEXT, page=2),
    )
    chunker = StructureChunker(
        max_tokens=200,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    first = chunker.split(doc)
    second = chunker.split(doc)

    assert first == second
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(
        len(c.chunk_id) == 40 and all(ch in "0123456789abcdef" for ch in c.chunk_id)
        for c in first
    )


def test_chunk_id_differs_by_doc_and_heading_path() -> None:
    chunker = StructureChunker(
        max_tokens=1000,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )
    docs = [
        make_doc(make_section(["A"], "same text"), doc_id="doc-1"),
        make_doc(make_section(["A"], "same text"), doc_id="doc-2"),
        make_doc(make_section(["B"], "same text"), doc_id="doc-1"),
    ]

    ids = {chunker.split(doc)[0].chunk_id for doc in docs}

    assert len(ids) == 3


def test_structure_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError):
        StructureChunker(max_tokens=0)


# --- semantic chunker ---


def topic_embed(texts: list[str]) -> list[list[float]]:
    """Sentences containing 'alpha' get [1,0], everything else gets [0,1]."""
    return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


def test_semantic_splits_at_low_similarity() -> None:
    text = "alpha one. alpha two. alpha three. beta one."
    doc = make_doc(make_section(["S"], text))
    chunker = SemanticChunker(
        embed_fn=topic_embed,
        max_tokens=1000,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    assert len(chunks) == 2
    assert "alpha" in chunks[0].text and "beta" not in chunks[0].text
    assert chunks[1].text == "beta one."
    assert all(c.heading_path == ["S"] for c in chunks)


def test_semantic_budget_cuts_oversized_group() -> None:
    text = "alpha one. alpha two. alpha three. alpha four. alpha five."
    doc = make_doc(make_section(["S"], text))
    chunker = SemanticChunker(
        embed_fn=topic_embed,
        max_tokens=20,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 20 for c in chunks)


def test_semantic_embeds_all_sentences_in_one_batch() -> None:
    calls: list[list[str]] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return topic_embed(texts)

    text = "alpha one. alpha two. beta one."
    chunker = SemanticChunker(
        embed_fn=recording_embed,
        max_tokens=1000,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunker.split(make_doc(make_section(["S"], text)))

    assert len(calls) == 1
    assert calls[0] == ["alpha one.", "alpha two.", "beta one."]


# --- parent-child chunker ---


def test_parent_child_links_children_to_parents() -> None:
    doc = make_doc(make_section(["Guide"], LONG_TEXT))
    chunker = ParentChildChunker(
        parent_max_tokens=150,
        child_max_tokens=30,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    chunks = chunker.split(doc)

    parents = [c for c in chunks if c.metadata["kind"] == "parent"]
    children = [c for c in chunks if c.metadata["kind"] == "child"]
    assert parents and children

    assert all(len(p.text) <= 150 for p in parents)
    assert all(len(c.text) <= 30 for c in children)
    assert all(c.heading_path == ["Guide"] for c in chunks)

    parent_by_id = {p.chunk_id: p for p in parents}
    for child in children:
        assert child.parent_id in parent_by_id
        assert child.text in parent_by_id[child.parent_id].text  # type: ignore[index]


def test_parent_child_is_idempotent() -> None:
    doc = make_doc(make_section(["Guide"], LONG_TEXT))
    chunker = ParentChildChunker(
        parent_max_tokens=150,
        child_max_tokens=30,
        token_counter=CHAR_COUNTER,
        chars_per_token=1,
    )

    first = chunker.split(doc)
    second = chunker.split(doc)

    assert first == second
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)  # no collisions


def test_parent_child_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        ParentChildChunker(parent_max_tokens=0)
    with pytest.raises(ValueError):
        ParentChildChunker(child_max_tokens=0)
