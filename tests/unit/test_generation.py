"""Unit tests for generation: context assembly (budget/dedupe/restore/numbering) and citations."""

import pytest

from ragforge.core.llm import LLMResult
from ragforge.core.vector_store import SearchHit
from ragforge.generation import ContextAssembler, Generator
from ragforge.ingestion import Chunk
from tests.unit.fakes import FakeLLM


def chunked_hit(chunk_id: str, text: str, score: float = 1.0) -> SearchHit:
    chunk = Chunk(chunk_id=chunk_id, doc_id="doc", text=text)
    return SearchHit(chunk_id=chunk_id, score=score, chunk=chunk)


# --- context assembly ---


async def test_assemble_sorts_by_score_and_numbers() -> None:
    hits = [
        chunked_hit("a", "low", score=0.1),
        chunked_hit("b", "high", score=0.9),
        chunked_hit("c", "mid", score=0.5),
    ]
    assembler = ContextAssembler(token_counter=len)

    context, citations = assembler.assemble("q", hits, max_tokens=1000)

    assert context == "[1] high\n\n[2] mid\n\n[3] low"
    assert [c.chunk_id for c in citations] == ["b", "c", "a"]


async def test_assemble_never_exceeds_token_budget() -> None:
    # unique texts of ~40 chars each so dedup does not collapse them
    hits = [chunked_hit(f"c{i}", f"text-{i} " + "x" * 36, score=float(100 - i)) for i in range(10)]
    assembler = ContextAssembler(token_counter=len)

    context, _ = assembler.assemble("q", hits, max_tokens=100)

    assert len(context) <= 100  # 1 char == 1 token with the len counter
    assert context.count("[") == 2  # [1] and [2] fit (44 each), [3] would overflow


async def test_assemble_deduplicates_identical_text() -> None:
    hits = [
        chunked_hit("a", "same text", score=0.9),
        chunked_hit("b", "same text", score=0.8),
    ]
    assembler = ContextAssembler(token_counter=len)

    context, citations = assembler.assemble("q", hits, max_tokens=1000)

    assert context == "[1] same text"
    assert [c.chunk_id for c in citations] == ["a"]  # best score wins


async def test_assemble_restores_parent_for_child_hits() -> None:
    parent = Chunk(chunk_id="p1", doc_id="doc", text="parent text")
    child1 = Chunk(chunk_id="c1", doc_id="doc", text="child one", parent_id="p1")
    child2 = Chunk(chunk_id="c2", doc_id="doc", text="child two", parent_id="p1")
    hits = [
        SearchHit(chunk_id="c1", score=0.9, chunk=child1),
        SearchHit(chunk_id="c2", score=0.7, chunk=child2),
        SearchHit(chunk_id="p1", score=0.5, chunk=parent),
    ]
    assembler = ContextAssembler(token_counter=len)

    context, citations = assembler.assemble("q", hits, max_tokens=1000)

    assert "parent text" in context
    assert "child one" not in context and "child two" not in context
    assert len(citations) == 1  # both children collapsed into their parent
    assert citations[0].chunk_id == "p1"


async def test_assemble_skips_oversized_chunk_keeps_smaller_ones() -> None:
    hits = [
        chunked_hit("big", "z" * 200, score=0.9),
        chunked_hit("small", "abc", score=0.5),
    ]
    assembler = ContextAssembler(token_counter=len)

    context, citations = assembler.assemble("q", hits, max_tokens=50)

    assert "abc" in context
    assert "z" * 200 not in context
    assert [c.chunk_id for c in citations] == ["small"]


def test_assemble_rejects_invalid_budget() -> None:
    assembler = ContextAssembler()

    with pytest.raises(ValueError):
        assembler.assemble("q", [], max_tokens=0)


# --- generator ---


def make_generator(llm: FakeLLM, max_context_tokens: int = 500) -> Generator:
    return Generator(
        llm=llm,
        assembler=ContextAssembler(token_counter=len),
        max_context_tokens=max_context_tokens,
    )


async def test_generate_returns_cited_answer_with_metrics() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text="答案来自 [1] 和 [2]。",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.01,
            latency_ms=3.0,
        )
    )
    generator = make_generator(llm)
    hits = [chunked_hit("a", "text a", score=0.9), chunked_hit("b", "text b", score=0.5)]

    result = await generator.generate("q", hits)

    assert result.answer == "答案来自 [1] 和 [2]。"
    # citations resolve back to the assembled chunks, in answer order
    assert [c.chunk_id for c in result.citations] == ["a", "b"]
    assert result.citations[0].score == pytest.approx(0.9)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cost == pytest.approx(0.01)
    assert result.latency_ms == pytest.approx(3.0)
    # the prompt contains the numbered context
    prompt = llm.complete_calls[0][0][-1].content
    assert "[1] text a" in prompt
    assert "[2] text b" in prompt


async def test_generate_ignores_out_of_range_citations() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="答案见 [9]。"))
    generator = make_generator(llm)

    result = await generator.generate("q", [chunked_hit("a", "text a")])

    assert result.answer == "答案见 [9]。"
    assert result.citations == []  # [9] does not exist in the context


async def test_generate_citations_follow_appearance_order() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="先看 [2] 再看 [1] 和 [2]。"))
    generator = make_generator(llm)
    hits = [chunked_hit("a", "text a", score=0.9), chunked_hit("b", "text b", score=0.5)]

    result = await generator.generate("q", hits)

    assert [c.chunk_id for c in result.citations] == ["b", "a"]  # appearance order, deduped


async def test_generate_stream_collects_answer() -> None:
    llm = FakeLLM()
    llm.enqueue_stream(["答案 ", "来自 [1]", "。"])
    generator = make_generator(llm)

    result = await generator.generate("q", [chunked_hit("a", "text a")], stream=True)

    assert result.answer == "答案 来自 [1]。"
    assert [c.chunk_id for c in result.citations] == ["a"]


async def test_generate_context_respects_budget() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="ok"))
    generator = make_generator(llm, max_context_tokens=100)
    hits = [chunked_hit(f"c{i}", "y" * 60, score=float(100 - i)) for i in range(5)]

    await generator.generate("q", hits)

    prompt = llm.complete_calls[0][0][-1].content
    assert "[1] y" in prompt  # the first chunk made it into the context
    assert "[2] y" not in prompt  # a second chunk would overflow the 100-token budget
