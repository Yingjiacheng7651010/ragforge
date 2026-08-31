"""Unit tests for Self-RAG: evaluation parsing and corrective retrieval branches."""

import pytest

from ragforge.core.llm import LLMResult
from ragforge.core.llm.base import LLMConnectionError
from ragforge.core.vector_store import SearchHit
from ragforge.ingestion import Chunk
from ragforge.retrieval import (
    CorrectiveRagRetriever,
    SelfRagEvaluator,
)
from tests.unit.fakes import FakeLLM, FakeRetriever


def chunked_hit(chunk_id: str, text: str, score: float = 1.0) -> SearchHit:
    chunk = Chunk(chunk_id=chunk_id, doc_id="doc", text=text)
    return SearchHit(chunk_id=chunk_id, score=score, chunk=chunk)


def make_chunks(*texts: str) -> list[Chunk]:
    return [Chunk(chunk_id=f"c{i}", doc_id="doc", text=text) for i, text in enumerate(texts)]


# --- evaluator ---


async def test_evaluate_sufficient_verdict() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "sufficient", "relevance": [true, false]}'))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("RAG 是什么？", make_chunks("a", "b"))

    assert assessment.verdict == "sufficient"
    assert assessment.relevance == [True, False]
    assert assessment.refined_query is None


async def test_evaluate_retry_verdict_with_refined_query() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text='{"verdict": "retry", "relevance": [false], '
            '"refined_query": "RAG 检索增强生成原理"}'
        )
    )
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("它是什么？", make_chunks("a"))

    assert assessment.verdict == "retry"
    assert assessment.refined_query == "RAG 检索增强生成原理"


async def test_evaluate_insufficient_verdict() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "insufficient", "relevance": []}'))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("q", make_chunks("a", "b"))

    assert assessment.verdict == "insufficient"
    assert assessment.relevance == [False, False]


async def test_evaluate_invalid_json_degrades_to_insufficient() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="抱歉，我无法判断。"))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("q", make_chunks("a", "b"))

    assert assessment.verdict == "insufficient"
    assert assessment.relevance == [False, False]


async def test_evaluate_unknown_verdict_degrades_to_insufficient() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "maybe", "relevance": [true]}'))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("q", make_chunks("a"))

    assert assessment.verdict == "insufficient"


async def test_evaluate_normalizes_relevance_length() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "sufficient", "relevance": [true]}'))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("q", make_chunks("a", "b", "c"))

    assert assessment.relevance == [True, False, False]


async def test_evaluate_llm_failure_degrades_to_insufficient() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMConnectionError("down"))
    evaluator = SelfRagEvaluator(llm)

    assessment = await evaluator.evaluate("q", make_chunks("a"))

    assert assessment.verdict == "insufficient"


async def test_evaluate_prompt_contains_query_and_chunks() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "sufficient", "relevance": [true]}'))
    evaluator = SelfRagEvaluator(llm)

    await evaluator.evaluate("RAG 是什么？", make_chunks("检索增强生成"))

    prompt = llm.complete_calls[0][0][-1].content
    assert "RAG 是什么？" in prompt
    assert "1. 检索增强生成" in prompt


# --- corrective retriever ---


async def test_corrective_sufficient_filters_by_relevance() -> None:
    retriever = FakeRetriever(
        [chunked_hit("c1", "one"), chunked_hit("c2", "two"), chunked_hit("c3", "three")]
    )
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(text='{"verdict": "sufficient", "relevance": [true, false, true]}')
    )
    corrective = CorrectiveRagRetriever(retriever=retriever, evaluator=SelfRagEvaluator(llm))

    hits = await corrective.retrieve("q", top_k=10)

    assert [h.chunk_id for h in hits] == ["c1", "c3"]  # irrelevant chunk filtered, order kept
    assert corrective.last_assessment is not None
    assert corrective.last_assessment.verdict == "sufficient"


async def test_corrective_retries_with_refined_query_then_succeeds() -> None:
    retriever = FakeRetriever([chunked_hit("c1", "old"), chunked_hit("c2", "old")])
    retriever.enqueue_hits([chunked_hit("c3", "new")])
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text='{"verdict": "retry", "relevance": [false, false], '
            '"refined_query": "refined query"}'
        )
    )
    llm.enqueue_completion(LLMResult(text='{"verdict": "sufficient", "relevance": [true]}'))
    corrective = CorrectiveRagRetriever(
        retriever=retriever,
        evaluator=SelfRagEvaluator(llm),
        max_retries=1,
    )

    hits = await corrective.retrieve("original query", top_k=10)

    assert [h.chunk_id for h in hits] == ["c3"]
    assert retriever.calls[0][0] == "original query"
    assert retriever.calls[1][0] == "refined query"  # second retrieval used the refined query
    assert len(retriever.calls) == 2


async def test_corrective_retry_exhausted_returns_empty() -> None:
    retriever = FakeRetriever([chunked_hit("c1", "a")])
    retriever.enqueue_hits([chunked_hit("c2", "b")])
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(text='{"verdict": "retry", "relevance": [false], "refined_query": "q2"}')
    )
    llm.enqueue_completion(
        LLMResult(text='{"verdict": "retry", "relevance": [false], "refined_query": "q3"}')
    )
    corrective = CorrectiveRagRetriever(
        retriever=retriever,
        evaluator=SelfRagEvaluator(llm),
        max_retries=1,
    )

    hits = await corrective.retrieve("q1", top_k=10)

    assert hits == []  # retries exhausted, still not sufficient
    assert len(retriever.calls) == 2  # bounded: original + 1 retry, no infinite loop
    assert corrective.last_assessment is not None
    assert corrective.last_assessment.verdict == "retry"


async def test_corrective_insufficient_returns_empty_and_marks() -> None:
    retriever = FakeRetriever([chunked_hit("c1", "a")])
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "insufficient", "relevance": [false]}'))
    corrective = CorrectiveRagRetriever(retriever=retriever, evaluator=SelfRagEvaluator(llm))

    hits = await corrective.retrieve("q", top_k=10)

    assert hits == []
    assert len(retriever.calls) == 1  # no retry on insufficient
    assert corrective.last_assessment is not None
    assert corrective.last_assessment.verdict == "insufficient"  # the "资料不足" marker


async def test_corrective_retry_without_refined_query_stops() -> None:
    retriever = FakeRetriever([chunked_hit("c1", "a")])
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(text='{"verdict": "retry", "relevance": [false], "refined_query": null}')
    )
    corrective = CorrectiveRagRetriever(
        retriever=retriever,
        evaluator=SelfRagEvaluator(llm),
        max_retries=2,
    )

    hits = await corrective.retrieve("q", top_k=10)

    assert hits == []
    assert len(retriever.calls) == 1  # no refined query -> stop immediately


def test_corrective_rejects_retry_loop_config() -> None:
    with pytest.raises(ValueError):
        CorrectiveRagRetriever(
            retriever=FakeRetriever(),
            evaluator=SelfRagEvaluator(FakeLLM()),
            max_retries=3,
        )
