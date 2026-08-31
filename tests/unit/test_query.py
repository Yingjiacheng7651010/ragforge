"""Unit tests for query understanding: parsing, degradation and orchestration."""

import pytest

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import LLMResult, Message
from ragforge.core.llm.base import LLMConnectionError
from ragforge.query import (
    HydeGenerator,
    IntentRouter,
    PromptStore,
    QueryExpander,
    QueryRewriter,
    QueryUnderstandingService,
)
from tests.unit.fakes import FakeLLM

HISTORY = [
    Message(role="user", content="什么是 RAG？"),
    Message(role="assistant", content="RAG 是检索增强生成。"),
]


# --- individual services ---


async def test_intent_router_extracts_intent() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"intent": "factual"}'))
    router = IntentRouter(llm)

    intent = await router.classify("RAG 是什么？", HISTORY)

    assert intent == "factual"
    prompt = llm.complete_calls[0][0][-1].content
    assert "RAG 是什么？" in prompt
    assert "user: 什么是 RAG？" in prompt  # history rendered into the prompt


async def test_intent_router_degrades_on_invalid_json() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="抱歉，我无法回答。"))
    router = IntentRouter(llm)

    assert await router.classify("RAG 是什么？", ()) == "general"


async def test_rewriter_rewrites_query() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"rewritten_query": "RAG 的检索增强生成原理是什么"}'))
    rewriter = QueryRewriter(llm)

    rewritten = await rewriter.rewrite("它的原理是什么？", HISTORY)

    assert rewritten == "RAG 的检索增强生成原理是什么"


async def test_rewriter_degrades_to_original_query() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="not json at all"))
    rewriter = QueryRewriter(llm)

    assert await rewriter.rewrite("它的原理是什么？", HISTORY) == "它的原理是什么？"


async def test_expander_expands_and_truncates_to_num() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(text='{"queries": ["q1", "q2", "q3", "q4", "q5"]}')
    )
    expander = QueryExpander(llm)

    queries = await expander.expand("RAG 原理", num=3)

    assert queries == ["q1", "q2", "q3"]


async def test_expander_degrades_to_original_query() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"queries": "not-a-list"}'))
    expander = QueryExpander(llm)

    assert await expander.expand("RAG 原理", num=3) == ["RAG 原理"]


async def test_hyde_generates_document() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"document": "RAG 通过检索外部知识库增强生成质量。"}'))
    hyde = HydeGenerator(llm)

    doc = await hyde.generate("RAG 是什么？")

    assert doc == "RAG 通过检索外部知识库增强生成质量。"


async def test_hyde_degrades_to_original_query() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="broken"))
    hyde = HydeGenerator(llm)

    assert await hyde.generate("RAG 是什么？") == "RAG 是什么？"


# --- orchestration ---


async def test_understand_full_pipeline() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"intent": "factual"}'))
    llm.enqueue_completion(LLMResult(text='{"rewritten_query": "检索增强生成的基本原理"}'))
    llm.enqueue_completion(LLMResult(text='{"queries": ["原理", "工作机制", "应用"]}'))
    llm.enqueue_completion(LLMResult(text='{"document": "RAG 是检索增强生成。"}'))
    service = QueryUnderstandingService(llm)

    understanding = await service.understand("它是什么？", HISTORY)

    assert understanding.raw_query == "它是什么？"
    assert understanding.intent == "factual"
    assert understanding.rewritten_query == "检索增强生成的基本原理"
    assert understanding.expanded_queries == ["原理", "工作机制", "应用"]
    assert understanding.hyde_doc == "RAG 是检索增强生成。"
    assert len(llm.complete_calls) == 4


async def test_understand_feeds_rewritten_query_to_expand_and_hyde() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"intent": "factual"}'))
    llm.enqueue_completion(LLMResult(text='{"rewritten_query": "独立完整查询"}'))
    llm.enqueue_completion(LLMResult(text='{"queries": ["a"]}'))
    llm.enqueue_completion(LLMResult(text='{"document": "d"}'))
    service = QueryUnderstandingService(llm)

    await service.understand("它是什么？", HISTORY)

    expand_prompt = llm.complete_calls[2][0][-1].content
    hyde_prompt = llm.complete_calls[3][0][-1].content
    assert "独立完整查询" in expand_prompt
    assert "独立完整查询" in hyde_prompt


async def test_understand_with_all_steps_disabled() -> None:
    llm = FakeLLM()
    service = QueryUnderstandingService(
        llm,
        enable_intent=False,
        enable_rewrite=False,
        enable_expand=False,
        enable_hyde=False,
    )

    understanding = await service.understand("它是什么？", HISTORY)

    assert understanding.raw_query == "它是什么？"
    assert understanding.intent is None
    assert understanding.rewritten_query is None
    assert understanding.expanded_queries is None
    assert understanding.hyde_doc is None
    assert llm.complete_calls == []  # the LLM is never called


async def test_understand_with_partial_steps() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"intent": "factual"}'))
    service = QueryUnderstandingService(llm, enable_rewrite=False, enable_hyde=False)

    understanding = await service.understand("它是什么？", HISTORY)

    assert understanding.intent == "factual"
    assert understanding.rewritten_query is None
    assert understanding.hyde_doc is None
    assert llm.complete_calls != []  # enabled steps still ran


async def test_understand_degrades_when_llm_fails() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMConnectionError("provider down"))
    llm.enqueue_completion(LLMResult(text='{"rewritten_query": "ok"}'))
    llm.enqueue_completion(LLMResult(text='{"queries": ["a"]}'))
    llm.enqueue_completion(LLMResult(text='{"document": "d"}'))
    service = QueryUnderstandingService(llm)

    understanding = await service.understand("它是什么？", HISTORY)

    assert understanding.intent is None  # failed step degrades to None
    assert understanding.rewritten_query == "ok"  # later steps still run
    assert understanding.expanded_queries == ["a"]
    assert understanding.hyde_doc == "d"


# --- prompts ---


def test_prompt_store_missing_prompt_raises() -> None:
    store = PromptStore("does-not-exist")

    with pytest.raises(RAGForgeError) as exc_info:
        store.load("intent")

    assert exc_info.value.code == "E_PROMPT_NOT_FOUND"
