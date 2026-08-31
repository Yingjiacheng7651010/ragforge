"""End-to-end flow tests: parse -> chunk -> index -> retrieve -> generate.

Uses real pipeline components with scripted LLMs and an in-memory store;
no external services required. The golden dataset samples are baked in as
assertions so regressions in the core flow surface here.
"""

import json
import pathlib

from ragforge.core.llm import LLMResult
from ragforge.generation import Generator
from ragforge.ingestion import Chunk, MarkdownParser, ParsedDocument, StructureChunker
from ragforge.retrieval import (
    CorrectiveRagRetriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalPipeline,
    SelfRagEvaluator,
    SparseRetriever,
)
from tests.unit.fakes import FakeEmbedding, FakeLLM, InMemoryVectorStore

DOC = """# RAG 入门

## 检索

RAG 系统首先从知识库检索相关片段。

## 生成

生成模型基于检索结果组织最终答案。

## 语义缓存

缓存命中要求查询向量相似度超过阈值 0.92。
"""


def make_llm_chain() -> tuple[list[FakeLLM], list[str]]:
    """Return (llms, answer) for the full chain: rewrite -> eval -> generate."""
    rewrite_llm = FakeLLM()
    rewrite_llm.enqueue_completion(
        LLMResult(text='{"rewritten_query": "RAG 检索增强生成的工作原理"}')
    )
    eval_llm = FakeLLM()
    eval_llm.enqueue_completion(
        LLMResult(text='{"verdict": "sufficient", "relevance": [true, true]}')
    )
    gen_llm = FakeLLM()
    gen_llm.enqueue_completion(
        LLMResult(
            text="RAG 先检索知识库 [1]，再由生成模型组织答案 [2]。",
            prompt_tokens=60,
            completion_tokens=20,
            cost=0.0008,
            latency_ms=500.0,
        )
    )
    return [rewrite_llm, eval_llm, gen_llm], "RAG 先检索知识库 [1]，再由生成模型组织答案 [2]。"


async def test_end_to_end_parse_chunk_index_retrieve_generate(tmp_path: pathlib.Path) -> None:
    # 1. parse a real markdown file
    path = tmp_path / "rag.md"
    path.write_text(DOC, encoding="utf-8")
    parsed: ParsedDocument = MarkdownParser().parse(path)
    assert all(section.heading_path[0] == "RAG 入门" for section in parsed.sections)
    assert len(parsed.sections) == 3

    # 2. chunk and index into the in-memory store
    chunker = StructureChunker(max_tokens=200, token_counter=len, chars_per_token=1)
    chunks = chunker.split(parsed)
    embedder = FakeEmbedding(dimensions=3)
    store = InMemoryVectorStore(embedder=embedder)
    await store.add(chunks)

    # 3. hybrid retrieval + corrective + pipeline
    llms, _ = make_llm_chain()
    rewrite_llm, eval_llm = llms[:2]
    hybrid = HybridRetriever(
        dense=DenseRetriever(store=store, embedder=embedder),
        sparse=SparseRetriever(store=store),
    )
    corrective = CorrectiveRagRetriever(
        retriever=hybrid,
        evaluator=SelfRagEvaluator(eval_llm),
    )
    pipeline = RetrievalPipeline(retriever=corrective, recall_k=10, rerank_n=5)
    hits = await pipeline.retrieve("RAG 检索增强生成", 5)

    assert hits, "expected hits from the indexed document"

    # 4. generate a cited answer and verify the citations resolve to chunks
    gen_llm = llms[2]
    generator = Generator(llm=gen_llm, max_context_tokens=500)
    result = await generator.generate("RAG 是怎么工作的？", hits)

    assert result.answer == "RAG 先检索知识库 [1]，再由生成模型组织答案 [2]。"
    assert result.citations, "answer must carry citations"
    assert all(citation.chunk_id in {hit.chunk_id for hit in hits} for citation in result.citations)
    assert result.cost > 0.0


async def test_golden_dataset_samples_flow_with_citations() -> None:
    """Golden dataset samples: each question produces a cited answer."""
    golden_path = pathlib.Path("data/golden/qa.jsonl")
    assert golden_path.exists(), "golden dataset must be committed"

    records = [
        json.loads(line)
        for line in golden_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) >= 3

    embedder = FakeEmbedding(dimensions=3)
    store = InMemoryVectorStore(embedder=embedder)
    # seed the store with the golden chunks so retrieval can hit them
    for record in records:
        for chunk_id in record["gold_chunks"]:
            await store.add(
                [
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=record.get("doc_id", "doc"),
                        text=f"{record['question']} 的答案依据内容",
                    )
                ]
            )

    for record in records:
        rewrite_llm = FakeLLM()
        rewrite_llm.enqueue_completion(
            LLMResult(text=f'{{"rewritten_query": "{record["question"]}"}}')
        )
        eval_llm = FakeLLM()
        eval_llm.enqueue_completion(
            LLMResult(text='{"verdict": "sufficient", "relevance": [true]}')
        )
        gen_llm = FakeLLM()
        gen_llm.enqueue_completion(
            LLMResult(
                text=f"根据资料，答案是：{record['gold_answer']} [1]。",
                prompt_tokens=10,
                completion_tokens=5,
                cost=0.0001,
                latency_ms=100.0,
            )
        )

        hybrid = HybridRetriever(
            dense=DenseRetriever(store=store, embedder=embedder),
            sparse=SparseRetriever(store=store),
        )
        corrective = CorrectiveRagRetriever(
            retriever=hybrid,
            evaluator=SelfRagEvaluator(eval_llm),
        )
        pipeline = RetrievalPipeline(retriever=corrective, recall_k=10, rerank_n=5)
        hits = await pipeline.retrieve(record["question"], 5)

        result = await Generator(llm=gen_llm, max_context_tokens=500).generate(
            record["question"], hits
        )

        assert record["gold_answer"] in result.answer
        assert result.citations, f"{record['id']}: answer must cite a source"
        cited_ids = {citation.chunk_id for citation in result.citations}
        retrieved_ids = {hit.chunk_id for hit in hits}
        assert cited_ids <= retrieved_ids, (
            f"{record['id']}: citations must reference retrieved chunks"
        )
        # the answer cites the golden chunk when it was retrieved
        if set(record["gold_chunks"]) & retrieved_ids:
            assert cited_ids & set(record["gold_chunks"])


def test_eval_cli_runs_on_golden_dataset(tmp_path: pathlib.Path) -> None:
    """The eval CLI produces a report with difficulty breakdown on the real dataset."""
    from ragforge.eval.cli import main

    output = tmp_path / "e2e_report.json"
    rc = main(
        [
            "run",
            "--dataset",
            "data/golden/qa.jsonl",
            "--metrics",
            "recall@5,mrr,hit_rate",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["retrieval"]["recall@5"] > 0.0
    assert set(report["by_difficulty"]) >= {"easy", "medium", "hard"}
    assert "qa_003" in report["failed_cases"]  # hard sample with no recall


def test_eval_module_entrypoint_runs() -> None:
    """``python -m ragforge.eval`` exits via the CLI main()."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "ragforge.eval", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert "--dataset" in completed.stdout

