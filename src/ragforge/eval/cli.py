"""Evaluation CLI: ``python -m ragforge.eval run ...`` (docs/05-评估体系.md)."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from ragforge.config import get_settings
from ragforge.core.errors import RAGForgeError
from ragforge.eval.compare import compare_reports
from ragforge.eval.judge import AnswerRelevanceJudge, FaithfulnessJudge
from ragforge.eval.runner import EvaluationRunner
from ragforge.generation import Generator
from ragforge.providers import ElasticsearchStore, MilvusVectorStore, OpenAIEmbedding, OpenAILLM
from ragforge.retrieval import DenseRetriever, HybridRetriever, Retriever, SparseRetriever

DEFAULT_OUTPUT = "report/latest.json"
_ES_INDEX = os.environ.get("EVAL_ES_INDEX", "chunks")
_ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
_MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ragforge.eval",
        description="RAG evaluation harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run an evaluation over a golden dataset")
    run.add_argument("--dataset", required=True, help="path to the golden JSONL dataset")
    run.add_argument(
        "--retriever",
        default="offline",
        help="dense | sparse | hybrid | offline (dataset ids)",
    )
    run.add_argument(
        "--metrics",
        default="recall@5,mrr,hit_rate",
        help="comma-separated metrics: recall@k, precision@k, mrr, hit_rate, "
        "faithfulness, answer_relevance",
    )
    run.add_argument("--k", type=int, default=5, help="k for recall@k / precision@k (default 5)")
    run.add_argument("--recall-top-k", type=int, default=50, help="retriever top_k during recall")
    run.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        help="recall@k below this marks a failed case",
    )
    run.add_argument("--index", default=_ES_INDEX, help="vector store index/collection name")
    run.add_argument("--baseline", default=None, help="previous report JSON to diff against")
    run.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="report output path (default report/latest.json)",
    )
    return parser


def parse_metrics(spec: str, default_k: int) -> tuple[list[str], list[str]]:
    """Split 'recall@5,mrr' into metric names and a k per metric-bearing name."""
    names: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item in ("recall", "precision") and "@" not in item:
            names.append(f"{item}@{default_k}")
        elif item in ("mrr", "hit_rate", "faithfulness", "answer_relevance"):
            names.append(item)
        else:
            name, _, k_part = item.partition("@")
            if name not in ("recall", "precision", "hit_rate") or not k_part.isdigit():
                raise RAGForgeError(f"unknown metric spec {item!r}", code="E_EVAL_METRIC")
            names.append(f"{name}@{k_part}")
    if not names:
        raise RAGForgeError("no metrics requested", code="E_EVAL_METRIC")
    retrieval_names = [
        n for n in names if n.split("@")[0] in ("recall", "precision", "mrr", "hit_rate")
    ]
    return retrieval_names, names


def _require_llm() -> OpenAILLM:
    settings = get_settings()
    if not settings.llm_api_key.get_secret_value():
        raise RAGForgeError(
            "RAGFORGE_LLM_API_KEY is required for generation metrics",
            code="E_EVAL_LLM_MISSING",
        )
    return OpenAILLM(model=settings.llm_model, api_key=settings.llm_api_key)


def _require_embedder() -> OpenAIEmbedding:
    settings = get_settings()
    if not settings.llm_api_key.get_secret_value():
        raise RAGForgeError(
            "RAGFORGE_LLM_API_KEY is required to build the embedding provider",
            code="E_EVAL_LLM_MISSING",
        )
    return OpenAIEmbedding(
        model=settings.embedding_model,
        dimensions=settings.embedding_dim,
        api_key=settings.llm_api_key,
    )


def build_retriever(name: str, index: str) -> Retriever:
    """Build a retriever from environment config (ES_URL / MILVUS_URI)."""
    if name == "offline":
        raise RAGForgeError(
            "offline needs no retriever; use --retriever dense/sparse/hybrid",
            code="E_EVAL_RETRIEVER",
        )
    embedder = _require_embedder()

    def es_store() -> ElasticsearchStore:
        return ElasticsearchStore(
            hosts=_ES_URL,
            index_name=index,
            dimension=embedder.dimensions,
            embedder=embedder,
        )

    if name == "dense":
        return DenseRetriever(store=es_store(), embedder=embedder)
    if name == "sparse":
        return SparseRetriever(store=es_store())
    if name == "hybrid":
        return HybridRetriever(
            dense=DenseRetriever(store=es_store(), embedder=embedder),
            sparse=SparseRetriever(store=es_store()),
        )
    if name == "milvus":
        store = MilvusVectorStore(
            uri=_MILVUS_URI,
            collection_name=index,
            dimension=embedder.dimensions,
            embedder=embedder,
        )
        return DenseRetriever(store=store, embedder=embedder)
    raise RAGForgeError(f"unknown retriever {name!r}", code="E_EVAL_RETRIEVER")


def run(args: argparse.Namespace) -> int:
    retrieval_names, all_names = parse_metrics(args.metrics, args.k)
    generation_requested = any(n in all_names for n in ("faithfulness", "answer_relevance"))

    retriever: Retriever | None = None
    if args.retriever != "offline":
        retriever = build_retriever(args.retriever, args.index)

    generator = None
    faithfulness_judge = None
    relevance_judge = None
    if generation_requested:
        llm = _require_llm()
        generator = Generator(llm=llm)
        if "faithfulness" in all_names:
            faithfulness_judge = FaithfulnessJudge(llm)
        if "answer_relevance" in all_names:
            relevance_judge = AnswerRelevanceJudge(llm)

    runner = EvaluationRunner(
        retriever=retriever,
        k=args.k,
        recall_top_k=args.recall_top_k,
        metric_names=retrieval_names,
        fail_below=args.fail_below,
        generator=generator,
        faithfulness_judge=faithfulness_judge,
        relevance_judge=relevance_judge,
    )
    report = asyncio.run(runner.evaluate(args.dataset))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        report["diff"] = compare_reports(baseline, report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "retrieval": report["retrieval"],
        "generation": report["generation"],
        "failed_cases": report["failed_cases"],
        "diff": report.get("diff"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    print(f"unknown command: {args.command}", file=sys.stderr)
    return 2
