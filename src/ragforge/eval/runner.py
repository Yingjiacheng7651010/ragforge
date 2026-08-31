"""Evaluation runner: golden-set evaluation with a report per docs/05-评估体系.md.

Sample flow: retrieve (or use pre-retrieved ids for offline runs) ->
compute retrieval metrics -> optionally generate an answer and judge it.
The report contains overall metrics, per-difficulty breakdown and failed
case ids.
"""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforge.core.vector_store import SearchHit
from ragforge.eval.judge import AnswerRelevanceJudge, FaithfulnessJudge
from ragforge.eval.metrics import hit_rate, mrr, precision_at_k, recall_at_k
from ragforge.generation import Generator
from ragforge.retrieval import Retriever

#: golden record field names per docs/05-评估体系.md
QUESTION_KEY = "question"
GOLD_CHUNKS_KEY = "gold_chunks"
DIFFICULTY_KEY = "difficulty"
RETRIEVED_KEY = "retrieved_chunk_ids"
_UNKNOWN_DIFFICULTY = "unknown"

_RETRIEVAL_METRICS = ("recall", "precision", "mrr", "hit_rate")


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL golden dataset (blank lines ignored, BOM tolerated)."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def parse_metric_name(name: str, default_k: int) -> tuple[str, int]:
    """Split 'recall@5' into ('recall', 5); bare names use ``default_k``."""
    if "@" in name:
        base, _, k_part = name.partition("@")
        return base, int(k_part)
    return name, default_k


def compute_retrieval_metrics(
    gold_chunks: Sequence[str],
    retrieved: Sequence[str],
    k: int,
    metric_names: Sequence[str],
) -> dict[str, float]:
    """Compute the requested retrieval metrics for one sample."""
    relevant = set(gold_chunks)
    result: dict[str, float] = {}
    for name in metric_names:
        base, metric_k = parse_metric_name(name, k)
        if base == "recall":
            result[f"recall@{metric_k}"] = recall_at_k(relevant, retrieved, metric_k)
        elif base == "precision":
            result[f"precision@{metric_k}"] = precision_at_k(relevant, retrieved, metric_k)
        elif base == "mrr":
            result["mrr"] = mrr(relevant, retrieved)
        elif base == "hit_rate":
            result["hit_rate"] = hit_rate(relevant, retrieved, metric_k)
    return result


@dataclass
class SampleResult:
    """Per-sample evaluation outcome."""

    sample_id: str
    difficulty: str
    retrieval: dict[str, float]
    generation: dict[str, float] = field(default_factory=dict)
    retrieved: list[str] = field(default_factory=list)
    gold: list[str] = field(default_factory=list)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


class EvaluationRunner:
    """Run retrieval (and optionally generation) evaluation over a golden set."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        k: int = 5,
        recall_top_k: int = 50,
        metric_names: Sequence[str] = ("recall", "mrr", "hit_rate"),
        fail_below: float = 0.0,
        generator: Generator | None = None,
        faithfulness_judge: FaithfulnessJudge | None = None,
        relevance_judge: AnswerRelevanceJudge | None = None,
        max_failures: int = 20,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self._retriever = retriever
        self._k = k
        self._recall_top_k = recall_top_k
        self._metric_names = list(metric_names)
        self._fail_below = fail_below
        self._generator = generator
        self._faithfulness_judge = faithfulness_judge
        self._relevance_judge = relevance_judge
        self._max_failures = max_failures

    async def evaluate(self, dataset_path: str | Path) -> dict[str, Any]:
        """Evaluate the dataset and return the report dict (docs/05 format)."""
        records = load_golden(dataset_path)
        results = [await self._evaluate_record(record) for record in records]
        return self._build_report(dataset_path, results)

    async def _evaluate_record(self, record: Mapping[str, Any]) -> SampleResult:
        question = str(record[QUESTION_KEY])
        gold_chunks = [str(item) for item in record.get(GOLD_CHUNKS_KEY, [])]
        sample_id = str(record.get("id", question))
        difficulty = str(record.get(DIFFICULTY_KEY, _UNKNOWN_DIFFICULTY))

        retrieved_ids: list[str]
        hits: list[SearchHit] = []
        if self._retriever is not None:
            hits = await self._retriever.retrieve(question, self._recall_top_k)
            retrieved_ids = [hit.chunk_id for hit in hits]
        else:
            # offline mode: use pre-retrieved ids from the dataset
            retrieved_ids = [str(item) for item in record.get(RETRIEVED_KEY, [])]

        retrieval = compute_retrieval_metrics(
            gold_chunks, retrieved_ids, self._k, self._metric_names
        )

        generation: dict[str, float] = {}
        if self._generator is not None:
            result = await self._generator.generate(question, hits)
            context = "\n\n".join(
                f"[{index + 1}] {hit.chunk.text if hit.chunk else ''}"
                for index, hit in enumerate(hits[: self._k])
            )
            if self._faithfulness_judge is not None:
                generation["faithfulness"] = await self._faithfulness_judge.judge(
                    question=question,
                    context=context,
                    answer=result.answer,
                )
            if self._relevance_judge is not None:
                generation["answer_relevance"] = await self._relevance_judge.judge(
                    question=question,
                    answer=result.answer,
                )

        return SampleResult(
            sample_id=sample_id,
            difficulty=difficulty,
            retrieval=retrieval,
            generation=generation,
            retrieved=retrieved_ids,
            gold=gold_chunks,
        )

    def _build_report(
        self,
        dataset_path: str | Path,
        results: Sequence[SampleResult],
    ) -> dict[str, Any]:
        metric_keys = list(results[0].retrieval) if results else []
        retrieval_metrics = {
            key: _mean([sample.retrieval[key] for sample in results]) for key in metric_keys
        }
        generation_metrics: dict[str, float] = {}
        if any(sample.generation for sample in results):
            for key in ("faithfulness", "answer_relevance"):
                values = [
                    sample.generation[key]
                    for sample in results
                    if key in sample.generation
                ]
                if values:
                    generation_metrics[key] = _mean(values)

        by_difficulty: dict[str, dict[str, Any]] = {}
        for difficulty in sorted({sample.difficulty for sample in results}):
            group = [sample for sample in results if sample.difficulty == difficulty]
            by_difficulty[difficulty] = {
                "samples": len(group),
                "retrieval": {
                    key: _mean([sample.retrieval[key] for sample in group])
                    for key in metric_keys
                },
                "generation": {
                    key: _mean(
                        [
                            sample.generation[key]
                            for sample in group
                            if key in sample.generation
                        ]
                    )
                    for key in generation_metrics
                },
            }

        primary = (
            f"recall@{self._k}"
            if f"recall@{self._k}" in metric_keys
            else (metric_keys[0] if metric_keys else f"recall@{self._k}")
        )
        failed = [
            sample.sample_id
            for sample in results
            if sample.retrieval.get(primary, 0.0) <= self._fail_below
        ][: self._max_failures]

        return {
            "dataset": str(dataset_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "retriever": self._retriever.__class__.__name__ if self._retriever else "offline",
            "k": self._k,
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "by_difficulty": by_difficulty,
            "failed_cases": failed,
        }
