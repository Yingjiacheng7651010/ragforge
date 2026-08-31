"""Evaluation: retrieval metrics, LLM-as-judge generation metrics, reports and diffing."""

from ragforge.eval.compare import compare_reports
from ragforge.eval.judge import AnswerRelevanceJudge, FaithfulnessJudge, Judge
from ragforge.eval.metrics import hit_rate, mrr, precision_at_k, recall_at_k
from ragforge.eval.runner import EvaluationRunner, SampleResult, load_golden

__all__ = [
    "AnswerRelevanceJudge",
    "EvaluationRunner",
    "FaithfulnessJudge",
    "Judge",
    "SampleResult",
    "compare_reports",
    "hit_rate",
    "load_golden",
    "mrr",
    "precision_at_k",
    "recall_at_k",
]
