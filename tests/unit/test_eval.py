"""Unit tests for evaluation: metrics, runner reports, judges, diff and the CLI."""

import json
import pathlib
from types import SimpleNamespace

import pytest

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import LLMResult
from ragforge.eval import (
    AnswerRelevanceJudge,
    EvaluationRunner,
    FaithfulnessJudge,
    compare_reports,
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
)
from ragforge.eval.cli import main, parse_metrics
from ragforge.generation import Generator
from tests.unit.fakes import FakeLLM


def golden_line(
    sample_id: str,
    gold_chunks: list[str],
    retrieved: list[str],
    difficulty: str | None = None,
) -> str:
    record = {
        "id": sample_id,
        "question": f"q-{sample_id}",
        "gold_answer": "a",
        "gold_chunks": gold_chunks,
        "retrieved_chunk_ids": retrieved,
    }
    if difficulty:
        record["difficulty"] = difficulty
    return json.dumps(record, ensure_ascii=False)


def write_golden(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- metric primitives ---


def test_recall_at_k() -> None:
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 2) == 0.5  # top-2 hits one of two
    assert recall_at_k({"a", "b"}, ["a", "b"], 5) == 1.0
    assert recall_at_k(set(), ["a"], 5) == 0.0  # no relevant chunks defined
    assert recall_at_k({"a"}, [], 5) == 0.0
    with pytest.raises(ValueError):
        recall_at_k({"a"}, ["a"], -1)


def test_precision_at_k() -> None:
    assert precision_at_k({"a"}, ["a", "b"], 2) == 0.5
    assert precision_at_k({"a"}, ["a", "b"], 1) == 1.0
    assert precision_at_k({"a"}, [], 0) == 0.0
    assert precision_at_k({"a"}, ["b"], 5) == 0.0


def test_mrr() -> None:
    assert mrr({"b"}, ["a", "b", "c"]) == pytest.approx(0.5)
    assert mrr({"a"}, ["a"]) == 1.0
    assert mrr({"z"}, ["a", "b"]) == 0.0


def test_hit_rate() -> None:
    assert hit_rate({"b"}, ["a", "b"], 2) == 1.0
    assert hit_rate({"b"}, ["a", "b"], 1) == 0.0  # b at rank 2, outside k=1
    assert hit_rate({"b"}, ["a"], 5) == 0.0


# --- runner ---


async def test_runner_report_structure(tmp_path: pathlib.Path) -> None:
    golden = tmp_path / "golden.jsonl"
    write_golden(
        golden,
        [
            golden_line("e1", ["c1"], ["c1", "c2"], difficulty="easy"),
            golden_line("e2", ["c9"], ["c1"], difficulty="hard"),
            golden_line("e3", ["c2"], ["c2"]),
        ],
    )
    runner = EvaluationRunner(k=5, metric_names=["recall", "mrr", "hit_rate"])

    report = await runner.evaluate(golden)

    assert report["retriever"] == "offline"
    assert report["retrieval"]["recall@5"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["retrieval"]["mrr"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["retrieval"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["by_difficulty"]["easy"]["samples"] == 1
    assert report["by_difficulty"]["easy"]["retrieval"]["recall@5"] == 1.0
    assert report["by_difficulty"]["hard"]["retrieval"]["recall@5"] == 0.0
    assert report["by_difficulty"]["unknown"]["samples"] == 1  # e3 has no difficulty
    assert report["failed_cases"] == ["e2"]  # recall@5 == 0.0 is a failed case


async def test_runner_with_generation_metrics(tmp_path: pathlib.Path) -> None:
    golden = tmp_path / "golden.jsonl"
    write_golden(golden, [golden_line("g1", ["c1"], ["c1"], difficulty="easy")])
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="生成的答案"))
    llm.enqueue_completion(LLMResult(text='{"score": 0.9, "reason": "ok"}'))
    llm.enqueue_completion(LLMResult(text='{"score": 0.8, "reason": "ok"}'))
    runner = EvaluationRunner(
        k=5,
        metric_names=["recall"],
        generator=Generator(llm=llm, max_context_tokens=1000),
        faithfulness_judge=FaithfulnessJudge(llm),
        relevance_judge=AnswerRelevanceJudge(llm),
    )

    report = await runner.evaluate(golden)

    assert report["generation"]["faithfulness"] == pytest.approx(0.9)
    assert report["generation"]["answer_relevance"] == pytest.approx(0.8)


# --- judges ---


async def test_faithfulness_judge_scores_and_renders_prompt() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"score": 0.8, "reason": "ok"}'))
    judge = FaithfulnessJudge(llm)

    score = await judge.judge(question="q", context="ctx", answer="ans")

    assert score == pytest.approx(0.8)
    prompt = llm.complete_calls[0][0][-1].content
    assert "q" in prompt and "ctx" in prompt and "ans" in prompt


async def test_judge_degrades_on_invalid_json() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="no json"))
    judge = FaithfulnessJudge(llm)

    assert await judge.judge(question="q", context="c", answer="a") == 0.0


async def test_judge_clamps_out_of_range_score() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"score": 1.7}'))
    judge = AnswerRelevanceJudge(llm)

    assert await judge.judge(question="q", answer="a") == 1.0


# --- diff ---


def test_compare_reports() -> None:
    before = {"retrieval": {"recall@5": 0.8, "mrr": 0.7}, "generation": {"faithfulness": 0.9}}
    after = {"retrieval": {"recall@5": 0.9, "mrr": 0.7}, "generation": {}}

    diff = compare_reports(before, after)

    assert diff["recall@5"] == {"before": 0.8, "after": 0.9, "delta": 0.1}
    assert diff["mrr"]["delta"] == 0.0
    assert "faithfulness" not in diff  # absent from the new report


# --- cli ---


def test_parse_metrics() -> None:
    names, all_names = parse_metrics("recall@3,mrr,faithfulness", 5)

    assert names == ["recall@3", "mrr"]
    assert all_names == ["recall@3", "mrr", "faithfulness"]

    with pytest.raises(RAGForgeError):
        parse_metrics("bogus", 5)


def test_build_retriever_offline_rejected() -> None:
    from ragforge.eval import cli

    with pytest.raises(RAGForgeError) as exc_info:
        cli.build_retriever("offline", "chunks")
    assert exc_info.value.code == "E_EVAL_RETRIEVER"


def test_build_retriever_unknown_name_rejected() -> None:
    from ragforge.eval import cli

    with pytest.raises(RAGForgeError) as exc_info:
        cli.build_retriever("bogus", "chunks")
    assert exc_info.value.code == "E_EVAL_RETRIEVER"


def test_build_retriever_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragforge.eval import cli

    monkeypatch.setattr(
        "ragforge.eval.cli.get_settings",
        lambda: SimpleNamespace(llm_api_key=SimpleNamespace(get_secret_value=lambda: "")),
    )

    with pytest.raises(RAGForgeError) as exc_info:
        cli.build_retriever("dense", "chunks")
    assert exc_info.value.code == "E_EVAL_LLM_MISSING"


def test_build_retriever_dense_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragforge.eval import cli
    from ragforge.retrieval import DenseRetriever

    monkeypatch.setattr(
        "ragforge.eval.cli.get_settings",
        lambda: SimpleNamespace(
            llm_api_key=SimpleNamespace(get_secret_value=lambda: "sk-test"),
            embedding_model="e",
            embedding_dim=3,
            llm_model="m",
        ),
    )
    monkeypatch.setenv("ES_URL", "http://127.0.0.1:1")

    retriever = cli.build_retriever("dense", "chunks")

    assert isinstance(retriever, DenseRetriever)


def test_cli_unknown_command_exits() -> None:
    from ragforge.eval.cli import main

    # argparse rejects unknown subcommands with SystemExit before run()
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_main_help_exits_zero() -> None:
    from ragforge.eval.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_offline_run_and_baseline_diff(tmp_path: pathlib.Path) -> None:
    golden = tmp_path / "golden.jsonl"
    write_golden(golden, [golden_line("a", ["c1"], ["c1"], difficulty="easy")])
    out1 = tmp_path / "report1.json"
    rc = main(["run", "--dataset", str(golden), "--metrics", "recall@2,mrr", "--output", str(out1)])
    assert rc == 0
    report1 = json.loads(out1.read_text(encoding="utf-8"))
    assert report1["retrieval"]["recall@2"] == 1.0

    # degrade retrieval, rerun against the baseline
    write_golden(golden, [golden_line("a", ["c1"], ["c9"], difficulty="easy")])
    out2 = tmp_path / "report2.json"
    assert (
        main(
            [
                "run",
                "--dataset",
                str(golden),
                "--metrics",
                "recall@2",
                "--output",
                str(out2),
                "--baseline",
                str(out1),
            ]
        )
        == 0
    )
    report2 = json.loads(out2.read_text(encoding="utf-8"))
    assert report2["retrieval"]["recall@2"] == 0.0
    assert report2["failed_cases"] == ["a"]
    assert report2["diff"]["recall@2"] == {"before": 1.0, "after": 0.0, "delta": -1.0}
