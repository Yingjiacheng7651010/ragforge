"""Unit tests for the agentic RAG engine: planning, execution, reflection loops."""

import pytest

from ragforge.agent import AgenticRagEngine, Executor, Planner, Reflector, Step, safe_calc
from ragforge.core.llm import LLMResult
from ragforge.core.vector_store import SearchHit
from ragforge.ingestion import Chunk
from tests.unit.fakes import FakeLLM, FakeRetriever


def chunked_hit(chunk_id: str, text: str, score: float = 0.9) -> SearchHit:
    chunk = Chunk(chunk_id=chunk_id, doc_id="doc", text=text)
    return SearchHit(chunk_id=chunk_id, score=score, chunk=chunk)


# --- safe calculator ---


def test_safe_calc_evaluates_expressions() -> None:
    assert safe_calc("1500 * 0.08") == pytest.approx(120.0)
    assert safe_calc("(2 + 3) * 4") == pytest.approx(20.0)
    assert safe_calc("sqrt(16)") == pytest.approx(4.0)
    assert safe_calc("10 / 4") == pytest.approx(2.5)


@pytest.mark.parametrize("evil", ["__import__('os')", "open('/etc/passwd')", "1; print(1)"])
def test_safe_calc_rejects_arbitrary_code(evil: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_calc(evil)


# --- planner ---


async def test_planner_parses_steps() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text='{"steps": [{"id": "1", "tool": "retrieve", "query": "公司A年收入"}, '
            '{"id": "2", "tool": "calculator", "query": "1500 * 0.08"}]}'
        )
    )
    planner = Planner(llm)

    steps = await planner.plan("A 的利润是多少？")

    assert [(step.tool, step.query) for step in steps] == [
        ("retrieve", "公司A年收入"),
        ("calculator", "1500 * 0.08"),
    ]


async def test_planner_falls_back_to_single_retrieve_on_invalid_json() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="无法规划"))
    planner = Planner(llm)

    steps = await planner.plan("q")

    assert steps == [Step(id="1", tool="retrieve", query="q")]


async def test_planner_filters_unknown_tools_and_duplicate_ids() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text='{"steps": [{"id": "1", "tool": "retrieve", "query": "a"}, '
            '{"id": "1", "tool": "calculator", "query": "1+1"}, '
            '{"id": "2", "tool": "hack", "query": "x"}, '
            '{"id": "3", "tool": "lookup_table", "query": "t"}]}'
        )
    )
    planner = Planner(llm)

    steps = await planner.plan("q")

    assert [(step.id, step.tool) for step in steps] == [("1", "retrieve"), ("3", "lookup_table")]


async def test_planner_caps_step_count() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(
            text='{"steps": [{"id": "1", "tool": "retrieve", "query": "a"}, '
            '{"id": "2", "tool": "retrieve", "query": "b"}, '
            '{"id": "3", "tool": "retrieve", "query": "c"}]}'
        )
    )
    planner = Planner(llm, max_steps=2)

    steps = await planner.plan("q")

    assert len(steps) == 2


# --- executor ---


async def test_executor_runs_all_tools() -> None:
    retriever = FakeRetriever([chunked_hit("c1", "公司A年收入1500万")])
    executor = Executor(
        retriever=retriever,
        tables={"companies": [["name", "revenue"], ["A", "1500"], ["B", "100"]]},
    )
    steps = [
        Step(id="1", tool="retrieve", query="公司A收入"),
        Step(id="2", tool="calculator", query="1500 * 0.08"),
        Step(id="3", tool="lookup_table", query="companies"),
    ]

    results = await executor.execute(steps)

    assert "公司A年收入1500万" in results["1"]
    assert results["2"] == "120"
    assert "A" in results["3"] and "1500" in results["3"]


async def test_executor_tolerates_tool_errors() -> None:
    executor = Executor(retriever=FakeRetriever())

    results = await executor.execute([Step(id="1", tool="calculator", query="__import__('os')")])

    assert "calc error" in results["1"]


async def test_executor_reports_missing_table() -> None:
    executor = Executor(retriever=FakeRetriever())

    results = await executor.execute([Step(id="1", tool="lookup_table", query="nope")])

    assert "table not found" in results["1"]


# --- reflector ---


async def test_reflector_ok() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "ok", "feedback": ""}'))
    reflector = Reflector(llm)

    verdict, feedback = await reflector.reflect("q", "ctx", "ans")

    assert verdict == "ok"
    assert feedback == ""


async def test_reflector_revise_with_feedback() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"verdict": "revise", "feedback": "缺少利润数据"}'))
    reflector = Reflector(llm)

    verdict, feedback = await reflector.reflect("q", "ctx", "ans")

    assert verdict == "revise"
    assert feedback == "缺少利润数据"


async def test_reflector_fails_closed_to_revise() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="无法判断"))
    reflector = Reflector(llm)

    verdict, _ = await reflector.reflect("q", "ctx", "ans")

    assert verdict == "revise"


# --- engine ---


def make_engine(
    *,
    plan_responses: list[str],
    answers: list[str],
    verdicts: list[str],
    max_rounds: int = 2,
) -> tuple[AgenticRagEngine, dict[str, FakeLLM]]:
    plan_llm = FakeLLM()
    for response in plan_responses:
        plan_llm.enqueue_completion(LLMResult(text=response))
    gen_llm = FakeLLM()
    for answer in answers:
        gen_llm.enqueue_completion(LLMResult(text=answer))
    reflect_llm = FakeLLM()
    for verdict in verdicts:
        reflect_llm.enqueue_completion(LLMResult(text=verdict))

    engine = AgenticRagEngine(
        planner=Planner(plan_llm),
        executor=Executor(
            retriever=FakeRetriever([chunked_hit("c1", "公司A年收入1500万，利润率8%")])
        ),
        reflector=Reflector(reflect_llm),
        llm=gen_llm,
        max_rounds=max_rounds,
    )
    return engine, {"plan": plan_llm, "gen": gen_llm, "reflect": reflect_llm}


async def test_engine_returns_answer_when_reflection_passes() -> None:
    engine, llms = make_engine(
        plan_responses=['{"steps": [{"id": "1", "tool": "retrieve", "query": "A收入"}]}'],
        answers=["A 年收入 1500 万。"],
        verdicts=['{"verdict": "ok", "feedback": ""}'],
    )

    answer = await engine.run("A 的收入是多少？")

    assert answer == "A 年收入 1500 万。"
    assert len(llms["plan"].complete_calls) == 1  # single round, no retry


async def test_engine_revises_then_succeeds_with_feedback() -> None:
    engine, llms = make_engine(
        plan_responses=[
            '{"steps": [{"id": "1", "tool": "retrieve", "query": "A收入"}]}',
            '{"steps": [{"id": "1", "tool": "calculator", "query": "1500 * 0.08"}]}',
        ],
        answers=["答案一", "答案二"],
        verdicts=[
            '{"verdict": "revise", "feedback": "缺少利润计算"}',
            '{"verdict": "ok", "feedback": ""}',
        ],
    )

    answer = await engine.run("A 的利润是多少？")

    assert answer == "答案二"
    assert len(llms["plan"].complete_calls) == 2
    # the second plan prompt carries the reflector feedback
    second_plan_prompt = llms["plan"].complete_calls[1][0][-1].content
    assert "缺少利润计算" in second_plan_prompt


async def test_engine_stops_after_max_rounds() -> None:
    engine, llms = make_engine(
        plan_responses=[
            '{"steps": [{"id": "1", "tool": "retrieve", "query": "a"}]}',
            '{"steps": [{"id": "1", "tool": "retrieve", "query": "b"}]}',
            '{"steps": [{"id": "1", "tool": "retrieve", "query": "c"}]}',
        ],
        answers=["r1", "r2", "r3"],
        verdicts=[
            '{"verdict": "revise", "feedback": "f1"}',
            '{"verdict": "revise", "feedback": "f2"}',
        ],
        max_rounds=2,
    )

    answer = await engine.run("q")

    assert answer == "r3"  # rounds exhausted, last answer returned
    assert len(llms["plan"].complete_calls) == 3  # bounded: no infinite loop


def test_engine_rejects_loop_config() -> None:
    with pytest.raises(ValueError):
        AgenticRagEngine(
            planner=Planner(FakeLLM()),
            executor=Executor(retriever=FakeRetriever()),
            reflector=Reflector(FakeLLM()),
            llm=FakeLLM(),
            max_rounds=3,
        )
