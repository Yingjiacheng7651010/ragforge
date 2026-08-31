"""Unit tests for the unified LLM base interface and structured output repair."""

import dataclasses

import pytest

from ragforge.core.errors import RAGForgeError
from ragforge.core.llm import LLMResult, Message
from ragforge.core.llm.base import _extract_json_object
from tests.unit.fakes import FakeLLM


def test_message_fields_and_to_dict() -> None:
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"
    assert message.to_dict() == {"role": "user", "content": "hello"}


def test_message_is_frozen() -> None:
    message = Message(role="user", content="hello")

    with pytest.raises(dataclasses.FrozenInstanceError):
        message.content = "mutated"  # type: ignore[misc]


def test_llm_result_defaults() -> None:
    result = LLMResult(text="hi")

    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.cost == 0.0
    assert result.latency_ms == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("not json at all", None),
        ("[]", None),
        ('{"broken": }', None),
        ("", None),
        (
            "Sure! Here is the answer:\n{\"nested\": {\"x\": [1, 2]}}\nHope this helps.",
            {"nested": {"x": [1, 2]}},
        ),
    ],
)
def test_extract_json_object(text: str, expected: dict[str, object] | None) -> None:
    assert _extract_json_object(text) == expected


async def test_complete_structured_success() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text='{"answer": 42}'))

    result = await llm.complete_structured(
        [Message(role="user", content="what is the answer?")],
        {"type": "object", "properties": {"answer": {"type": "integer"}}},
    )

    assert result == {"answer": 42}
    # The schema must be injected as a system instruction.
    system_msg = llm.complete_calls[0][0][-1]
    assert system_msg.role == "system"
    assert '"type": "object"' in system_msg.content


async def test_complete_structured_repairs_invalid_json() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(
        LLMResult(text="I'm sorry, here is my answer: [oops]"),
        LLMResult(text='{"ok": true}'),
    )

    result = await llm.complete_structured([Message(role="user", content="hi")], {"type": "object"})

    assert result == {"ok": True}
    assert len(llm.complete_calls) == 2
    # The repair pass appends a user message asking for JSON only.
    repair_msg = llm.complete_calls[1][0][-1]
    assert repair_msg.role == "user"
    assert "valid JSON" in repair_msg.content


async def test_complete_structured_exhausts_repairs() -> None:
    llm = FakeLLM()
    llm.enqueue_completion(LLMResult(text="nope"), LLMResult(text="still nope"))

    with pytest.raises(RAGForgeError) as exc_info:
        await llm.complete_structured([Message(role="user", content="hi")], {"type": "object"})

    assert exc_info.value.code == "E_LLM_JSON_INVALID"
    assert len(llm.complete_calls) == 2
