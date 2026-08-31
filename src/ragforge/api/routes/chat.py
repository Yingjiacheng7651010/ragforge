"""Chat routes: non-streaming answer and SSE streaming."""

import json
from collections.abc import AsyncIterator, Sequence
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ragforge.api.app import current_trace_id
from ragforge.api.schemas import ChatData, ChatMessage, ChatRequest, ChatResponse, CitationModel
from ragforge.api.services import AppServices
from ragforge.core.llm import Message
from ragforge.core.vector_store import SearchHit
from ragforge.generation import GenerationResult
from ragforge.observability import get_metrics

router = APIRouter(tags=["chat"])


def _to_messages(history: list[ChatMessage]) -> list[Message]:
    return [Message(role=item.role, content=item.content) for item in history]


def _citation_models(result: GenerationResult) -> list[CitationModel]:
    return [
        CitationModel(
            chunk_id=citation.chunk_id,
            page=citation.page,
            text=citation.text,
            score=citation.score,
        )
        for citation in result.citations
    ]


def _context_text(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(
        f"[{index + 1}] {hit.chunk.text if hit.chunk else ''}"
        for index, hit in enumerate(hits[:8])
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    services: AppServices = req.app.state.services
    await services.input_guard.enforce(user_input=request.query)

    cached = await services.cache.get(request.query)
    if cached is not None:
        return ChatResponse(
            code=0,
            data=ChatData(
                answer=cached.answer,
                citations=[
                    CitationModel(
                        chunk_id=citation.chunk_id,
                        page=citation.page,
                        text=citation.text,
                        score=citation.score,
                    )
                    for citation in cached.citations
                ],
            ),
            trace_id=current_trace_id(),
            cost=0.0,
        )

    understanding = await services.understanding.understand(
        request.query, _to_messages(request.history)
    )
    question = understanding.rewritten_query or request.query
    hits = await services.pipeline.retrieve(question, 50)
    result = await services.generator.generate(question, hits)

    await services.output_guard.enforce(
        context=_context_text(hits),
        answer=result.answer,
    )
    await services.cache.set(request.query, result.answer, result.citations)
    get_metrics().record_cost(result.cost)

    return ChatResponse(
        code=0,
        data=ChatData(answer=result.answer, citations=_citation_models(result)),
        trace_id=current_trace_id(),
        cost=result.cost,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request) -> StreamingResponse:
    services: AppServices = req.app.state.services
    await services.input_guard.enforce(user_input=request.query)

    async def event_source() -> AsyncIterator[str]:
        cached = await services.cache.get(request.query)
        if cached is not None:
            payload = {
                "type": "answer",
                "answer": cached.answer,
                "citations": [
                    {
                        "chunk_id": citation.chunk_id,
                        "page": citation.page,
                        "text": citation.text,
                        "score": citation.score,
                    }
                    for citation in cached.citations
                ],
                "cost": 0.0,
                "trace_id": current_trace_id(),
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        understanding = await services.understanding.understand(
            request.query, _to_messages(request.history)
        )
        question = understanding.rewritten_query or request.query
        hits = await services.pipeline.retrieve(question, 50)

        result: GenerationResult | None = None
        async for kind, value in services.generator.stream_answer(question, hits):
            if kind == "token":
                payload = {"type": "token", "content": value}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            else:
                result = cast(GenerationResult, value)
        if result is None:
            raise RuntimeError("stream ended without a result")

        await services.output_guard.enforce(context=_context_text(hits), answer=result.answer)
        await services.cache.set(request.query, result.answer, result.citations)
        final = {
            "type": "answer",
            "answer": result.answer,
            "citations": [
                {
                    "chunk_id": citation.chunk_id,
                    "page": citation.page,
                    "text": citation.text,
                    "score": citation.score,
                }
                for citation in result.citations
            ],
            "cost": result.cost,
            "trace_id": current_trace_id(),
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
