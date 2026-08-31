"""API request/response models."""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096, description="用户问题")
    history: list[ChatMessage] = Field(default_factory=list, description="对话历史")


class CitationModel(BaseModel):
    chunk_id: str
    page: int | None = None
    text: str = ""
    score: float = 0.0


class ChatData(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)


class ChatResponse(BaseModel):
    code: int = 0
    data: ChatData
    trace_id: str
    cost: float = 0.0


class DocumentSubmitResponse(BaseModel):
    code: int = 0
    data: dict[str, object]


class ErrorResponse(BaseModel):
    code: str
    message: str
    trace_id: str
