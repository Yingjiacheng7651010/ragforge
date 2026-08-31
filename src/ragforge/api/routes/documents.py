"""Document routes: async ingestion via Celery."""

from fastapi import APIRouter, Request, UploadFile

from ragforge.api.app import current_trace_id
from ragforge.api.schemas import DocumentSubmitResponse
from ragforge.api.services import AppServices, new_doc_id

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentSubmitResponse)
async def upload_document(file: UploadFile, req: Request) -> DocumentSubmitResponse:
    services: AppServices = req.app.state.services
    content = await file.read()  # async read, never blocking
    doc_id = new_doc_id()
    services.ingestor.submit(doc_id, file.filename or "unnamed", content)
    return DocumentSubmitResponse(
        code=0,
        data={"doc_id": doc_id, "trace_id": current_trace_id()},
    )


@router.get("/documents/{doc_id}", response_model=DocumentSubmitResponse)
async def document_status(doc_id: str, req: Request) -> DocumentSubmitResponse:
    services: AppServices = req.app.state.services
    state, info = await services.ingestor.status(doc_id)
    return DocumentSubmitResponse(
        code=0,
        data={"doc_id": doc_id, "status": state, "info": info},
    )
