"""Celery: async document ingestion worker tasks."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from celery import Celery

from ragforge.api.services import AppServices, build_services
from ragforge.ingestion import DEFAULT_REGISTRY, Chunker, StructureChunker

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "ragforge",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
)


class CeleryIngestor:
    """Submit documents to the Celery worker and query their status."""

    def submit(self, doc_id: str, filename: str, content: bytes) -> None:
        celery_app.send_task(
            "ragforge.ingest_document",
            args=[doc_id, filename, content],
        )

    async def status(self, doc_id: str) -> tuple[str, dict[str, Any]]:
        result = celery_app.AsyncResult(doc_id)
        state, info = await asyncio.to_thread(self._read, result)
        return state, info

    @staticmethod
    def _read(result: Any) -> tuple[str, dict[str, Any]]:
        info = result.info or {}
        return str(result.state), dict(info) if isinstance(info, dict) else {"result": info}


@celery_app.task(name="ragforge.ingest_document")
def ingest_document(doc_id: str, filename: str, content: bytes) -> dict[str, Any]:
    """Parse, chunk and index one uploaded document (runs in the worker)."""
    return asyncio.run(_ingest_document(doc_id, filename, content))


async def _ingest_document(doc_id: str, filename: str, content: bytes) -> dict[str, Any]:
    # Build the pipeline lazily inside the worker process.
    services: AppServices = build_services()
    temp_path: Path | None = None
    try:
        suffix = Path(filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        parsed = DEFAULT_REGISTRY.parse(temp_path)
        chunker: Chunker = StructureChunker(max_tokens=512)
        chunks = chunker.split(parsed)
        if services.es_store is None:
            raise RuntimeError("elasticsearch store not configured")
        await services.es_store.add(chunks)
        return {"chunks": len(chunks), "status": "indexed"}
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
