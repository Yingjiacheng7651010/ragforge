"""System routes: health and Prometheus metrics."""

from fastapi import APIRouter, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ragforge.api.services import AppServices

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(req: Request) -> dict[str, object]:
    services: AppServices = req.app.state.services
    checks = await services.check_health()
    status = "ok" if all(value == "ok" for value in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
