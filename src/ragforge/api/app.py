"""FastAPI application factory: middleware, exception handlers, routing."""

import uuid
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace

from ragforge.api.limits import TokenBucketLimiter
from ragforge.api.services import AppServices, build_services
from ragforge.core.errors import RAGForgeError
from ragforge.observability import get_logger, get_request_id, set_request_id

logger = get_logger(__name__)

#: user-facing status codes for known error codes
_FRIENDLY_STATUS = {
    "E_GUARD_BLOCKED": (400, "输入未通过安全检查，已拦截。"),
    "E_UNSUPPORTED_FORMAT": (400, "不支持的文件格式。"),
    "E_VALIDATION": (422, "请求参数校验失败。"),
}

_DEFAULT_MESSAGE = "服务处理失败，请稍后重试。"


def current_trace_id() -> str:
    """The active OTel trace id, falling back to the request id."""
    span_context = trace.get_current_span().get_span_context()
    if span_context.is_valid:
        return format(span_context.trace_id, "032x")
    return get_request_id() or ""


def get_services(request: Request) -> AppServices:
    return cast(AppServices, request.app.state.services)


def create_app(
    services: AppServices | None = None,
    *,
    rate_limit_capacity: float = 10.0,
    rate_limit_refill: float = 1.0,
) -> FastAPI:
    """Build the API app; ``services`` is injectable for tests and demos."""
    services = services or build_services()
    app = FastAPI(title="ragforge API", version="0.1.0")
    app.state.services = services

    limiter = TokenBucketLimiter(capacity=rate_limit_capacity, refill_rate=rate_limit_refill)

    @app.middleware("http")
    async def request_context(request: Request, call_next: object) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        set_request_id(request_id)
        limited = request.method == "POST" and (
            request.url.path.startswith("/v1/chat") or request.url.path.startswith("/v1/documents")
        )
        if limited and not await limiter.acquire():
            return JSONResponse(
                status_code=429,
                content={
                    "code": "E_RATE_LIMITED",
                    "message": "请求过于频繁，请稍后再试。",
                    "trace_id": current_trace_id(),
                },
            )
        response = cast(JSONResponse, await call_next(request))  # type: ignore[operator]
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RAGForgeError)
    async def ragforge_error_handler(request: Request, exc: RAGForgeError) -> JSONResponse:
        status, message = _FRIENDLY_STATUS.get(exc.code, (500, _DEFAULT_MESSAGE))
        logger.warning(
            "ragforge error",
            code=exc.code,
            message=exc.message,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status,
            content={"code": exc.code, "message": message, "trace_id": current_trace_id()},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "E_VALIDATION",
                "message": "请求参数校验失败。",
                "trace_id": current_trace_id(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "code": "E_INTERNAL",
                "message": _DEFAULT_MESSAGE,
                "trace_id": current_trace_id(),
            },
        )

    from ragforge.api.routes import chat, documents, system

    app.include_router(chat.router, prefix="/v1")
    app.include_router(documents.router, prefix="/v1")
    app.include_router(system.router, prefix="/v1")
    return app


__all__ = ["create_app"]
