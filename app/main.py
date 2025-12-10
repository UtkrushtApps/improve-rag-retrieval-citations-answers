"""FastAPI entrypoint for the Utkrusht RAG assistant.

This module wires together:
- configuration
- logging
- HTTP routes
- RAG service and ChromaDB repository

The application is designed to run inside Docker and talk to a ChromaDB
server over HTTP. Retrieval is tuned to pull richer context and the
responses include explicit, traceable citations to the underlying
chunks used to answer the question.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import Settings, get_settings
from app.core.logging_config import configure_logging
from app.services.chroma_repository import ChromaRepository, get_chroma_repository
from app.services.exceptions import ChromaUnavailableError


# Configure root logger before creating the app
configure_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS configuration – permissive for internal tool usage
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware for request IDs and basic timing/observability
    @app.middleware("http")
    async def add_request_context(
        request: Request, call_next: Callable
    ):  # type: ignore[override]
        start = time.monotonic()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Attach to state so handlers can access it
        request.state.request_id = request_id

        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Unhandled exception during request",
                extra={"request_id": request_id},
            )
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )

        # Propagate request ID back to client for easier tracing
        response.headers["X-Request-ID"] = request_id
        return response

    # Error handler for Chroma unavailability
    @app.exception_handler(ChromaUnavailableError)
    async def chroma_unavailable_handler(
        request: Request, exc: ChromaUnavailableError
    ) -> JSONResponse:
        logger.error(
            "ChromaUnavailableError raised",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "detail": exc.message,
            },
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Vector store is currently unavailable. Please try again later.",
                "reason": exc.message,
            },
        )

    # Health check endpoint that verifies Chroma connectivity
    @app.get("/health", tags=["system"])
    async def health(
        settings: Settings = Depends(get_settings),
        chroma_repo: ChromaRepository = Depends(get_chroma_repository),
    ) -> dict:
        try:
            heartbeat = chroma_repo.healthcheck()
        except ChromaUnavailableError as exc:
            # Map repository-level error to a failing health check
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "status": "ok",
            "app_version": settings.app_version,
            "chroma_heartbeat": heartbeat,
        }

    # Main RAG endpoints
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()
