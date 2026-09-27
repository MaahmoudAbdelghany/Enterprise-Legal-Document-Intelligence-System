"""LexisGraph — Enterprise Legal Document Intelligence System FastAPI application entry point.

Initializes the FastAPI application with:
- Lifespan context manager for resource initialization and graceful teardown
- CORS middleware for multi-origin API access
- Request timing and tracing middleware (X-Request-ID and X-Process-Time-Ms)
- Structured logging configuration
- Health, document management, and RAG query route registration
- Centralized exception handlers for standardized error responses
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import sys
import time
from typing import Any, AsyncIterator
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_qdrant_store,
    reset_dependencies,
)
from app.api.routes import documents_router, health_router, query_router
from app.config import Settings, get_settings
from app.models.schemas import HTTPErrorResponse

# Application metadata constants
APP_TITLE = "LexisGraph — Legal Document Intelligence System"
APP_DESCRIPTION = (
    "Enterprise Arabic Legal Document Intelligence System with ColPali OCR-Free "
    "Visual Retrieval, Qdrant Multi-Vector Store, and Neo4j Knowledge Graph."
)
APP_VERSION = "0.1.0"


# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

def configure_logging(settings: Settings) -> None:
    """Configure Loguru structured logging based on application settings."""
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=log_format,
        colorize=True,
        backtrace=settings.debug,
        diagnose=settings.debug,
    )


# ------------------------------------------------------------------------------
# Lifespan Context Manager
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycles."""
    settings = get_settings()
    configure_logging(settings)

    logger.info(
        f"Starting {settings.app_name} v{APP_VERSION} "
        f"[env={settings.app_env}, debug={settings.debug}]"
    )

    # 1. Ensure local storage directories exist
    try:
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage directory verified at '{settings.storage_path.resolve()}'")
    except Exception as exc:
        logger.warning(f"Could not create local storage directory '{settings.local_storage_dir}': {exc}")

    # 2. Probe Qdrant Store (non-blocking warm-up)
    try:
        qdrant = get_qdrant_store()
        logger.info(
            f"Qdrant connection ready (collection='{qdrant.collection_name}', "
            f"host='{settings.qdrant_host}:{settings.qdrant_port}')"
        )
    except Exception as exc:
        logger.warning(
            f"Qdrant vector store could not be reached during startup: {exc}. "
            f"The application will start in degraded mode."
        )

    # 3. Log ColPali Embedder configuration
    try:
        embedder = get_colpali_embedder()
        logger.info(
            f"ColPali embedder configured (model='{embedder.model_name}', "
            f"device='{embedder.device}', batch_size={embedder.batch_size})"
        )
    except Exception as exc:
        logger.warning(f"ColPali embedder initialization warning: {exc}")

    logger.info("Application startup lifecycle complete. Ready to receive requests.")

    yield

    # Teardown logic
    logger.info("Initiating application shutdown...")
    reset_dependencies()
    logger.info("All shared application dependencies closed. Shutdown complete.")


# ------------------------------------------------------------------------------
# Application Factory
# ------------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional custom Settings instance (useful for testing).

    Returns:
        Configured FastAPI application.
    """
    app_settings = settings or get_settings()

    fastapi_app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --------------------------------------------------------------------------
    # Middleware Setup
    # --------------------------------------------------------------------------

    # 1. CORS Middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Request Timing & Correlation ID Middleware
    @fastapi_app.middleware("http")
    async def request_timing_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Inject request_id into state for downstream handlers
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            # Let exception handlers capture and format the response
            raise

        process_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(process_time_ms)

        # Log request summary
        logger.debug(
            f"{request.method} {request.url.path} "
            f"-> status={response.status_code} "
            f"time={process_time_ms}ms "
            f"[request_id={request_id}]"
        )
        return response

    # --------------------------------------------------------------------------
    # Exception Handlers
    # --------------------------------------------------------------------------

    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error_detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        )
        logger.warning(f"Validation error on {request.url.path}: {error_detail}")
        err_response = HTTPErrorResponse(
            error="VALIDATION_ERROR",
            message=error_detail,
            details={"errors": [dict(err) for err in exc.errors()]},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=err_response.model_dump(mode="json"),
        )

    @fastapi_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        logger.warning(f"HTTPException on {request.url.path} [{exc.status_code}]: {exc.detail}")
        err_response = HTTPErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=err_response.model_dump(mode="json"),
            headers=getattr(exc, "headers", None),
        )

    @fastapi_app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
        err_response = HTTPErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred while processing the request.",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err_response.model_dump(mode="json"),
        )

    # --------------------------------------------------------------------------
    # Route Registration
    # --------------------------------------------------------------------------

    # Root service metadata endpoint
    @fastapi_app.get(
        "/",
        tags=["Root"],
        summary="Service Root & Metadata",
        description="Returns system metadata, documentation links, and operational status.",
    )
    async def root_endpoint() -> dict[str, Any]:
        return {
            "name": app_settings.app_name,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION,
            "environment": app_settings.app_env,
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json",
            "api_v1_prefix": app_settings.api_v1_prefix,
            "status": "online",
        }

    # Register Health routes (both at root and under api_v1_prefix)
    fastapi_app.include_router(health_router)
    fastapi_app.include_router(health_router, prefix=app_settings.api_v1_prefix)

    # Register API v1 functional route handlers
    fastapi_app.include_router(documents_router, prefix=app_settings.api_v1_prefix)
    fastapi_app.include_router(query_router, prefix=app_settings.api_v1_prefix)

    return fastapi_app


# ------------------------------------------------------------------------------
# Module-level Application Instance
# ------------------------------------------------------------------------------

app: FastAPI = create_app()


# ------------------------------------------------------------------------------
# CLI Server Runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    current_settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=current_settings.host,
        port=current_settings.port,
        reload=current_settings.debug,
        log_level=current_settings.log_level.lower(),
    )
