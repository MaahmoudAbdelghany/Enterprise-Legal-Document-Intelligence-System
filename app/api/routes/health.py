"""System health and readiness diagnostic API routes for LexisGraph.

Provides REST endpoints for:
- Liveness probes (`/health`, `/live`) for Kubernetes and container orchestrators
- Readiness probes (`/ready`) performing end-to-end dependency health checks (Qdrant, LLM, Embedder, Neo4j, Storage)
- Fine-grained per-component diagnostics (`/health/components/{component}`)
"""

from __future__ import annotations

import datetime
from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_qdrant_store,
)
from app.config import Settings
from app.core.ingestion.colpali_embedder import ColPaliEmbedder
from app.core.retrieval.qdrant_store import QdrantStore
from app.models.schemas import (
    ComponentHealth,
    ComponentStatus,
    HealthCheckResponse,
    HTTPErrorResponse,
)

router = APIRouter(tags=["Health"])

APP_VERSION = "0.1.0"


# ------------------------------------------------------------------------------
# Component Health Checkers
# ------------------------------------------------------------------------------

def check_qdrant_health(qdrant_store: QdrantStore | None) -> ComponentHealth:
    """Probe Qdrant multi-vector store connectivity and collection status."""
    start_t = time.perf_counter()
    if qdrant_store is None:
        return ComponentHealth(
            status=ComponentStatus.UNHEALTHY,
            details="Qdrant store service dependency is not available",
            latency_ms=0.0,
        )

    try:
        exists = qdrant_store.collection_exists()
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"Collection '{qdrant_store.collection_name}' is active (exists={exists})",
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        logger.warning(f"Qdrant health check failed: {exc}")
        return ComponentHealth(
            status=ComponentStatus.UNHEALTHY,
            details=f"Qdrant connection error: {exc}",
            latency_ms=latency_ms,
        )


def check_embedder_health(embedder: ColPaliEmbedder | None) -> ComponentHealth:
    """Probe ColPali multi-vector embedding engine configuration and state."""
    start_t = time.perf_counter()
    if embedder is None:
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details="ColPali embedder dependency is uninitialized",
            latency_ms=0.0,
        )

    try:
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=(
                f"Model: {embedder.model_name}, device: {embedder.device}, "
                f"dim: {embedder.vector_dim}, loaded: {embedder.is_loaded}"
            ),
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        logger.warning(f"Embedder health check error: {exc}")
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details=f"ColPali embedder check error: {exc}",
            latency_ms=latency_ms,
        )


def check_llm_health(settings: Settings) -> ComponentHealth:
    """Verify primary and fallback LLM API credentials and model configurations."""
    start_t = time.perf_counter()
    has_groq = settings.has_groq
    has_openai = settings.has_openai
    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    if has_groq and has_openai:
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"Groq ('{settings.groq_model}') & fallback OpenAI ('{settings.openai_model}') configured",
            latency_ms=latency_ms,
        )
    elif has_groq:
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"Primary Groq ('{settings.groq_model}') configured",
            latency_ms=latency_ms,
        )
    elif has_openai:
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"Fallback OpenAI ('{settings.openai_model}') configured",
            latency_ms=latency_ms,
        )
    else:
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details="No LLM API keys configured (GROQ_API_KEY or OPENAI_API_KEY required for reasoning)",
            latency_ms=latency_ms,
        )


def check_storage_health(settings: Settings) -> ComponentHealth:
    """Verify local filesystem or S3 document storage accessibility and write permissions."""
    start_t = time.perf_counter()
    try:
        path = settings.storage_path
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / f".healthcheck_probe_{int(time.time())}"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"Storage mode '{settings.storage_mode}' at '{path}' is writable",
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        logger.warning(f"Storage health check error: {exc}")
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details=f"Storage inaccessible or not writable: {exc}",
            latency_ms=latency_ms,
        )


def check_neo4j_health(settings: Settings) -> ComponentHealth:
    """Verify Neo4j knowledge graph connectivity and database availability."""
    start_t = time.perf_counter()
    uri = settings.neo4j_uri
    if not uri:
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details="Neo4j URI not configured",
            latency_ms=0.0,
        )

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            connection_timeout=2.0,
        )
        try:
            with driver.session(database=settings.neo4j_database) as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single()
                if record and record["ping"] == 1:
                    latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
                    return ComponentHealth(
                        status=ComponentStatus.HEALTHY,
                        details=f"Neo4j connection verified at {uri}",
                        latency_ms=latency_ms,
                    )
        finally:
            driver.close()
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        # Neo4j is a Phase 2 component; failure is treated as degraded in Phase 1
        return ComponentHealth(
            status=ComponentStatus.DEGRADED,
            details=f"Neo4j connection unavailable: {exc}",
            latency_ms=latency_ms,
        )

    return ComponentHealth(
        status=ComponentStatus.DEGRADED,
        details="Neo4j connection ping returned unexpected result",
        latency_ms=round((time.perf_counter() - start_t) * 1000.0, 2),
    )


def aggregate_system_status(components: dict[str, ComponentHealth]) -> ComponentStatus:
    """Aggregate individual subsystem statuses into an overall system health verdict.

    Rules:
    - If any component is UNHEALTHY -> Overall UNHEALTHY.
    - If any component is DEGRADED (and none UNHEALTHY) -> Overall DEGRADED.
    - Otherwise -> Overall HEALTHY.
    """
    statuses = [comp.status for comp in components.values()]
    if ComponentStatus.UNHEALTHY in statuses:
        return ComponentStatus.UNHEALTHY
    if ComponentStatus.DEGRADED in statuses:
        return ComponentStatus.DEGRADED
    return ComponentStatus.HEALTHY


# ------------------------------------------------------------------------------
# Route Handlers
# ------------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Application liveness probe",
    description="Lightweight liveness probe indicating application server process is running and accepting HTTP requests.",
)
def liveness_check(
    settings: Settings = Depends(get_app_settings),
) -> HealthCheckResponse:
    """Return fast liveness confirmation."""
    return HealthCheckResponse(
        status=ComponentStatus.HEALTHY,
        version=APP_VERSION,
        environment=settings.app_env,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        components={
            "api": ComponentHealth(
                status=ComponentStatus.HEALTHY,
                details=f"{settings.app_name} API is operational",
                latency_ms=0.0,
            )
        },
    )


@router.get(
    "/live",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes liveness probe alias",
    description="Alias for `/health` matching standard Kubernetes container liveness probe paths.",
    include_in_schema=False,
)
def kubernetes_liveness_alias(
    settings: Settings = Depends(get_app_settings),
) -> HealthCheckResponse:
    """Alias for liveness_check."""
    return liveness_check(settings=settings)


@router.get(
    "/ready",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "model": HealthCheckResponse,
            "description": "System is ready and operational (HEALTHY or DEGRADED non-critical)",
        },
        503: {
            "model": HealthCheckResponse,
            "description": "System is unready or critical components failed (UNHEALTHY)",
        },
    },
    summary="Deep readiness probe",
    description="Comprehensive readiness check querying Qdrant, ColPali embedder, LLM configurations, storage, and Neo4j.",
)
def readiness_check(
    response: Response,
    settings: Settings = Depends(get_app_settings),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    embedder: ColPaliEmbedder = Depends(get_colpali_embedder),
) -> HealthCheckResponse:
    """Perform deep readiness probe across all subsystems and set 503 if unhealthy."""
    components: dict[str, ComponentHealth] = {
        "qdrant": check_qdrant_health(qdrant_store),
        "embedder": check_embedder_health(embedder),
        "llm": check_llm_health(settings),
        "storage": check_storage_health(settings),
        "neo4j": check_neo4j_health(settings),
    }

    overall_status = aggregate_system_status(components)

    if overall_status == ComponentStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthCheckResponse(
        status=overall_status,
        version=APP_VERSION,
        environment=settings.app_env,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        components=components,
    )


@router.get(
    "/health/components/{component_name}",
    response_model=ComponentHealth,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"model": ComponentHealth, "description": "Component health report"},
        404: {"model": HTTPErrorResponse, "description": "Unknown component name"},
    },
    summary="Single component health check",
    description="Diagnose a specific subsystem: 'qdrant', 'embedder', 'llm', 'storage', or 'neo4j'.",
)
def component_health_check(
    component_name: str,
    settings: Settings = Depends(get_app_settings),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    embedder: ColPaliEmbedder = Depends(get_colpali_embedder),
) -> ComponentHealth:
    """Probe and return health of an individual named component."""
    target = component_name.lower().strip()

    if target == "qdrant":
        return check_qdrant_health(qdrant_store)
    elif target in ("embedder", "colpali"):
        return check_embedder_health(embedder)
    elif target == "llm":
        return check_llm_health(settings)
    elif target == "storage":
        return check_storage_health(settings)
    elif target == "neo4j":
        return check_neo4j_health(settings)
    elif target in ("api", "server"):
        return ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details=f"{settings.app_name} web server is active",
            latency_ms=0.0,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "component_not_found",
                "message": (
                    f"Unknown component '{component_name}'. "
                    "Valid components are: 'qdrant', 'embedder', 'llm', 'storage', 'neo4j', 'api'."
                ),
            },
        )
