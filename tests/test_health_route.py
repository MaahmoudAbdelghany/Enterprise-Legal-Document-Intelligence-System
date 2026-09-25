"""Unit tests for LexisGraph health and readiness API routes (app/api/routes/health.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_qdrant_store,
)
from app.api.routes.health import (
    aggregate_system_status,
    check_embedder_health,
    check_llm_health,
    check_neo4j_health,
    check_qdrant_health,
    check_storage_health,
    router as health_router,
)
from app.config import Settings
from app.core.ingestion.colpali_embedder import ColPaliEmbedder
from app.core.retrieval.qdrant_store import QdrantStore
from app.models.schemas import (
    ComponentHealth,
    ComponentStatus,
    HealthCheckResponse,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_store():
    """Mock QdrantStore instance reporting healthy collection status."""
    store = MagicMock(spec=QdrantStore)
    store.collection_name = "lexisgraph_legal_documents"
    store.vector_dim = 128
    store.collection_exists.return_value = True
    return store


@pytest.fixture
def mock_colpali_embedder():
    """Mock ColPaliEmbedder instance."""
    embedder = MagicMock(spec=ColPaliEmbedder)
    embedder.model_name = "vidore/colpali-v1.3-hf"
    embedder.device = "cpu"
    embedder.vector_dim = 128
    embedder.is_loaded = False
    return embedder


@pytest.fixture
def test_settings(tmp_path):
    """Test Settings with mock keys and isolated temp directory."""
    return Settings(
        app_name="LexisGraph Test",
        app_env="test",
        groq_api_key="gsk_test_mock_key",
        groq_model="llama-3.3-70b-versatile",
        openai_api_key="sk-test-mock-key",
        openai_model="gpt-4o",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection_name="test_collection",
        local_storage_dir=str(tmp_path / "test_storage"),
        neo4j_uri="bolt://localhost:7687",
    )


@pytest.fixture
def client(test_settings, mock_qdrant_store, mock_colpali_embedder):
    """FastAPI TestClient wired with mock dependencies."""
    app = FastAPI(title="LexisGraph Health Test API")
    app.include_router(health_router)

    app.dependency_overrides[get_app_settings] = lambda: test_settings
    app.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant_store
    app.dependency_overrides[get_colpali_embedder] = lambda: mock_colpali_embedder

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ------------------------------------------------------------------------------
# Liveness Probe Tests (/health, /live)
# ------------------------------------------------------------------------------

def test_liveness_check_returns_200(client, test_settings):
    """Verify /health returns 200 OK and expected HealthCheckResponse structure."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == ComponentStatus.HEALTHY
    assert data["version"] == "0.1.0"
    assert data["environment"] == "test"
    assert "timestamp" in data
    assert "api" in data["components"]
    assert data["components"]["api"]["status"] == ComponentStatus.HEALTHY


def test_kubernetes_liveness_alias_returns_200(client):
    """Verify /live alias behaves identically to /health."""
    response = client.get("/live")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == ComponentStatus.HEALTHY
    assert "api" in data["components"]


# ------------------------------------------------------------------------------
# Readiness Probe Tests (/ready)
# ------------------------------------------------------------------------------

def test_readiness_check_all_healthy(client):
    """Verify /ready returns 200 when components are healthy or degraded non-critical."""
    with patch("app.api.routes.health.check_neo4j_health") as mock_neo:
        mock_neo.return_value = ComponentHealth(
            status=ComponentStatus.HEALTHY,
            details="Neo4j connection verified",
            latency_ms=1.2,
        )
        response = client.get("/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)
        assert "qdrant" in data["components"]
        assert "embedder" in data["components"]
        assert "llm" in data["components"]
        assert "storage" in data["components"]
        assert "neo4j" in data["components"]
        assert data["components"]["qdrant"]["status"] == ComponentStatus.HEALTHY


def test_readiness_check_unhealthy_returns_503(client, mock_qdrant_store):
    """Verify /ready returns 503 SERVICE_UNAVAILABLE when a critical component is unhealthy."""
    mock_qdrant_store.collection_exists.side_effect = ConnectionError("Cannot reach Qdrant cluster")

    response = client.get("/ready")
    assert response.status_code == 503

    data = response.json()
    assert data["status"] == ComponentStatus.UNHEALTHY
    assert data["components"]["qdrant"]["status"] == ComponentStatus.UNHEALTHY
    assert "Cannot reach Qdrant" in data["components"]["qdrant"]["details"]


# ------------------------------------------------------------------------------
# Single Component Health Tests (/health/components/{component})
# ------------------------------------------------------------------------------

def test_component_health_qdrant(client):
    """Verify individual probe for qdrant."""
    response = client.get("/health/components/qdrant")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == ComponentStatus.HEALTHY
    assert "test_collection" in data["details"] or "reachable" in data["details"] or "active" in data["details"]


def test_component_health_embedder(client):
    """Verify individual probe for embedder and colpali alias."""
    for comp in ["embedder", "colpali"]:
        response = client.get(f"/health/components/{comp}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ComponentStatus.HEALTHY
        assert "vidore/colpali-v1.3-hf" in data["details"]


def test_component_health_llm(client):
    """Verify individual probe for llm."""
    response = client.get("/health/components/llm")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == ComponentStatus.HEALTHY
    assert "Groq" in data["details"]


def test_component_health_storage(client):
    """Verify individual probe for storage."""
    response = client.get("/health/components/storage")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == ComponentStatus.HEALTHY
    assert "writable" in data["details"]


def test_component_health_neo4j(client):
    """Verify individual probe for neo4j."""
    response = client.get("/health/components/neo4j")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)


def test_component_health_api(client):
    """Verify individual probe for api / server."""
    for comp in ["api", "server"]:
        response = client.get(f"/health/components/{comp}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ComponentStatus.HEALTHY


def test_component_health_unknown_returns_404(client):
    """Verify probing an invalid component returns 404 NOT_FOUND."""
    response = client.get("/health/components/nonexistent_subsystem")
    assert response.status_code == 404

    data = response.json()
    assert "detail" in data
    assert data["detail"]["error"] == "component_not_found"


# ------------------------------------------------------------------------------
# Unit Tests for Checker Helper Functions
# ------------------------------------------------------------------------------

def test_check_qdrant_health_none_store():
    """Verify check_qdrant_health handles None store gracefully."""
    res = check_qdrant_health(None)
    assert res.status == ComponentStatus.UNHEALTHY
    assert "not available" in res.details


def test_check_embedder_health_none_embedder():
    """Verify check_embedder_health handles None embedder gracefully."""
    res = check_embedder_health(None)
    assert res.status == ComponentStatus.DEGRADED
    assert "uninitialized" in res.details


def test_check_embedder_health_exception():
    """Verify check_embedder_health handles exception gracefully."""
    embedder = MagicMock()
    type(embedder).model_name = property(lambda self: (_ for _ in ()).throw(RuntimeError("CUDA boom")))
    res = check_embedder_health(embedder)
    assert res.status == ComponentStatus.DEGRADED
    assert "CUDA boom" in res.details


def test_check_llm_health_configurations():
    """Verify check_llm_health with different provider configurations."""
    # Both Groq and OpenAI
    s_both = Settings(groq_api_key="gsk_123", openai_api_key="sk_123")
    res_both = check_llm_health(s_both)
    assert res_both.status == ComponentStatus.HEALTHY
    assert "Groq" in res_both.details and "fallback OpenAI" in res_both.details

    # Groq only
    s_groq = Settings(groq_api_key="gsk_123", openai_api_key="")
    res_groq = check_llm_health(s_groq)
    assert res_groq.status == ComponentStatus.HEALTHY
    assert "Primary Groq" in res_groq.details

    # OpenAI only
    s_openai = Settings(groq_api_key="", openai_api_key="sk_123")
    res_openai = check_llm_health(s_openai)
    assert res_openai.status == ComponentStatus.HEALTHY
    assert "Fallback OpenAI" in res_openai.details

    # Neither
    s_none = Settings(groq_api_key="", openai_api_key="")
    res_none = check_llm_health(s_none)
    assert res_none.status == ComponentStatus.DEGRADED
    assert "No LLM API keys configured" in res_none.details


def test_check_storage_health_error(tmp_path):
    """Verify check_storage_health handles write errors gracefully."""
    settings = Settings(local_storage_dir=str(tmp_path / "mock_dir"))
    broken_path = MagicMock()
    broken_path.mkdir.side_effect = PermissionError("Access denied")
    with patch.object(Settings, "storage_path", new_callable=lambda: broken_path):
        res = check_storage_health(settings)
        assert res.status == ComponentStatus.DEGRADED
        assert "Access denied" in res.details


def test_check_neo4j_health_no_uri():
    """Verify check_neo4j_health when uri is empty."""
    settings = Settings(neo4j_uri="")
    res = check_neo4j_health(settings)
    assert res.status == ComponentStatus.DEGRADED
    assert "not configured" in res.details


def test_check_neo4j_health_driver_success():
    """Verify check_neo4j_health when driver session returns ping."""
    settings = Settings(neo4j_uri="bolt://localhost:7687")
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"ping": 1}
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        res = check_neo4j_health(settings)
        assert res.status == ComponentStatus.HEALTHY
        assert "verified" in res.details


def test_check_neo4j_health_unexpected_ping():
    """Verify check_neo4j_health when driver session returns unexpected ping value."""
    settings = Settings(neo4j_uri="bolt://localhost:7687")
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = {"ping": 999}
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("neo4j.GraphDatabase.driver", return_value=mock_driver):
        res = check_neo4j_health(settings)
        assert res.status == ComponentStatus.DEGRADED
        assert "unexpected result" in res.details


def test_aggregate_system_status():
    """Verify aggregate_system_status logic."""
    # All healthy
    comps_all_healthy = {
        "a": ComponentHealth(status=ComponentStatus.HEALTHY),
        "b": ComponentHealth(status=ComponentStatus.HEALTHY),
    }
    assert aggregate_system_status(comps_all_healthy) == ComponentStatus.HEALTHY

    # One degraded, none unhealthy -> degraded
    comps_degraded = {
        "a": ComponentHealth(status=ComponentStatus.HEALTHY),
        "b": ComponentHealth(status=ComponentStatus.DEGRADED),
    }
    assert aggregate_system_status(comps_degraded) == ComponentStatus.DEGRADED

    # One unhealthy -> unhealthy
    comps_unhealthy = {
        "a": ComponentHealth(status=ComponentStatus.HEALTHY),
        "b": ComponentHealth(status=ComponentStatus.DEGRADED),
        "c": ComponentHealth(status=ComponentStatus.UNHEALTHY),
    }
    assert aggregate_system_status(comps_unhealthy) == ComponentStatus.UNHEALTHY
