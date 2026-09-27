"""Unit tests for FastAPI main entry point application (app/main.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_legal_rag_chain,
    get_qdrant_store,
)
from app.config import Settings
from app.core.generation.chain import LegalRAGChain
from app.core.ingestion.colpali_embedder import ColPaliEmbedder
from app.core.retrieval.qdrant_store import QdrantStore
from app.main import APP_DESCRIPTION, APP_TITLE, APP_VERSION, create_app


@pytest.fixture
def mock_qdrant():
    """Mock healthy QdrantStore instance."""
    store = MagicMock(spec=QdrantStore)
    store.collection_name = "test_collection"
    store.collection_exists.return_value = True
    return store


@pytest.fixture
def mock_embedder():
    """Mock ColPaliEmbedder instance."""
    embedder = MagicMock(spec=ColPaliEmbedder)
    embedder.model_name = "vidore/colpali-v1.3-hf"
    embedder.device = "cpu"
    embedder.batch_size = 4
    embedder.vector_dim = 128
    embedder.is_loaded = False
    return embedder


@pytest.fixture
def mock_chain():
    """Mock LegalRAGChain instance."""
    return MagicMock(spec=LegalRAGChain)


@pytest.fixture
def test_settings(tmp_path):
    """Test Settings with mock values and isolated temp directory."""
    return Settings(
        app_name="LexisGraph Main Test",
        app_env="test",
        debug=True,
        log_level="DEBUG",
        local_storage_dir=str(tmp_path / "storage"),
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection_name="test_collection",
    )


@pytest.fixture
def app_instance(test_settings, mock_qdrant, mock_embedder, mock_chain):
    """FastAPI app instance with dependency overrides."""
    application = create_app(settings=test_settings)
    application.dependency_overrides[get_app_settings] = lambda: test_settings
    application.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant
    application.dependency_overrides[get_colpali_embedder] = lambda: mock_embedder
    application.dependency_overrides[get_legal_rag_chain] = lambda: mock_chain
    return application


@pytest.fixture
def client(app_instance):
    """TestClient context with overridden dependencies."""
    with TestClient(app_instance) as test_client:
        yield test_client


def test_root_endpoint(client, test_settings):
    """Verify GET / returns 200 and expected metadata dictionary."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_settings.app_name
    assert data["version"] == APP_VERSION
    assert data["description"] == APP_DESCRIPTION
    assert data["environment"] == "test"
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"
    assert data["api_v1_prefix"] == "/api/v1"


def test_request_timing_headers(client):
    """Verify middleware adds X-Request-ID and X-Process-Time-Ms response headers."""
    response = client.get("/")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers
    # Process time should be a valid float
    process_time = float(response.headers["x-process-time-ms"])
    assert process_time >= 0.0


def test_custom_request_id_preserved(client):
    """Verify incoming X-Request-ID is preserved and echoed back."""
    custom_id = "test-request-uuid-12345"
    response = client.get("/", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id


def test_cors_headers(client):
    """Verify CORS preflight and allow origin headers."""
    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_health_routes_mounted(client):
    """Verify health routes are accessible at root and under /api/v1."""
    # Root mounts
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] in ["healthy", "degraded"]

    res_live = client.get("/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "healthy"

    # API v1 mounts
    res_api_health = client.get("/api/v1/health")
    assert res_api_health.status_code == 200

    res_api_live = client.get("/api/v1/live")
    assert res_api_live.status_code == 200


def test_documents_routes_mounted(client):
    """Verify documents router is accessible under /api/v1/documents."""
    response = client.get("/api/v1/documents")
    # Should be 200 or 404/empty list, definitely not a missing route 404 on FastAPI level
    assert response.status_code == 200
    assert "documents" in response.json()


def test_query_route_mounted(client):
    """Verify query router is mounted under /api/v1/query (validates 422 on empty POST)."""
    response = client.post("/api/v1/query", json={})
    # Missing required body field 'question' produces 422 Unprocessable Entity
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"] == "VALIDATION_ERROR"
    assert "message" in data


def test_404_handler(client):
    """Verify non-existent route returns 404."""
    response = client.get("/api/v1/non_existent_endpoint_xyz")
    assert response.status_code == 404
