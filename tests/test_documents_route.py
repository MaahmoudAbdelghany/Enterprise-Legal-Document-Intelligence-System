"""Unit tests for LexisGraph document management API routes (app/api/routes/documents.py)."""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_pdf_processor,
    get_qdrant_store,
    get_text_extractor,
)
from app.api.routes.documents import _DOCUMENT_STATUS, router as documents_router
from app.config import Settings
from app.core.ingestion.colpali_embedder import ColPaliEmbedder, PageEmbedding
from app.core.ingestion.pdf_processor import PDFMetadata, PDFProcessor
from app.core.ingestion.text_extractor import PageText, TextExtractor
from app.core.retrieval.qdrant_store import QdrantStore
from app.models.schemas import DocumentStatus


# ------------------------------------------------------------------------------
# Fixtures & Test Setup
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_store():
    """Create an isolated in-memory QdrantStore."""
    store = QdrantStore(
        in_memory=True,
        collection_name="test_documents_api",
        vector_dim=128,
        auto_init=True,
    )
    yield store
    store.close()


@pytest.fixture
def mock_pdf_processor():
    """Mock PDFProcessor to avoid requiring system poppler binaries in tests."""
    processor = MagicMock(spec=PDFProcessor)
    dummy_img = Image.new("RGB", (200, 300), color=(255, 255, 255))
    processor.convert_to_images.return_value = [
        (1, dummy_img),
        (2, dummy_img),
    ]
    processor.get_metadata.return_value = PDFMetadata(
        total_pages=2,
        title="Test Legal Contract",
        is_encrypted=False,
    )
    return processor


@pytest.fixture
def mock_colpali_embedder():
    """Mock ColPaliEmbedder returning predictable dummy multi-vector embeddings."""
    embedder = MagicMock(spec=ColPaliEmbedder)

    def _embed_pages(pages):
        results = []
        for idx, _ in enumerate(pages, start=1):
            dummy_vecs = np.ones((8, 128), dtype=np.float32) / np.sqrt(128)
            results.append(PageEmbedding(page_number=idx, embeddings=dummy_vecs, vector_dim=128))
        return results

    embedder.embed_pages.side_effect = _embed_pages
    return embedder


@pytest.fixture
def mock_text_extractor():
    """Mock TextExtractor returning Arabic legal text for test pages."""
    extractor = MagicMock(spec=TextExtractor)
    extractor.has_active_provider = True
    extractor.extract_document_text.return_value = [
        PageText(
            page_number=1,
            raw_text="عقد بيع تجاري - الطرف الأول شركة الأمل",
            normalized_text="عقد بيع تجاري - الطرف الاول شركة الامل",
            language="ar",
            has_tables=False,
        ),
        PageText(
            page_number=2,
            raw_text="البند الثاني: التزامات المشتري وغرامات التأخير",
            normalized_text="البند الثاني: التزامات المشتري وغرامات التاخير",
            language="ar",
            has_tables=True,
        ),
    ]
    return extractor


@pytest.fixture
def client(
    mock_qdrant_store,
    mock_pdf_processor,
    mock_colpali_embedder,
    mock_text_extractor,
    tmp_path,
):
    """FastAPI TestClient with all dependencies cleanly injected and overridden."""
    test_settings = Settings(
        local_storage_dir=str(tmp_path / "documents"),
        qdrant_collection_name="test_documents_api",
        qdrant_vector_dim=128,
    )

    app = FastAPI()
    app.include_router(documents_router, prefix="/api/v1")

    app.dependency_overrides[get_app_settings] = lambda: test_settings
    app.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant_store
    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_colpali_embedder] = lambda: mock_colpali_embedder
    app.dependency_overrides[get_text_extractor] = lambda: mock_text_extractor

    _DOCUMENT_STATUS.clear()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    _DOCUMENT_STATUS.clear()


# ------------------------------------------------------------------------------
# Test Cases: Upload Endpoint
# ------------------------------------------------------------------------------

def test_upload_valid_pdf_synchronous(client):
    """Test successful synchronous upload of a valid Arabic legal PDF."""
    pdf_bytes = b"%PDF-1.4 dummy valid PDF header and bytes"
    files = {"file": ("contract.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {
        "document_id": "doc_contract_001",
        "legal_category": "commercial_contract",
        "parties": "شركة الأمل, مؤسسة البناء",
        "tags": "عقد, تجاري, 2026",
        "metadata_json": json.dumps({"court": "المحكمة التجارية بالرياض"}),
        "background": "false",
    }

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 201
    payload = response.json()
    assert payload["document_id"] == "doc_contract_001"
    assert payload["filename"] == "contract.pdf"
    assert payload["page_count"] == 2
    assert payload["status"] == "completed"
    assert "uploaded, embedded, and indexed" in payload["message"]
    assert payload["metadata"]["legal_category"] == "commercial_contract"
    assert "شركة الأمل" in payload["metadata"]["parties"]


def test_upload_with_background_task(client):
    """Test upload with background=True returns pending status immediately."""
    pdf_bytes = b"%PDF-1.7 background legal dossier"
    files = {"file": ("large_court_record.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {
        "document_id": "doc_bg_001",
        "background": "true",
    }

    response = client.post("/api/v1/documents/upload", files=files, data=data)
    assert response.status_code == 201
    payload = response.json()
    assert payload["document_id"] == "doc_bg_001"
    assert payload["filename"] == "large_court_record.pdf"
    # Background response returns 0 pages until worker completes
    assert payload["status"] in ("pending", "completed")


def test_upload_invalid_file_extension(client):
    """Reject uploads with non-PDF file extensions."""
    files = {"file": ("notes.txt", io.BytesIO(b"Just plain text"), "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_empty_file(client):
    """Reject zero-byte empty uploads."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_invalid_pdf_header(client):
    """Reject files that have .pdf extension but lack %PDF- header."""
    files = {"file": ("fake.pdf", io.BytesIO(b"NOT A REAL PDF CONTENT"), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file header" in response.json()["detail"]


# ------------------------------------------------------------------------------
# Test Cases: List, Detail, and Delete Endpoints
# ------------------------------------------------------------------------------

def test_list_documents_empty(client):
    """Test listing documents when none have been indexed."""
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 0
    assert payload["documents"] == []


def test_list_and_get_document(client):
    """Test listing documents and retrieving details for an indexed document."""
    # First upload a document
    pdf_bytes = b"%PDF-1.4 test document"
    files = {"file": ("deed.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"document_id": "doc_deed_100", "legal_category": "real_estate"},
    )

    # List documents
    list_resp = client.get("/api/v1/documents?limit=10&offset=0")
    assert list_resp.status_code == 200
    list_payload = list_resp.json()
    assert list_payload["total_count"] >= 1
    doc_ids = [d["document_id"] for d in list_payload["documents"]]
    assert "doc_deed_100" in doc_ids

    # Get document detail
    detail_resp = client.get("/api/v1/documents/doc_deed_100")
    assert detail_resp.status_code == 200
    detail_payload = detail_resp.json()
    assert detail_payload["document"]["document_id"] == "doc_deed_100"
    assert detail_payload["document"]["page_count"] == 2
    assert len(detail_payload["pages"]) == 2
    assert detail_payload["pages"][0]["page_number"] == 1


def test_get_document_not_found(client):
    """Test 404 response when querying a non-existent document ID."""
    response = client.get("/api/v1/documents/non_existent_doc_id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_document_page(client):
    """Test retrieving an individual page's verbatim text and metadata."""
    pdf_bytes = b"%PDF-1.4 test document"
    files = {"file": ("ruling.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"document_id": "doc_ruling_200"},
    )

    # Get page 1
    page_resp = client.get("/api/v1/documents/doc_ruling_200/pages/1")
    assert page_resp.status_code == 200
    page_data = page_resp.json()
    assert page_data["page_number"] == 1
    assert "شركة الأمل" in page_data["raw_text"]

    # Get non-existent page
    bad_page_resp = client.get("/api/v1/documents/doc_ruling_200/pages/99")
    assert bad_page_resp.status_code == 404


def test_get_document_status(client):
    """Test status endpoint returns status for indexed document."""
    pdf_bytes = b"%PDF-1.4 test document"
    files = {"file": ("labor.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"document_id": "doc_labor_300"},
    )

    status_resp = client.get("/api/v1/documents/doc_labor_300/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["document_id"] == "doc_labor_300"
    assert status_data["status"] == "completed"


def test_delete_document(client):
    """Test successful deletion of an indexed document."""
    pdf_bytes = b"%PDF-1.4 test document"
    files = {"file": ("nda.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    client.post(
        "/api/v1/documents/upload",
        files=files,
        data={"document_id": "doc_nda_400"},
    )

    # Delete
    del_resp = client.delete("/api/v1/documents/doc_nda_400")
    assert del_resp.status_code == 200
    del_payload = del_resp.json()
    assert del_payload["deleted"] is True
    assert del_payload["document_id"] == "doc_nda_400"
    assert del_payload["qdrant_points_deleted"] == 2

    # Verify 404 after deletion
    get_resp = client.get("/api/v1/documents/doc_nda_400")
    assert get_resp.status_code == 404


def test_delete_non_existent_document(client):
    """Test 404 on deleting a document that does not exist."""
    response = client.delete("/api/v1/documents/unknown_doc_999")
    assert response.status_code == 404
