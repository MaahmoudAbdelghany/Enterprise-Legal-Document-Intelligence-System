"""Unit tests for LexisGraph legal RAG query API routes (app/api/routes/query.py)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_legal_rag_chain,
    get_qdrant_store,
)
from app.api.routes.query import router as query_router
from app.config import Settings
from app.core.generation.chain import (
    ContextFormattingError,
    LegalGenerationError,
    LegalRAGChain,
    LegalRAGInput,
    LegalRAGOutput,
    NoLLMProviderConfiguredError,
    SourceDocument,
)
from app.core.ingestion.colpali_embedder import (
    ColPaliEmbedder,
    EmptyInputError,
    QueryEmbedding,
)
from app.core.retrieval.qdrant_store import (
    QdrantSearchResult,
    QdrantStore,
    QdrantStoreError,
    SearchError,
)
from app.models.schemas import LegalLanguage, RetrievalStrategy, TaskType


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_colpali_embedder():
    """Mock ColPaliEmbedder returning predictable dummy query multi-vector embeddings."""
    embedder = MagicMock(spec=ColPaliEmbedder)
    dummy_vecs = np.ones((5, 128), dtype=np.float32) / np.sqrt(128)
    embedder.embed_query.return_value = QueryEmbedding(
        query="ما هي شروط إنهاء العقد؟",
        embeddings=dummy_vecs,
        vector_dim=128,
    )
    return embedder


@pytest.fixture
def mock_qdrant_store():
    """Mock QdrantStore returning search results."""
    store = MagicMock(spec=QdrantStore)
    store.search.return_value = [
        QdrantSearchResult(
            point_id="pt-001",
            document_id="doc-contract-1",
            page_number=1,
            score=0.925,
            raw_text="المادة 15: يحق لأي من الطرفين إنهاء العقد بإخطار كتابي مدته 30 يوماً.",
            normalized_text="المادة 15: يحق لاي من الطرفين انهاء العقد باخطار كتابي مدته 30 يوما.",
            language="ar",
            has_tables=False,
            metadata={"legal_category": "commercial"},
            filename="contract_ar.pdf",
        ),
        QdrantSearchResult(
            point_id="pt-002",
            document_id="doc-contract-1",
            page_number=2,
            score=0.871,
            raw_text="جدول الغرامات والتعويضات في حال الإنهاء التعسفي.",
            normalized_text="جدول الغرامات والتعويضات في حال الانهاء التعسفي.",
            language="ar",
            has_tables=True,
            metadata={"legal_category": "commercial"},
            filename="contract_ar.pdf",
        ),
    ]
    return store


@pytest.fixture
def mock_legal_rag_chain():
    """Mock LegalRAGChain returning structured legal answer and streaming tokens."""
    chain = MagicMock(spec=LegalRAGChain)
    chain.model_name = "mock-llama-3.3-70b"

    # Async invoke mock
    async def _mock_ainvoke(rag_input: LegalRAGInput):
        sources = [
            SourceDocument(
                document_id="doc-contract-1",
                page_number=1,
                score=0.925,
                filename="contract_ar.pdf",
                text_snippet="المادة 15: يحق لأي من الطرفين إنهاء العقد بإخطار كتابي مدته 30 يوماً.",
                has_tables=False,
            ),
            SourceDocument(
                document_id="doc-contract-1",
                page_number=2,
                score=0.871,
                filename="contract_ar.pdf",
                text_snippet="جدول الغرامات والتعويضات في حال الإنهاء التعسفي.",
                has_tables=True,
            ),
        ]
        return LegalRAGOutput(
            answer="وفقاً لنص المادة 15 من العقد، يحق لأي من الطرفين إنهاء العقد بإخطار كتابي مسبق مدته 30 يوماً.",
            sources=sources,
            question=rag_input.question,
            language="ar",
            model_used="mock-llama-3.3-70b",
            latency_seconds=0.35,
        )

    chain.ainvoke = AsyncMock(side_effect=_mock_ainvoke)

    # Async stream mock
    async def _mock_astream(rag_input: LegalRAGInput):
        tokens = ["وفقاً ", "للمادة ", "15، ", "فترة ", "الإخطار ", "30 ", "يوماً."]
        for t in tokens:
            yield t

    chain.astream = _mock_astream
    return chain


@pytest.fixture
def client(mock_colpali_embedder, mock_qdrant_store, mock_legal_rag_chain):
    """FastAPI TestClient with all query route dependencies overridden."""
    app = FastAPI()
    app.include_router(query_router, prefix="/api/v1")

    test_settings = Settings(
        qdrant_collection_name="test_query_collection",
        qdrant_vector_dim=128,
    )

    app.dependency_overrides[get_app_settings] = lambda: test_settings
    app.dependency_overrides[get_colpali_embedder] = lambda: mock_colpali_embedder
    app.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant_store
    app.dependency_overrides[get_legal_rag_chain] = lambda: mock_legal_rag_chain

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------

def test_query_sync_success(client, mock_qdrant_store, mock_colpali_embedder):
    """Test successful synchronous legal query."""
    payload = {
        "question": "ما هي شروط إنهاء العقد وفترة الإخطار المطلوبة؟",
        "top_k": 3,
        "include_sources": True,
        "strategy": "auto",
        "task_type": "contract",
        "language": "ar",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == payload["question"]
    assert "المادة 15" in data["answer"]
    assert data["confidence"] == 0.925
    assert data["strategy_used"] == "visual"
    assert data["task_type"] == "contract"
    assert data["language"] == "ar"
    assert data["model_used"] == "mock-llama-3.3-70b"
    assert data["latency_seconds"] > 0
    assert len(data["sources"]) == 2
    assert data["sources"][0]["document_id"] == "doc-contract-1"
    assert data["sources"][0]["page_number"] == 1
    assert data["sources"][0]["score"] == 0.925
    assert data["sources"][1]["has_tables"] is True
    assert "retrieved_page_count" in data["metadata"]
    assert data["metadata"]["retrieved_page_count"] == 2

    # Verify calls
    mock_colpali_embedder.embed_query.assert_called_once_with(payload["question"])
    mock_qdrant_store.search.assert_called_once()


def test_query_with_filters(client, mock_qdrant_store):
    """Test query with complex metadata filters (doc ID, page range, tables, tags)."""
    payload = {
        "question": "ما هي التزامات المشتري؟",
        "top_k": 5,
        "filters": {
            "document_ids": ["doc-contract-1"],
            "filenames": ["contract_ar.pdf"],
            "page_range": [1, 5],
            "has_tables": True,
            "legal_category": "commercial",
            "tags": ["urgent", "contract"],
        },
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["question"] == payload["question"]
    mock_qdrant_store.search.assert_called_once()
    kwargs = mock_qdrant_store.search.call_args.kwargs
    assert kwargs["document_ids"] == ["doc-contract-1"]
    assert kwargs["page_range"] == (1, 5)
    assert kwargs["extra_filter"] is not None


def test_query_without_sources(client):
    """Test query with include_sources=False returns empty source list."""
    payload = {
        "question": "ما هو القانون الواجب التطبيق؟",
        "include_sources": False,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []


def test_query_stream_sse(client):
    """Test streaming query returns Server-Sent Events (SSE) stream."""
    payload = {
        "question": "ما هي مهلة السداد؟",
        "stream": True,
        "include_sources": True,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    data_events = [line for line in lines if line.startswith("data: ")]
    assert len(data_events) >= 3

    # Check first token event
    first_payload = json.loads(data_events[0].replace("data: ", ""))
    assert "token" in first_payload
    assert first_payload["is_final"] is False

    # Check last data payload before DONE
    final_payload = json.loads(data_events[-2].replace("data: ", ""))
    assert final_payload["is_final"] is True
    assert final_payload["sources"] is not None
    assert len(final_payload["sources"]) == 2

    # Check terminal DONE event
    assert data_events[-1] == "data: [DONE]"


def test_query_stream_convenience_endpoint(client):
    """Test dedicated /api/v1/query/stream endpoint streams SSE."""
    payload = {
        "question": "ما هي الاختصاصات القضائية المحددة؟",
        "include_sources": True,
    }
    response = client.post("/api/v1/query/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data: [DONE]" in response.text


def test_query_empty_or_short_question(client):
    """Test validation failure when question is too short or empty."""
    # Question < 2 characters
    response = client.post("/api/v1/query", json={"question": "أ"})
    assert response.status_code == 422

    # Empty question
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_query_no_documents_found(client, mock_qdrant_store, mock_legal_rag_chain):
    """Test query behavior when Qdrant returns 0 search results."""
    mock_qdrant_store.search.return_value = []

    # Mock chain returns grounded fallback answer
    async def _mock_no_docs_ainvoke(rag_input: LegalRAGInput):
        return LegalRAGOutput(
            answer="بناءً على المستندات المتاحة، لا توجد معلومات كافية للإجابة على هذا السؤال.",
            sources=[],
            question=rag_input.question,
            language="ar",
            model_used="mock-llama-3.3-70b",
            latency_seconds=0.15,
        )

    mock_legal_rag_chain.ainvoke = AsyncMock(side_effect=_mock_no_docs_ainvoke)

    response = client.post("/api/v1/query", json={"question": "سؤال عن موضوع غير مفهرس؟"})
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] == 0.5
    assert data["sources"] == []
    assert "لا توجد معلومات كافية" in data["answer"]


def test_query_no_llm_configured(client, mock_legal_rag_chain):
    """Test HTTP 503 error when no LLM provider is configured."""
    mock_legal_rag_chain.ainvoke = AsyncMock(
        side_effect=NoLLMProviderConfiguredError("No LLM key configured")
    )

    response = client.post("/api/v1/query", json={"question": "ما هي المادة 1؟"})
    assert response.status_code == 503
    data = response.json()
    assert "LLM provider is not configured" in data["detail"]


def test_query_qdrant_error(client, mock_qdrant_store):
    """Test HTTP 502 error when Qdrant store is unavailable."""
    mock_qdrant_store.search.side_effect = QdrantStoreError("Connection refused")

    response = client.post("/api/v1/query", json={"question": "ما هي شروط العقد؟"})
    assert response.status_code == 502
    assert "Vector store unavailable" in response.json()["detail"]


def test_query_search_error(client, mock_qdrant_store):
    """Test HTTP 500 error when vector search query fails."""
    mock_qdrant_store.search.side_effect = SearchError("Invalid vector dimension")

    response = client.post("/api/v1/query", json={"question": "ما هي شروط العقد؟"})
    assert response.status_code == 500
    assert "Vector search failed" in response.json()["detail"]


def test_query_embedder_empty_input_error(client, mock_colpali_embedder):
    """Test HTTP 400 error when embedder raises EmptyInputError."""
    mock_colpali_embedder.embed_query.side_effect = EmptyInputError("Empty query")

    response = client.post("/api/v1/query", json={"question": "سؤال"})
    assert response.status_code == 400
    assert "Invalid question" in response.json()["detail"]


def test_query_context_formatting_error(client, mock_legal_rag_chain):
    """Test HTTP 422 error when context formatting fails."""
    mock_legal_rag_chain.ainvoke = AsyncMock(
        side_effect=ContextFormattingError("Context formatting failed")
    )

    response = client.post("/api/v1/query", json={"question": "ما هي المادة 1؟"})
    assert response.status_code == 422
    assert "Context formatting error" in response.json()["detail"]


def test_query_streaming_error(client, mock_legal_rag_chain):
    """Test streaming error chunk emission when chain.astream fails."""
    async def _failing_stream(rag_input):
        raise RuntimeError("LLM streaming connection aborted")
        yield "token"

    mock_legal_rag_chain.astream = _failing_stream

    response = client.post("/api/v1/query", json={"question": "سؤال للاختبار؟", "stream": True})
    assert response.status_code == 200
    assert "Streaming generation failed" in response.text
