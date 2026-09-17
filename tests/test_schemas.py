"""Unit tests for LexisGraph Pydantic schemas (app/models/schemas.py)."""

import datetime
import json
import pytest
from pydantic import ValidationError

from app.models.schemas import (
    BaseSchema,
    ComponentHealth,
    ComponentStatus,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    EntityNode,
    EntityRelationship,
    EntityType,
    ErrorDetail,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthCheckResponse,
    HTTPErrorResponse,
    LegalLanguage,
    PageMetadata,
    QueryFilter,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
    SourceReference,
    StreamTokenChunk,
    TaskType,
)


class TestEnumerations:
    """Tests for system enums."""

    def test_retrieval_strategy_values(self):
        assert RetrievalStrategy.AUTO == "auto"
        assert RetrievalStrategy.VISUAL == "visual"
        assert RetrievalStrategy.KEYWORD == "keyword"
        assert RetrievalStrategy.GRAPH == "graph"
        assert RetrievalStrategy.HYBRID == "hybrid"

    def test_task_type_values(self):
        assert TaskType.GENERAL == "general"
        assert TaskType.CONTRACT == "contract"
        assert TaskType.COURT_RULING == "court_ruling"
        assert TaskType.LEGISLATION == "legislation"

    def test_legal_language_values(self):
        assert LegalLanguage.AR == "ar"
        assert LegalLanguage.EN == "en"
        assert LegalLanguage.AUTO == "auto"

    def test_document_status_values(self):
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_component_status_values(self):
        assert ComponentStatus.HEALTHY == "healthy"
        assert ComponentStatus.DEGRADED == "degraded"
        assert ComponentStatus.UNHEALTHY == "unhealthy"

    def test_entity_type_values(self):
        assert EntityType.PERSON == "Person"
        assert EntityType.ORGANIZATION == "Organization"
        assert EntityType.LEGAL_ARTICLE == "LegalArticle"
        assert EntityType.COURT == "Court"
        assert EntityType.DATE == "Date"
        assert EntityType.LAW == "Law"
        assert EntityType.CONTRACT == "Contract"
        assert EntityType.CLAUSE == "Clause"


class TestDocumentSchemas:
    """Tests for document management schemas."""

    def test_page_metadata_defaults(self):
        page = PageMetadata(page_number=1)
        assert page.page_number == 1
        assert page.language == "ar"
        assert not page.has_tables
        assert page.character_count == 0
        assert page.preview_text == ""
        assert isinstance(page.created_at, datetime.datetime)

    def test_page_metadata_invalid_page(self):
        with pytest.raises(ValidationError):
            PageMetadata(page_number=0)

    def test_document_metadata(self):
        doc = DocumentMetadata(
            document_id="doc-123",
            filename="commercial_contract.pdf",
            page_count=5,
            legal_category="commercial",
            parties=["الطرف الأول", "الطرف الثاني"],
            tags=["contract", "nda"],
        )
        assert doc.document_id == "doc-123"
        assert doc.filename == "commercial_contract.pdf"
        assert doc.page_count == 5
        assert doc.status == DocumentStatus.COMPLETED
        assert len(doc.parties) == 2
        assert "contract" in doc.tags

        # JSON serialization check
        dumped = doc.model_dump()
        assert dumped["document_id"] == "doc-123"
        assert dumped["status"] == "completed"

    def test_document_upload_response(self):
        resp = DocumentUploadResponse(
            document_id="doc-999",
            filename="court_decision.pdf",
            page_count=3,
        )
        assert resp.document_id == "doc-999"
        assert resp.filename == "court_decision.pdf"
        assert resp.page_count == 3
        assert resp.status == DocumentStatus.COMPLETED
        assert "successfully" in resp.message

    def test_document_detail_response(self):
        doc = DocumentMetadata(
            document_id="doc-123",
            filename="test.pdf",
            page_count=1,
        )
        page = PageMetadata(page_number=1, character_count=500, preview_text="بسم الله")
        detail = DocumentDetailResponse(document=doc, pages=[page])
        assert detail.document.document_id == "doc-123"
        assert len(detail.pages) == 1
        assert detail.pages[0].preview_text == "بسم الله"

    def test_document_list_response(self):
        doc1 = DocumentMetadata(document_id="d1", filename="1.pdf", page_count=1)
        doc2 = DocumentMetadata(document_id="d2", filename="2.pdf", page_count=2)
        resp = DocumentListResponse(
            documents=[doc1, doc2],
            total_count=2,
            limit=10,
            offset=0,
        )
        assert len(resp.documents) == 2
        assert resp.total_count == 2
        assert resp.limit == 10

    def test_document_delete_response(self):
        resp = DocumentDeleteResponse(
            document_id="d1",
            deleted=True,
            qdrant_points_deleted=5,
            graph_nodes_deleted=12,
        )
        assert resp.document_id == "d1"
        assert resp.deleted is True
        assert resp.qdrant_points_deleted == 5
        assert resp.graph_nodes_deleted == 12


class TestQuerySchemas:
    """Tests for retrieval and query schemas."""

    def test_query_filter_valid_page_range(self):
        qf = QueryFilter(page_range=[1, 5])
        assert qf.page_range == [1, 5]

    def test_query_filter_invalid_page_range_length(self):
        with pytest.raises(ValidationError, match="exactly two integers"):
            QueryFilter(page_range=[1, 2, 3])

    def test_query_filter_invalid_page_range_order(self):
        with pytest.raises(ValidationError, match="start_page cannot be greater"):
            QueryFilter(page_range=[10, 2])

    def test_query_filter_invalid_page_range_bounds(self):
        with pytest.raises(ValidationError, match="page numbers must be >= 1"):
            QueryFilter(page_range=[0, 5])

    def test_query_request_defaults(self):
        req = QueryRequest(question="ما هو التزام الطرف الأول؟")
        assert req.question == "ما هو التزام الطرف الأول؟"
        assert req.strategy == RetrievalStrategy.AUTO
        assert req.task_type == TaskType.GENERAL
        assert req.language == LegalLanguage.AUTO
        assert req.top_k == 5
        assert not req.stream
        assert req.include_sources

    def test_query_request_validation(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="a")  # too short (< 2)

        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", top_k=0)  # ge=1

        with pytest.raises(ValidationError):
            QueryRequest(question="Valid question", top_k=100)  # le=50

    def test_source_reference(self):
        src = SourceReference(
            document_id="doc-abc",
            page_number=3,
            score=0.92,
            filename="labor_law.pdf",
            text_snippet="المادة 5: يحق للموظف إجازة سنوية...",
            has_tables=True,
        )
        assert src.document_id == "doc-abc"
        assert src.page_number == 3
        assert src.score == 0.92
        assert src.has_tables is True

    def test_query_response(self):
        resp = QueryResponse(
            question="ما هو التعويض المستحق؟",
            answer="يستحق الطرف الثاني تعويضاً قدره 50,000 ريال وفقاً للمادة 12.",
            sources=[
                SourceReference(
                    document_id="d-1",
                    page_number=2,
                    filename="contract.pdf",
                    text_snippet="نص المادة 12...",
                )
            ],
            confidence=0.95,
            strategy_used=RetrievalStrategy.HYBRID,
            task_type=TaskType.CONTRACT,
            language="ar",
            model_used="groq:llama-3.3-70b-versatile",
            latency_seconds=1.234,
        )
        assert resp.confidence == 0.95
        assert len(resp.sources) == 1
        assert resp.strategy_used == "hybrid"
        assert resp.task_type == "contract"
        assert resp.language == "ar"

        # Serialization to JSON
        json_data = resp.model_dump_json()
        data = json.loads(json_data)
        assert data["question"] == "ما هو التعويض المستحق؟"
        assert data["strategy_used"] == "hybrid"

    def test_stream_token_chunk(self):
        chunk1 = StreamTokenChunk(token="العقد ")
        assert chunk1.token == "العقد "
        assert chunk1.is_final is False
        assert chunk1.sources is None

        chunk2 = StreamTokenChunk(
            token="",
            is_final=True,
            sources=[SourceReference(document_id="d1", page_number=1)],
        )
        assert chunk2.is_final is True
        assert len(chunk2.sources) == 1


class TestHealthAndDiagnosticSchemas:
    """Tests for health and diagnostics models."""

    def test_health_check_response(self):
        qdrant_health = ComponentHealth(status=ComponentStatus.HEALTHY, latency_ms=4.2)
        neo4j_health = ComponentHealth(status=ComponentStatus.HEALTHY, latency_ms=12.5)

        health = HealthCheckResponse(
            status=ComponentStatus.HEALTHY,
            version="0.1.0",
            environment="test",
            components={"qdrant": qdrant_health, "neo4j": neo4j_health},
        )
        assert health.status == "healthy"
        assert health.components["qdrant"].status == "healthy"
        assert health.components["qdrant"].latency_ms == 4.2
        assert health.components["neo4j"].latency_ms == 12.5


class TestKnowledgeGraphSchemas:
    """Tests for knowledge graph models."""

    def test_graph_node_and_relationship(self):
        node1 = EntityNode(
            id="person:1",
            name="أحمد بن محمد",
            type=EntityType.PERSON,
            properties={"role": "مشتري"},
        )
        node2 = EntityNode(
            id="contract:1",
            name="عقد بيع عقار",
            type=EntityType.CONTRACT,
        )
        rel = EntityRelationship(
            source_id=node1.id,
            target_id=node2.id,
            relation_type="SIGNED_BY",
            properties={"date": "2025-01-01"},
        )
        assert node1.type == "Person"
        assert rel.relation_type == "SIGNED_BY"

        resp = GraphQueryResponse(
            nodes=[node1, node2],
            relationships=[rel],
            summary="تم العثور على علاقة توقيع عقد بين الطرفين.",
        )
        assert len(resp.nodes) == 2
        assert len(resp.relationships) == 1
        assert resp.summary is not None


class TestErrorSchemas:
    """Tests for error models."""

    def test_http_error_response(self):
        err = HTTPErrorResponse(
            error="DOCUMENT_NOT_FOUND",
            message="المستند المطلوب غير موجود في النظام.",
            details=[ErrorDetail(loc=["body", "document_id"], msg="Not found", type="not_found")],
        )
        assert err.error == "DOCUMENT_NOT_FOUND"
        assert len(err.details) == 1
        assert err.details[0].loc == ["body", "document_id"]
