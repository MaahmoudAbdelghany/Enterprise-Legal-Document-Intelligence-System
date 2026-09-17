"""Pydantic request and response schemas for LexisGraph.

Provides type-safe, validated domain models for the LexisGraph REST API,
including document ingestion, multi-vector retrieval, LCEL generation,
knowledge graph querying, health diagnostics, and error responses.
Built with Pydantic v2 conventions and comprehensive Arabic legal support.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# Enumerations
# ==============================================================================

class RetrievalStrategy(str, Enum):
    """Retrieval strategies supported by LexisGraph query pipeline."""

    AUTO = "auto"
    VISUAL = "visual"
    KEYWORD = "keyword"
    GRAPH = "graph"
    HYBRID = "hybrid"


class TaskType(str, Enum):
    """Specialized legal task types for tailored prompt engineering."""

    GENERAL = "general"
    CONTRACT = "contract"
    COURT_RULING = "court_ruling"
    LEGISLATION = "legislation"


class LegalLanguage(str, Enum):
    """Supported language modes for queries and generation."""

    AR = "ar"
    EN = "en"
    AUTO = "auto"


class DocumentStatus(str, Enum):
    """Processing and indexing lifecycle status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentStatus(str, Enum):
    """Operational health status of an infrastructure component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class EntityType(str, Enum):
    """Core legal domain entity types for knowledge graph representation."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    LEGAL_ARTICLE = "LegalArticle"
    COURT = "Court"
    DATE = "Date"
    LAW = "Law"
    CONTRACT = "Contract"
    CLAUSE = "Clause"


# ==============================================================================
# Common / Base Schema
# ==============================================================================

class BaseSchema(BaseModel):
    """Base Pydantic configuration model with common serializers and settings."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )


# ==============================================================================
# Document Ingestion & Management Models
# ==============================================================================

class PageMetadata(BaseSchema):
    """Metadata describing an individual page within an indexed document.

    Attributes:
        page_number: 1-indexed page sequence number.
        language: Detected primary language ('ar', 'en', etc.).
        has_tables: Indicates presence of tables, schedules, or tabular figures.
        character_count: Total extracted text length in characters.
        preview_text: Brief excerpt of the page's beginning text.
        created_at: ISO timestamp of extraction.
    """

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    language: str = Field(default="ar", description="Detected language code")
    has_tables: bool = Field(default=False, description="Whether the page contains tabular data")
    character_count: int = Field(default=0, ge=0, description="Length of extracted text in characters")
    preview_text: str = Field(default="", description="Snippet or preview of the page text")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Page indexing timestamp",
    )


class DocumentMetadata(BaseSchema):
    """Comprehensive metadata of an indexed legal document.

    Attributes:
        document_id: Unique identifier for the document.
        filename: Original file name.
        page_count: Total number of pages indexed.
        file_size_bytes: Document file size in bytes if available.
        status: Current indexing status.
        created_at: Upload and indexing timestamp.
        updated_at: Last modification timestamp.
        legal_category: Categorization (e.g., 'commercial_contract', 'labor_law').
        parties: Discovered or annotated legal parties.
        tags: Categorization tags.
        metadata: Arbitrary additional metadata key-value pairs.
    """

    document_id: str = Field(..., description="Unique document ID (e.g. UUID)")
    filename: str = Field(..., description="Original filename of the legal document")
    page_count: int = Field(..., ge=0, description="Total number of pages")
    file_size_bytes: int | None = Field(default=None, ge=0, description="File size in bytes")
    status: DocumentStatus = Field(default=DocumentStatus.COMPLETED, description="Indexing lifecycle status")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime.datetime | None = Field(default=None, description="Last update timestamp")
    legal_category: str | None = Field(default=None, description="Legal document category")
    parties: list[str] = Field(default_factory=list, description="Extracted or declared legal parties")
    tags: list[str] = Field(default_factory=list, description="Custom document tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom arbitrary metadata")


class DocumentUploadResponse(BaseSchema):
    """Response returned upon uploading and ingesting a document.

    Attributes:
        document_id: Unique generated or assigned ID of the document.
        filename: Original file name.
        page_count: Number of successfully converted and indexed pages.
        status: Ingestion status.
        message: Human-readable status message in Arabic or English.
        created_at: Indexing timestamp.
        metadata: Associated document metadata.
    """

    document_id: str = Field(..., description="Assigned unique document ID")
    filename: str = Field(..., description="Uploaded filename")
    page_count: int = Field(..., ge=0, description="Number of indexed pages")
    status: DocumentStatus = Field(default=DocumentStatus.COMPLETED, description="Ingestion status")
    message: str = Field(default="Document uploaded and indexed successfully.", description="Status message")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Indexing timestamp",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentDetailResponse(BaseSchema):
    """Detailed response for a single document including page-level records."""

    document: DocumentMetadata = Field(..., description="Document metadata")
    pages: list[PageMetadata] = Field(default_factory=list, description="List of page details")


class DocumentListResponse(BaseSchema):
    """Paginated response containing a list of indexed legal documents."""

    documents: list[DocumentMetadata] = Field(default_factory=list, description="List of document metadata items")
    total_count: int = Field(..., ge=0, description="Total number of documents matching query")
    limit: int = Field(default=20, ge=1, description="Pagination limit")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class DocumentDeleteResponse(BaseSchema):
    """Response returned when a document is deleted from storage and indexes."""

    document_id: str = Field(..., description="ID of the deleted document")
    deleted: bool = Field(default=True, description="Whether deletion was successful")
    message: str = Field(default="Document deleted successfully.", description="Status message")
    qdrant_points_deleted: int = Field(default=0, ge=0, description="Number of vector points deleted")
    graph_nodes_deleted: int = Field(default=0, ge=0, description="Number of knowledge graph nodes deleted")


# ==============================================================================
# Retrieval & RAG Query Models
# ==============================================================================

class QueryFilter(BaseSchema):
    """Filtering criteria applied to vector and hybrid retrieval.

    Attributes:
        document_ids: Restrict search to specific document identifiers.
        filenames: Restrict search to specific filenames.
        date_from: Filter documents created on or after this date.
        date_to: Filter documents created on or before this date.
        page_range: Restrict to a (start_page, end_page) inclusive range.
        has_tables: Filter specifically for pages with or without tables.
        legal_category: Filter by specific legal domain or category.
        tags: Require all or any of the specified tags.
    """

    document_ids: list[str] | None = Field(default=None, description="Target document IDs")
    filenames: list[str] | None = Field(default=None, description="Target document filenames")
    date_from: datetime.date | str | None = Field(default=None, description="Start date filter (YYYY-MM-DD)")
    date_to: datetime.date | str | None = Field(default=None, description="End date filter (YYYY-MM-DD)")
    page_range: list[int] | None = Field(
        default=None,
        description="Optional page bounds [min_page, max_page]",
    )
    has_tables: bool | None = Field(default=None, description="Filter by presence of tabular data")
    legal_category: str | None = Field(default=None, description="Filter by legal category")
    tags: list[str] | None = Field(default=None, description="Filter by tags")

    @field_validator("page_range")
    @classmethod
    def validate_page_range(cls, v: list[int] | None) -> list[int] | None:
        """Ensure page range contains 2 integers [min, max] with min <= max."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("page_range must contain exactly two integers: [start_page, end_page]")
            if v[0] > v[1]:
                raise ValueError("start_page cannot be greater than end_page in page_range")
            if v[0] < 1:
                raise ValueError("page numbers must be >= 1")
        return v


class QueryRequest(BaseSchema):
    """Legal RAG query request submitted to `/api/v1/query`.

    Attributes:
        question: The user query or legal research question.
        filters: Optional retrieval filter constraints.
        strategy: Retrieval strategy ('auto', 'visual', 'keyword', 'graph', 'hybrid').
        task_type: Specialized legal task type for prompt customization.
        language: Language preference for the answer ('ar', 'en', 'auto').
        top_k: Number of relevant pages or items to retrieve.
        stream: Whether to stream tokens via SSE (Server-Sent Events).
        include_sources: Whether to include cited source documents in response.
    """

    question: str = Field(
        ...,
        min_length=2,
        max_length=4000,
        description="Legal question or query prompt in Arabic or English",
        examples=["ما هي شروط إنهاء العقد وفترة الإخطار المطلوبة؟"],
    )
    filters: QueryFilter | None = Field(default=None, description="Optional metadata and document filters")
    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.AUTO,
        description="Retrieval strategy to execute",
    )
    task_type: TaskType = Field(
        default=TaskType.GENERAL,
        description="Specialized legal task prompt template to apply",
    )
    language: LegalLanguage = Field(
        default=LegalLanguage.AUTO,
        description="Preferred language for the generated legal analysis",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of context pages to retrieve",
    )
    stream: bool = Field(
        default=False,
        description="If True, response will be streamed via Server-Sent Events",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source attribution citations in response",
    )


class SourceReference(BaseSchema):
    """Source document page cited in the legal answer.

    Matches the internal `SourceDocument` from `app.core.generation.chain`.

    Attributes:
        document_id: ID of the source document.
        page_number: 1-indexed page number where information appears.
        score: Relevance score (MaxSim similarity or RRF score).
        filename: Original filename of the document.
        text_snippet: Text excerpt extracted from this page.
        has_tables: Whether tabular data exists on this page.
        metadata: Additional page-level metadata.
    """

    document_id: str = Field(..., description="Document identifier")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    score: float = Field(default=0.0, description="Relevance or similarity score")
    filename: str = Field(default="", description="Original file name")
    text_snippet: str = Field(default="", description="Text excerpt or content preview")
    has_tables: bool = Field(default=False, description="Whether page contains tabular content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResponse(BaseSchema):
    """Structured legal analysis and answer returned by `/api/v1/query`.

    Attributes:
        question: The original user question.
        answer: Generated grounded legal answer with citations.
        sources: List of cited source documents and pages.
        confidence: Confidence score of the response (0.0 to 1.0).
        strategy_used: Retrieval strategy actually employed.
        task_type: Applied legal task template.
        language: Detected or chosen language of the answer.
        model_used: Identifier of LLM that performed inference.
        latency_seconds: Total time in seconds to process query and generate answer.
        created_at: Timestamp of query execution.
        metadata: Telemetry and diagnostic metadata.
    """

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded legal answer with citations")
    sources: list[SourceReference] = Field(default_factory=list, description="Cited source documents")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Answer confidence score")
    strategy_used: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID, description="Strategy executed")
    task_type: TaskType = Field(default=TaskType.GENERAL, description="Applied legal task type")
    language: str = Field(default="ar", description="Language of generated answer")
    model_used: str = Field(default="", description="LLM model identifier")
    latency_seconds: float = Field(default=0.0, ge=0.0, description="Total pipeline latency in seconds")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Response generation timestamp",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Pipeline diagnostic metadata")


class StreamTokenChunk(BaseSchema):
    """Incremental streaming token event for Server-Sent Events (SSE)."""

    model_config = ConfigDict(str_strip_whitespace=False)

    token: str = Field(..., description="Next incremental token string")
    is_final: bool = Field(default=False, description="True when generation is complete")
    sources: list[SourceReference] | None = Field(default=None, description="Sources emitted at completion")


# ==============================================================================
# Health & Diagnostics Models
# ==============================================================================

class ComponentHealth(BaseSchema):
    """Health status and latency metrics for an individual system component."""

    status: ComponentStatus = Field(..., description="Operational status of component")
    details: str | None = Field(default=None, description="Descriptive status or diagnostic details")
    latency_ms: float | None = Field(default=None, ge=0.0, description="Round-trip latency in milliseconds")


class HealthCheckResponse(BaseSchema):
    """System-wide health check response for `/health` and `/ready` probes.

    Attributes:
        status: Overall aggregate system health.
        version: Application version.
        environment: Current operating environment ('development', 'production').
        timestamp: Current server UTC timestamp.
        components: Health breakdown per subsystem (qdrant, neo4j, llm, embedder).
    """

    status: ComponentStatus = Field(default=ComponentStatus.HEALTHY, description="Aggregate system health")
    version: str = Field(default="0.1.0", description="LexisGraph application version")
    environment: str = Field(default="development", description="Execution environment")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Report timestamp",
    )
    components: dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Subsystem health statuses (qdrant, neo4j, llm, embedder)",
    )


# ==============================================================================
# Knowledge Graph Models (Phase 2 Forward Compatibility)
# ==============================================================================

class EntityNode(BaseSchema):
    """Knowledge graph entity node representing a legal actor, concept, or document."""

    id: str = Field(..., description="Unique entity identifier in graph")
    name: str = Field(..., description="Canonical entity name in Arabic or English")
    type: EntityType | str = Field(..., description="Legal entity category")
    properties: dict[str, Any] = Field(default_factory=dict, description="Arbitrary entity properties")


class EntityRelationship(BaseSchema):
    """Knowledge graph relationship connecting two legal entities."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Relationship predicate (e.g., 'SIGNED_BY', 'AMENDS')")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relationship metadata")


class GraphQueryRequest(BaseSchema):
    """Request schema for querying the legal knowledge graph."""

    query: str | None = Field(default=None, description="Natural language or Cypher query")
    entity_types: list[EntityType | str] | None = Field(default=None, description="Filter by entity types")
    document_id: str | None = Field(default=None, description="Filter by containing document ID")
    limit: int = Field(default=50, ge=1, le=500, description="Max graph elements to return")


class GraphQueryResponse(BaseSchema):
    """Response containing extracted subgraph entities and relationships."""

    nodes: list[EntityNode] = Field(default_factory=list, description="Extracted entity nodes")
    relationships: list[EntityRelationship] = Field(default_factory=list, description="Extracted relationships")
    summary: str | None = Field(default=None, description="Optional textual summary of graph findings")


# ==============================================================================
# Error Handling Models
# ==============================================================================

class ErrorDetail(BaseSchema):
    """Detailed error item explaining validation or field errors."""

    loc: list[str | int] | None = Field(default=None, description="Field location path causing error")
    msg: str = Field(..., description="Human-readable error description")
    type: str = Field(default="value_error", description="Error category type")


class HTTPErrorResponse(BaseSchema):
    """Standardized error envelope returned on 4xx/5xx responses."""

    error: str = Field(..., description="Error identifier code")
    message: str = Field(..., description="User-facing error message")
    details: list[ErrorDetail] | dict[str, Any] | None = Field(
        default=None,
        description="Detailed diagnostic or validation breakdown",
    )
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Timestamp when error occurred",
    )
