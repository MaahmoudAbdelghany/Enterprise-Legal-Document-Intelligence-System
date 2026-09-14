"""Qdrant multi-vector store module for LexisGraph.

Provides storage, indexing, and MaxSim late-interaction retrieval for ColPali
document visual multi-vector embeddings, paired with extracted page text and
metadata for Arabic and multilingual legal documents.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence
import uuid

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

if TYPE_CHECKING:  # pragma: no cover
    from app.core.ingestion.colpali_embedder import PageEmbedding, QueryEmbedding
    from app.core.ingestion.text_extractor import PageText

from app.config import get_settings


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class QdrantStoreError(Exception):
    """Base exception for all Qdrant store operations."""


class CollectionInitError(QdrantStoreError):
    """Raised when collection creation, verification, or schema migration fails."""


class DocumentUpsertError(QdrantStoreError):
    """Raised when document or page insertion/updating fails."""


class SearchError(QdrantStoreError):
    """Raised when vector search query execution fails."""


class DocumentNotFoundError(QdrantStoreError):
    """Raised when a requested document or page does not exist in the collection."""


class InvalidEmbeddingError(QdrantStoreError):
    """Raised when input vectors have invalid shape, dimension, or formatting."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class QdrantPointPayload:
    """Structured payload stored alongside each page multi-vector point.

    Attributes:
        document_id: Unique identifier of the parent document.
        page_number: 1-indexed page number within the document.
        filename: Original filename of the document (optional).
        raw_text: Full verbatim text extracted from the page.
        normalized_text: Standardized/normalized Arabic text.
        language: Detected primary language ('ar', 'en', etc.).
        has_tables: Whether tabular structures are present.
        metadata: Custom arbitrary metadata (dates, court names, parties, tags).
        created_at: ISO-8601 timestamp of indexing.
    """

    document_id: str
    page_number: int
    filename: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    language: str = "ar"
    has_tables: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to a JSON-serializable dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "filename": self.filename,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "has_tables": self.has_tables,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QdrantPointPayload:
        """Instantiate payload from dictionary."""
        return cls(
            document_id=data.get("document_id", ""),
            page_number=data.get("page_number", 1),
            filename=data.get("filename", ""),
            raw_text=data.get("raw_text", ""),
            normalized_text=data.get("normalized_text", ""),
            language=data.get("language", "ar"),
            has_tables=data.get("has_tables", False),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class QdrantSearchResult:
    """Result item returned from ColPali multi-vector late-interaction search.

    Attributes:
        point_id: Unique Qdrant point UUID string.
        document_id: Parent document identifier.
        page_number: 1-indexed page number of the match.
        score: MaxSim late-interaction similarity score.
        raw_text: Verbatim page text for answer generation.
        normalized_text: Normalized Arabic text.
        language: Page language.
        has_tables: Whether the page contains tables.
        metadata: Document and page metadata dictionary.
        filename: Original file name.
        vector: Optional multi-vector representation if requested.
    """

    point_id: str
    document_id: str
    page_number: int
    score: float
    raw_text: str = ""
    normalized_text: str = ""
    language: str = "ar"
    has_tables: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    filename: str = ""
    vector: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "point_id": self.point_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "score": self.score,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "has_tables": self.has_tables,
            "metadata": dict(self.metadata),
            "filename": self.filename,
        }


@dataclass(frozen=True)
class QdrantPageRecord:
    """Direct record retrieval model for a document page stored in Qdrant."""

    point_id: str
    document_id: str
    page_number: int
    filename: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    language: str = "ar"
    has_tables: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


# ------------------------------------------------------------------------------
# ID Generation & Vector Helpers
# ------------------------------------------------------------------------------

def generate_point_id(doc_id: str, page_number: int) -> str:
    """Generate a deterministic RFC 4122 UUID for a specific document page.

    Ensures idempotency when documents or pages are re-indexed.

    Args:
        doc_id: Unique document identifier.
        page_number: 1-indexed page number.

    Returns:
        UUID string representation.
    """
    clean_doc = doc_id.strip() if doc_id else ""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lexisgraph:doc:{clean_doc}:page:{page_number}"))


def _extract_vector_list(embedding: Any) -> list[list[float]]:
    """Convert supported embedding types into nested float lists for Qdrant.

    Supports:
        - PageEmbedding / QueryEmbedding dataclass objects
        - 2D numpy arrays
        - Nested Python lists of floats

    Args:
        embedding: Multi-vector embedding representation.

    Returns:
        Nested list of floats of shape (num_vectors, vector_dim).

    Raises:
        InvalidEmbeddingError: If input cannot be converted or is not 2D.
    """
    if hasattr(embedding, "to_list") and callable(embedding.to_list):
        vec_list = embedding.to_list()
    elif isinstance(embedding, np.ndarray):
        if embedding.ndim != 2:
            raise InvalidEmbeddingError(
                f"Multi-vector array must be 2D, got shape {embedding.shape}"
            )
        vec_list = embedding.tolist()
    elif isinstance(embedding, (list, tuple)):
        vec_list = list(embedding)
    else:
        raise InvalidEmbeddingError(
            f"Unsupported embedding type '{type(embedding)}'. Expected PageEmbedding, "
            "QueryEmbedding, np.ndarray, or list of lists."
        )

    if not vec_list or not isinstance(vec_list, list):
        raise InvalidEmbeddingError("Embedding vector list cannot be empty.")

    first_item = vec_list[0]
    if not isinstance(first_item, (list, tuple, np.ndarray)):
        raise InvalidEmbeddingError(
            "Multi-vector must be a 2D sequence of token/patch vectors, got 1D sequence."
        )

    return [[float(v) for v in row] for row in vec_list]


# ------------------------------------------------------------------------------
# QdrantStore Engine Class
# ------------------------------------------------------------------------------

class QdrantStore:
    """Manages LexisGraph Qdrant multi-vector collections and CRUD operations.

    Configures late-interaction (MaxSim) comparator for OCR-free ColPali retrieval,
    supports filtered queries by document ID, page ranges, and metadata, and provides
    clean upsert and deletion methods.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_dim: int | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        url: str | None = None,
        in_memory: bool = False,
        auto_init: bool = True,
    ) -> None:
        """Initialize QdrantStore client and configuration.

        Args:
            client: Optional pre-configured QdrantClient instance.
            collection_name: Target collection name (default from settings).
            vector_dim: Multi-vector dimensionality (default from settings, typically 128).
            host: Qdrant host address.
            port: Qdrant HTTP port.
            api_key: Optional Qdrant API key.
            url: Optional full Qdrant URL.
            in_memory: If True, spin up an ephemeral in-memory Qdrant client (for testing).
            auto_init: If True, automatically ensure collection exists on startup.
        """
        settings = get_settings()
        self._collection_name = collection_name or settings.qdrant_collection_name
        self._vector_dim = vector_dim or settings.qdrant_vector_dim

        if client is not None:
            self._client = client
        elif in_memory:
            self._client = QdrantClient(":memory:")
            logger.info("Initialized in-memory QdrantClient.")
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key or settings.qdrant_api_key or None)
            logger.info(f"Initialized QdrantClient with URL: {url}")
        else:
            target_host = host or settings.qdrant_host
            target_port = port or settings.qdrant_port
            target_key = api_key or (settings.qdrant_api_key if settings.qdrant_api_key else None)
            use_https = settings.qdrant_use_https
            self._client = QdrantClient(
                host=target_host,
                port=target_port,
                api_key=target_key,
                https=use_https,
            )
            logger.info(
                f"Initialized QdrantClient connecting to {target_host}:{target_port} (https={use_https})"
            )

        if auto_init:
            self.init_collection(force_recreate=False)

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """Return underlying QdrantClient instance."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Return active collection name."""
        return self._collection_name

    @property
    def vector_dim(self) -> int:
        """Return configured multi-vector dimension."""
        return self._vector_dim

    # --------------------------------------------------------------------------
    # Context Management
    # --------------------------------------------------------------------------

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close underlying client connections."""
        try:
            self._client.close()
        except Exception as e:
            logger.debug(f"Error during QdrantClient close: {e}")

    # --------------------------------------------------------------------------
    # Collection Lifecycle
    # --------------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Check if the configured collection exists."""
        try:
            return bool(self._client.collection_exists(self._collection_name))
        except Exception as e:
            raise CollectionInitError(
                f"Failed to check collection existence for '{self._collection_name}': {e}"
            ) from e

    def init_collection(self, force_recreate: bool = False) -> None:
        """Create or verify the Qdrant multi-vector collection.

        Configures:
            - Vector size: vector_dim (e.g., 128 for ColPali)
            - Distance metric: COSINE
            - Multivector comparator: MAX_SIM (ColBERT late interaction)
            - Keyword and integer payload indexes for fast filtering

        Args:
            force_recreate: If True, delete existing collection before creating.

        Raises:
            CollectionInitError: If collection initialization fails.
        """
        try:
            exists = self.collection_exists()
            if exists:
                if force_recreate:
                    logger.warning(
                        f"Recreating Qdrant collection '{self._collection_name}' (force_recreate=True)..."
                    )
                    self._client.delete_collection(self._collection_name)
                else:
                    logger.info(f"Qdrant collection '{self._collection_name}' already exists.")
                    return

            logger.info(
                f"Creating Qdrant multi-vector collection '{self._collection_name}' "
                f"with dim={self._vector_dim}, distance=COSINE, comparator=MAX_SIM"
            )

            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

            # Create payload indexes for fast filtered lookups
            for field_name, schema in [
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("page_number", models.PayloadSchemaType.INTEGER),
                ("language", models.PayloadSchemaType.KEYWORD),
                ("has_tables", models.PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field_name,
                        field_schema=schema,
                    )
                except Exception as idx_err:
                    # In-memory Qdrant logs a harmless warning regarding payload indexes
                    logger.debug(f"Payload index note on '{field_name}': {idx_err}")

            logger.info(f"Successfully created collection '{self._collection_name}'.")

        except Exception as e:
            error_msg = f"Failed to initialize collection '{self._collection_name}': {e}"
            logger.error(error_msg)
            raise CollectionInitError(error_msg) from e

    def delete_collection(self) -> bool:
        """Delete the entire collection from Qdrant.

        Returns:
            True if collection was deleted or did not exist.
        """
        try:
            if self.collection_exists():
                self._client.delete_collection(self._collection_name)
                logger.info(f"Deleted collection '{self._collection_name}'.")
            return True
        except Exception as e:
            raise CollectionInitError(f"Failed to delete collection '{self._collection_name}': {e}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Retrieve telemetry and configuration info for the collection."""
        try:
            info = self._client.get_collection(self._collection_name)
            return {
                "status": getattr(info, "status", "unknown"),
                "points_count": getattr(info, "points_count", 0),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "vectors_count": getattr(info, "vectors_count", 0),
            }
        except Exception as e:
            raise CollectionInitError(f"Failed to get info for '{self._collection_name}': {e}") from e

    # --------------------------------------------------------------------------
    # Upsert Operations
    # --------------------------------------------------------------------------

    def upsert_page(
        self,
        doc_id: str,
        page_number: int,
        embedding: Any,
        page_text: PageText | None = None,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
    ) -> str:
        """Insert or update a single page multi-vector point in Qdrant.

        Args:
            doc_id: Unique identifier for the document.
            page_number: 1-indexed page number.
            embedding: PageEmbedding, 2D np.ndarray, or nested list of floats.
            page_text: Optional extracted PageText object.
            metadata: Additional user-defined metadata dictionary.
            filename: Original document filename.

        Returns:
            Generated deterministic point UUID string.

        Raises:
            DocumentUpsertError: If upsert operation fails.
            InvalidEmbeddingError: If embedding vector format is invalid.
        """
        if not doc_id or not doc_id.strip():
            raise DocumentUpsertError("document_id cannot be empty.")
        if page_number < 1:
            raise DocumentUpsertError(f"page_number must be >= 1, got {page_number}")

        clean_doc = doc_id.strip()
        vector_list = _extract_vector_list(embedding)

        if len(vector_list[0]) != self._vector_dim:
            raise InvalidEmbeddingError(
                f"Embedding dimension {len(vector_list[0])} does not match collection "
                f"dimension {self._vector_dim}"
            )

        merged_meta = dict(metadata or {})
        raw_text = ""
        normalized_text = ""
        language = "ar"
        has_tables = False

        if page_text is not None:
            raw_text = page_text.raw_text
            normalized_text = page_text.normalized_text
            language = page_text.language
            has_tables = page_text.has_tables
            if page_text.metadata:
                merged_meta.update(page_text.metadata)

        point_id = generate_point_id(clean_doc, page_number)
        payload = QdrantPointPayload(
            document_id=clean_doc,
            page_number=page_number,
            filename=filename or "",
            raw_text=raw_text,
            normalized_text=normalized_text,
            language=language,
            has_tables=has_tables,
            metadata=merged_meta,
        )

        point = models.PointStruct(
            id=point_id,
            vector=vector_list,
            payload=payload.to_dict(),
        )

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[point],
                wait=True,
            )
            logger.debug(
                f"Upserted page {page_number} for doc '{clean_doc}' (point_id={point_id}, "
                f"{len(vector_list)} vectors)"
            )
            return point_id
        except Exception as e:
            error_msg = f"Failed to upsert page {page_number} for doc '{clean_doc}': {e}"
            logger.error(error_msg)
            raise DocumentUpsertError(error_msg) from e

    def upsert_document_pages(
        self,
        doc_id: str,
        page_embeddings: Sequence[Any],
        page_texts: Sequence[PageText | None] | None = None,
        metadata: dict[str, Any] | None = None,
        filename: str | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        """Batch insert or update all pages of a document.

        Args:
            doc_id: Unique identifier for the document.
            page_embeddings: Sequence of page multi-vectors.
            page_texts: Optional sequence of PageText objects (matched by index).
            metadata: Common metadata dictionary applied to all pages.
            filename: Document filename.
            batch_size: Number of points sent per Qdrant upsert call.

        Returns:
            List of point UUID strings upserted.

        Raises:
            DocumentUpsertError: If inputs are invalid or upsert fails.
        """
        if not doc_id or not doc_id.strip():
            raise DocumentUpsertError("document_id cannot be empty.")
        if not page_embeddings:
            raise DocumentUpsertError("page_embeddings sequence cannot be empty.")

        clean_doc = doc_id.strip()
        common_meta = dict(metadata or {})
        points: list[models.PointStruct] = []
        point_ids: list[str] = []

        for idx, emb_item in enumerate(page_embeddings):
            # Derive page number (1-indexed)
            if hasattr(emb_item, "page_number"):
                page_num = int(emb_item.page_number)
            else:
                page_num = idx + 1

            p_text: PageText | None = None
            if page_texts is not None and idx < len(page_texts):
                p_text = page_texts[idx]
                if p_text is not None and hasattr(p_text, "page_number"):
                    page_num = p_text.page_number

            vector_list = _extract_vector_list(emb_item)
            if len(vector_list[0]) != self._vector_dim:
                raise InvalidEmbeddingError(
                    f"Page {page_num} embedding dimension {len(vector_list[0])} does not match "
                    f"collection dimension {self._vector_dim}"
                )

            item_meta = dict(common_meta)
            raw_text = ""
            normalized_text = ""
            language = "ar"
            has_tables = False

            if p_text is not None:
                raw_text = p_text.raw_text
                normalized_text = p_text.normalized_text
                language = p_text.language
                has_tables = p_text.has_tables
                if p_text.metadata:
                    item_meta.update(p_text.metadata)

            pid = generate_point_id(clean_doc, page_num)
            payload = QdrantPointPayload(
                document_id=clean_doc,
                page_number=page_num,
                filename=filename or "",
                raw_text=raw_text,
                normalized_text=normalized_text,
                language=language,
                has_tables=has_tables,
                metadata=item_meta,
            )

            points.append(
                models.PointStruct(
                    id=pid,
                    vector=vector_list,
                    payload=payload.to_dict(),
                )
            )
            point_ids.append(pid)

        # Execute batched upserts
        try:
            for b_start in range(0, len(points), batch_size):
                chunk = points[b_start : b_start + batch_size]
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=chunk,
                    wait=True,
                )
            logger.info(
                f"Successfully upserted {len(points)} pages for document '{clean_doc}'."
            )
            return point_ids
        except Exception as e:
            error_msg = f"Failed to batch upsert pages for document '{clean_doc}': {e}"
            logger.error(error_msg)
            raise DocumentUpsertError(error_msg) from e

    # --------------------------------------------------------------------------
    # Read & Inspection Operations
    # --------------------------------------------------------------------------

    def get_page(self, doc_id: str, page_number: int) -> QdrantPageRecord | None:
        """Retrieve stored record for a specific document page.

        Args:
            doc_id: Unique document identifier.
            page_number: 1-indexed page number.

        Returns:
            QdrantPageRecord if found, None otherwise.
        """
        clean_doc = doc_id.strip() if doc_id else ""
        point_id = generate_point_id(clean_doc, page_number)
        try:
            records = self._client.retrieve(
                collection_name=self._collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                return None
            rec = records[0]
            p = rec.payload or {}
            return QdrantPageRecord(
                point_id=str(rec.id),
                document_id=p.get("document_id", clean_doc),
                page_number=p.get("page_number", page_number),
                filename=p.get("filename", ""),
                raw_text=p.get("raw_text", ""),
                normalized_text=p.get("normalized_text", ""),
                language=p.get("language", "ar"),
                has_tables=p.get("has_tables", False),
                metadata=p.get("metadata", {}),
                created_at=p.get("created_at", ""),
            )
        except Exception as e:
            raise QdrantStoreError(f"Failed to get page {page_number} for doc '{doc_id}': {e}") from e

    def get_document_pages(self, doc_id: str, limit: int = 1000) -> list[QdrantPageRecord]:
        """Retrieve all indexed pages for a given document, sorted by page number.

        Args:
            doc_id: Unique document identifier.
            limit: Maximum pages to retrieve.

        Returns:
            List of QdrantPageRecord objects sorted ascending by page_number.
        """
        clean_doc = doc_id.strip() if doc_id else ""
        if not clean_doc:
            return []

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=clean_doc),
                )
            ]
        )

        try:
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results: list[QdrantPageRecord] = []
            for rec in records:
                p = rec.payload or {}
                results.append(
                    QdrantPageRecord(
                        point_id=str(rec.id),
                        document_id=p.get("document_id", clean_doc),
                        page_number=int(p.get("page_number", 0)),
                        filename=p.get("filename", ""),
                        raw_text=p.get("raw_text", ""),
                        normalized_text=p.get("normalized_text", ""),
                        language=p.get("language", "ar"),
                        has_tables=p.get("has_tables", False),
                        metadata=p.get("metadata", {}),
                        created_at=p.get("created_at", ""),
                    )
                )
            results.sort(key=lambda x: x.page_number)
            return results
        except Exception as e:
            raise QdrantStoreError(f"Failed to get document pages for doc '{doc_id}': {e}") from e

    def count_pages(self, doc_id: str | None = None) -> int:
        """Count total pages indexed, optionally filtered by document_id.

        Args:
            doc_id: Optional document ID filter.

        Returns:
            Total count of page points.
        """
        try:
            count_filter: models.Filter | None = None
            if doc_id is not None:
                clean_doc = doc_id.strip()
                count_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=clean_doc),
                        )
                    ]
                )
            res = self._client.count(
                collection_name=self._collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(res.count)
        except Exception as e:
            raise QdrantStoreError(f"Failed to count pages in collection: {e}") from e

    def document_exists(self, doc_id: str) -> bool:
        """Check whether at least one page exists for a given document_id."""
        return self.count_pages(doc_id=doc_id) > 0

    def list_document_ids(self, limit: int = 1000) -> list[str]:
        """List distinct document IDs present in the collection.

        Args:
            limit: Maximum unique document IDs to return.

        Returns:
            List of unique document ID strings.
        """
        try:
            seen: set[str] = set()
            ordered: list[str] = []
            offset: Any = None

            while len(ordered) < limit:
                batch_limit = min(250, limit - len(ordered))
                records, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=None,
                    limit=batch_limit,
                    offset=offset,
                    with_payload=["document_id"],
                    with_vectors=False,
                )
                for rec in records:
                    if rec.payload and "document_id" in rec.payload:
                        d_id = str(rec.payload["document_id"]).strip()
                        if d_id and d_id not in seen:
                            seen.add(d_id)
                            ordered.append(d_id)
                            if len(ordered) >= limit:
                                break

                if next_offset is None or not records:
                    break
                offset = next_offset

            return ordered
        except Exception as e:
            raise QdrantStoreError(f"Failed to list document IDs: {e}") from e

    # --------------------------------------------------------------------------
    # Delete Operations
    # --------------------------------------------------------------------------

    def delete_page(self, doc_id: str, page_number: int) -> bool:
        """Delete a single page point from the collection.

        Args:
            doc_id: Unique document identifier.
            page_number: 1-indexed page number.

        Returns:
            True if deletion was accepted.
        """
        clean_doc = doc_id.strip() if doc_id else ""
        point_id = generate_point_id(clean_doc, page_number)
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=[point_id],
                wait=True,
            )
            logger.info(f"Deleted page {page_number} for doc '{clean_doc}'.")
            return True
        except Exception as e:
            raise DocumentUpsertError(
                f"Failed to delete page {page_number} for doc '{doc_id}': {e}"
            ) from e

    def delete_document(self, doc_id: str) -> bool:
        """Delete all points associated with a document_id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            True if deletion was accepted.
        """
        clean_doc = doc_id.strip() if doc_id else ""
        if not clean_doc:
            raise DocumentUpsertError("Cannot delete with empty document_id.")

        try:
            del_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=clean_doc),
                    )
                ]
            )
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=del_filter,
                wait=True,
            )
            logger.info(f"Deleted all points for document '{clean_doc}'.")
            return True
        except Exception as e:
            raise DocumentUpsertError(f"Failed to delete document '{doc_id}': {e}") from e

    # --------------------------------------------------------------------------
    # Search Operations (Late-Interaction MaxSim)
    # --------------------------------------------------------------------------

    def search(
        self,
        query: Any,
        top_k: int = 5,
        document_id: str | None = None,
        document_ids: Sequence[str] | None = None,
        page_range: tuple[int, int] | None = None,
        score_threshold: float | None = None,
        extra_filter: models.Filter | None = None,
        with_vectors: bool = False,
    ) -> list[QdrantSearchResult]:
        """Perform ColPali multi-vector late-interaction MaxSim search.

        Args:
            query: QueryEmbedding, 2D np.ndarray, or nested list of floats.
            top_k: Maximum number of ranked results to return.
            document_id: Optional exact document ID filter.
            document_ids: Optional list of document IDs to restrict search to.
            page_range: Optional inclusive (min_page, max_page) range filter.
            score_threshold: Minimum MaxSim score required for results.
            extra_filter: Additional arbitrary Qdrant Filter object.
            with_vectors: If True, return page multi-vectors in search result.

        Returns:
            List of ranked QdrantSearchResult objects.

        Raises:
            SearchError: If search execution fails.
            InvalidEmbeddingError: If query vectors are invalid.
        """
        if top_k < 1:
            raise SearchError(f"top_k must be >= 1, got {top_k}")

        query_vectors = _extract_vector_list(query)
        if len(query_vectors[0]) != self._vector_dim:
            raise InvalidEmbeddingError(
                f"Query vector dimension {len(query_vectors[0])} does not match "
                f"collection dimension {self._vector_dim}"
            )

        # Build filter conditions
        must_conditions: list[Any] = []

        if document_id is not None:
            clean_doc = document_id.strip()
            if clean_doc:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=clean_doc),
                    )
                )
        elif document_ids is not None:
            clean_ids = [d.strip() for d in document_ids if d and d.strip()]
            if clean_ids:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=clean_ids),
                    )
                )

        if page_range is not None:
            min_p, max_p = page_range
            must_conditions.append(
                models.FieldCondition(
                    key="page_number",
                    range=models.Range(gte=float(min_p), lte=float(max_p)),
                )
            )

        # Merge with extra_filter if provided
        final_filter: models.Filter | None = None
        if extra_filter is not None:
            combined_must = list(must_conditions)
            if extra_filter.must:
                combined_must.extend(extra_filter.must)
            final_filter = models.Filter(
                must=combined_must if combined_must else None,
                should=extra_filter.should,
                must_not=extra_filter.must_not,
            )
        elif must_conditions:
            final_filter = models.Filter(must=must_conditions)

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vectors,
                query_filter=final_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )

            results: list[QdrantSearchResult] = []
            for pt in response.points:
                p = pt.payload or {}
                # Extract vector if requested
                vec_data: list[list[float]] | None = None
                if with_vectors and pt.vector is not None:
                    if isinstance(pt.vector, list):
                        vec_data = pt.vector
                    elif isinstance(pt.vector, dict):
                        vec_data = list(pt.vector.values())[0]

                results.append(
                    QdrantSearchResult(
                        point_id=str(pt.id),
                        document_id=str(p.get("document_id", "")),
                        page_number=int(p.get("page_number", 0)),
                        score=float(pt.score) if pt.score is not None else 0.0,
                        raw_text=str(p.get("raw_text", "")),
                        normalized_text=str(p.get("normalized_text", "")),
                        language=str(p.get("language", "ar")),
                        has_tables=bool(p.get("has_tables", False)),
                        metadata=dict(p.get("metadata", {})),
                        filename=str(p.get("filename", "")),
                        vector=vec_data,
                    )
                )

            logger.debug(
                f"Search returned {len(results)} results (top_k={top_k}, "
                f"filter={'set' if final_filter else 'none'})"
            )
            return results

        except Exception as e:
            error_msg = f"Search query failed on collection '{self._collection_name}': {e}"
            logger.error(error_msg)
            raise SearchError(error_msg) from e
