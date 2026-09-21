"""Document management and ingestion API routes for LexisGraph.

Provides REST endpoints for:
- Uploading Arabic and multilingual legal PDFs (synchronous or background task)
- Listing indexed documents with pagination
- Retrieving detailed document metadata and page records
- Deleting documents from vector and knowledge graph stores
- Inspecting specific page contents and background ingestion statuses
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Path as FastPath,
    Query,
    UploadFile,
    status,
)
from loguru import logger
from pypdf import PdfReader

from app.api.dependencies import (
    get_app_settings,
    get_colpali_embedder,
    get_pdf_processor,
    get_qdrant_store,
    get_text_extractor,
)
from app.config import Settings
from app.core.ingestion.colpali_embedder import ColPaliEmbedder
from app.core.ingestion.pdf_processor import (
    PDFEncryptedError,
    PDFProcessingError,
    PDFProcessor,
)
from app.core.ingestion.text_extractor import (
    NoLLMProviderConfiguredError,
    PageText,
    TextExtractionError,
    TextExtractor,
    normalize_arabic,
)
from app.core.retrieval.qdrant_store import (
    DocumentNotFoundError,
    DocumentUpsertError,
    QdrantPageRecord,
    QdrantStore,
    QdrantStoreError,
)
from app.models.schemas import (
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
    HTTPErrorResponse,
    PageMetadata,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory registry for tracking background ingestion lifecycle statuses
_DOCUMENT_STATUS: dict[str, dict[str, Any]] = {}


def _parse_list_field(val: str | None) -> list[str]:
    """Parse comma-separated or JSON list strings into a clean list of strings."""
    if not val or not val.strip():
        return []
    clean = val.strip()
    if clean.startswith("[") and clean.endswith("]"):
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    return [item.strip() for item in clean.split(",") if item.strip()]


def _parse_dict_field(val: str | None) -> dict[str, Any]:
    """Parse JSON string into a dictionary."""
    if not val or not val.strip():
        return {}
    try:
        parsed = json.loads(val.strip())
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to parse JSON metadata: {e}")
        return {}


def _extract_fallback_text(pdf_bytes: bytes, total_pages: int) -> list[PageText]:
    """Extract page text using pypdf as fallback when vision LLM is unconfigured."""
    pages_text: list[PageText] = []
    try:
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        for idx in range(total_pages):
            page_num = idx + 1
            raw = ""
            if idx < len(reader.pages):
                try:
                    raw = reader.pages[idx].extract_text() or ""
                except Exception:
                    raw = ""
            norm = normalize_arabic(raw)
            pages_text.append(
                PageText(
                    page_number=page_num,
                    raw_text=raw,
                    normalized_text=norm,
                    language="ar",
                    has_tables=False,
                    metadata={"extractor": "pypdf_fallback"},
                )
            )
    except Exception as e:
        logger.warning(f"pypdf text extraction fallback failed: {e}")
        for idx in range(total_pages):
            pages_text.append(
                PageText(
                    page_number=idx + 1,
                    raw_text="",
                    normalized_text="",
                    language="ar",
                    has_tables=False,
                    metadata={"extractor": "empty"},
                )
            )
    return pages_text


def _process_and_index_pipeline(
    doc_id: str,
    filename: str,
    pdf_bytes: bytes,
    legal_category: str | None,
    parties: list[str],
    tags: list[str],
    custom_metadata: dict[str, Any],
    pdf_processor: PDFProcessor,
    colpali_embedder: ColPaliEmbedder,
    text_extractor: TextExtractor,
    qdrant_store: QdrantStore,
) -> int:
    """Core synchronous processing pipeline for a legal PDF document.

    Converts pages to images, extracts embeddings and text, and indexes in Qdrant.
    Returns total pages indexed.
    """
    logger.info(f"Starting ingestion pipeline for doc '{doc_id}' ({filename})...")
    _DOCUMENT_STATUS[doc_id] = {
        "status": DocumentStatus.PROCESSING,
        "filename": filename,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "error": None,
    }

    try:
        # Step 1: Convert PDF to images
        page_images = pdf_processor.convert_to_images(pdf_bytes)
        if not page_images:
            raise PDFProcessingError(f"PDF '{filename}' produced 0 renderable pages.")

        page_count = len(page_images)
        logger.info(f"Rendered {page_count} page images for doc '{doc_id}'.")

        # Step 2: Extract text (Vision LLM or fallback)
        page_texts: list[PageText] = []
        try:
            if text_extractor.has_active_provider:
                # Use LLM vision text extraction
                page_texts = text_extractor.extract_document_text(page_images)
            else:
                logger.info(
                    f"No vision LLM provider configured. Using pypdf fallback for doc '{doc_id}'."
                )
                page_texts = _extract_fallback_text(pdf_bytes, page_count)
        except (NoLLMProviderConfiguredError, TextExtractionError) as te:
            logger.warning(
                f"Vision text extraction failed ({te}). Falling back to pypdf for doc '{doc_id}'."
            )
            page_texts = _extract_fallback_text(pdf_bytes, page_count)

        # Step 3: Compute ColPali multi-vector embeddings
        logger.info(f"Generating ColPali embeddings for {page_count} pages (doc '{doc_id}')...")
        page_embeddings = colpali_embedder.embed_pages(page_images)

        # Step 4: Index into Qdrant multi-vector collection
        doc_metadata: dict[str, Any] = {
            "legal_category": legal_category,
            "parties": parties,
            "tags": tags,
            **custom_metadata,
        }

        qdrant_store.upsert_document_pages(
            doc_id=doc_id,
            page_embeddings=page_embeddings,
            page_texts=page_texts,
            metadata=doc_metadata,
            filename=filename,
        )

        _DOCUMENT_STATUS[doc_id] = {
            "status": DocumentStatus.COMPLETED,
            "filename": filename,
            "page_count": page_count,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": None,
        }
        logger.info(f"Successfully completed ingestion for doc '{doc_id}' ({page_count} pages).")
        return page_count

    except Exception as exc:
        err_str = str(exc)
        logger.error(f"Ingestion failed for doc '{doc_id}': {err_str}")
        _DOCUMENT_STATUS[doc_id] = {
            "status": DocumentStatus.FAILED,
            "filename": filename,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": err_str,
        }
        raise


# ==============================================================================
# Endpoint 1: Upload Document
# ==============================================================================
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a legal PDF document",
    description=(
        "Uploads a PDF document, converts pages to images, computes ColPali late-interaction "
        "multi-vector visual embeddings, extracts normalized Arabic text, and indexes pages into Qdrant. "
        "Can run synchronously or asynchronously via BackgroundTasks."
    ),
    responses={
        400: {"model": HTTPErrorResponse, "description": "Invalid file format or corrupted PDF"},
        500: {"model": HTTPErrorResponse, "description": "Ingestion pipeline failure"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Legal PDF document file"),
    document_id: str | None = Form(default=None, description="Optional custom unique document ID"),
    legal_category: str | None = Form(default=None, description="Legal category (e.g., 'commercial_contract')"),
    parties: str | None = Form(default=None, description="Comma-separated or JSON list of legal parties"),
    tags: str | None = Form(default=None, description="Comma-separated or JSON list of tags"),
    metadata_json: str | None = Form(default=None, description="Optional JSON string of additional metadata"),
    background: bool = Form(default=False, description="If True, process ingestion asynchronously in background"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    settings: Settings = Depends(get_app_settings),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    colpali_embedder: ColPaliEmbedder = Depends(get_colpali_embedder),
    text_extractor: TextExtractor = Depends(get_text_extractor),
) -> DocumentUploadResponse:
    """Upload, process, embed, and index a legal PDF document."""
    # 1. Validate file format
    original_filename = file.filename or "uploaded_document.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Received: '{original_filename}'",
        )

    # 2. Read file contents into memory
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file content: {e}",
        ) from e

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # Check PDF magic bytes (%PDF-)
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file header. The uploaded file is not a valid PDF document.",
        )

    # 3. Resolve document_id
    doc_id = document_id.strip() if document_id and document_id.strip() else str(uuid.uuid4())

    # 4. Save file to storage location
    storage_dir = Path(settings.local_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    saved_path = storage_dir / f"{doc_id}.pdf"
    try:
        saved_path.write_bytes(content)
        logger.info(f"Saved uploaded document to {saved_path}")
    except Exception as e:
        logger.error(f"Failed to persist file to disk: {e}")
        # Proceed with in-memory bytes even if disk save encounters issue

    # 5. Parse metadata fields
    parsed_parties = _parse_list_field(parties)
    parsed_tags = _parse_list_field(tags)
    parsed_meta = _parse_dict_field(metadata_json)

    # 6. Branch: Background vs. Synchronous execution
    if background:
        _DOCUMENT_STATUS[doc_id] = {
            "status": DocumentStatus.PENDING,
            "filename": original_filename,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "error": None,
        }

        background_tasks.add_task(
            _process_and_index_pipeline,
            doc_id=doc_id,
            filename=original_filename,
            pdf_bytes=content,
            legal_category=legal_category,
            parties=parsed_parties,
            tags=parsed_tags,
            custom_metadata=parsed_meta,
            pdf_processor=pdf_processor,
            colpali_embedder=colpali_embedder,
            text_extractor=text_extractor,
            qdrant_store=qdrant_store,
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=original_filename,
            page_count=0,
            status=DocumentStatus.PENDING,
            message="Document accepted. Ingestion scheduled in background.",
            metadata={
                "legal_category": legal_category,
                "parties": parsed_parties,
                "tags": parsed_tags,
                **parsed_meta,
            },
        )

    # Synchronous processing
    try:
        page_count = _process_and_index_pipeline(
            doc_id=doc_id,
            filename=original_filename,
            pdf_bytes=content,
            legal_category=legal_category,
            parties=parsed_parties,
            tags=parsed_tags,
            custom_metadata=parsed_meta,
            pdf_processor=pdf_processor,
            colpali_embedder=colpali_embedder,
            text_extractor=text_extractor,
            qdrant_store=qdrant_store,
        )
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=original_filename,
            page_count=page_count,
            status=DocumentStatus.COMPLETED,
            message="Document uploaded, embedded, and indexed successfully.",
            metadata={
                "legal_category": legal_category,
                "parties": parsed_parties,
                "tags": parsed_tags,
                **parsed_meta,
            },
        )
    except PDFEncryptedError as enc_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PDF is encrypted or password protected: {enc_err}",
        ) from enc_err
    except (PDFProcessingError, DocumentUpsertError) as proc_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {proc_err}",
        ) from proc_err
    except Exception as exc:
        logger.exception(f"Unexpected error during document ingestion: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal ingestion pipeline error: {exc}",
        ) from exc


# ==============================================================================
# Endpoint 2: List Documents
# ==============================================================================
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed legal documents",
    description="Returns a paginated list of documents currently stored and indexed in Qdrant.",
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip"),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentListResponse:
    """Retrieve paginated list of all indexed documents."""
    try:
        all_doc_ids = qdrant_store.list_document_ids(limit=1000)
        total_count = len(all_doc_ids)

        page_ids = all_doc_ids[offset : offset + limit]
        documents: list[DocumentMetadata] = []

        for d_id in page_ids:
            pages = qdrant_store.get_document_pages(d_id, limit=1)
            page_count = qdrant_store.count_pages(d_id)

            filename = f"{d_id}.pdf"
            meta: dict[str, Any] = {}
            created_at = datetime.datetime.now(datetime.timezone.utc)

            if pages:
                first_page = pages[0]
                if first_page.filename:
                    filename = first_page.filename
                meta = dict(first_page.metadata)
                if first_page.created_at:
                    try:
                        created_at = datetime.datetime.fromisoformat(first_page.created_at)
                    except Exception:
                        pass

            documents.append(
                DocumentMetadata(
                    document_id=d_id,
                    filename=filename,
                    page_count=page_count,
                    status=DocumentStatus.COMPLETED,
                    created_at=created_at,
                    legal_category=meta.get("legal_category"),
                    parties=meta.get("parties", []),
                    tags=meta.get("tags", []),
                    metadata=meta,
                )
            )

        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document list: {e}",
        ) from e


# ==============================================================================
# Endpoint 3: Document Detail
# ==============================================================================
@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and pages",
    description="Retrieves comprehensive metadata and per-page records for a specific document ID.",
    responses={
        404: {"model": HTTPErrorResponse, "description": "Document not found"},
    },
)
async def get_document(
    document_id: str = FastPath(..., description="Unique document ID"),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentDetailResponse:
    """Retrieve full details and page breakdowns for a specific document."""
    try:
        pages = qdrant_store.get_document_pages(document_id)

        # Check background status if not yet in Qdrant
        if not pages:
            if document_id in _DOCUMENT_STATUS:
                bg_info = _DOCUMENT_STATUS[document_id]
                bg_status = bg_info.get("status", DocumentStatus.PENDING)
                return DocumentDetailResponse(
                    document=DocumentMetadata(
                        document_id=document_id,
                        filename=bg_info.get("filename", f"{document_id}.pdf"),
                        page_count=bg_info.get("page_count", 0),
                        status=bg_status,
                    ),
                    pages=[],
                )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        first_page = pages[0]
        filename = first_page.filename or f"{document_id}.pdf"
        meta = dict(first_page.metadata)
        created_at = datetime.datetime.now(datetime.timezone.utc)
        if first_page.created_at:
            try:
                created_at = datetime.datetime.fromisoformat(first_page.created_at)
            except Exception:
                pass

        doc_meta = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            status=DocumentStatus.COMPLETED,
            created_at=created_at,
            legal_category=meta.get("legal_category"),
            parties=meta.get("parties", []),
            tags=meta.get("tags", []),
            metadata=meta,
        )

        pages_meta: list[PageMetadata] = []
        for p in pages:
            preview = p.raw_text[:200] if p.raw_text else ""
            pages_meta.append(
                PageMetadata(
                    page_number=p.page_number,
                    language=p.language,
                    has_tables=p.has_tables,
                    character_count=len(p.raw_text),
                    preview_text=preview,
                )
            )

        return DocumentDetailResponse(
            document=doc_meta,
            pages=pages_meta,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document '{document_id}': {e}",
        ) from e


# ==============================================================================
# Endpoint 4: Delete Document
# ==============================================================================
@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its vectors",
    description="Removes all vector points and metadata associated with the document from Qdrant and local disk.",
    responses={
        404: {"model": HTTPErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    document_id: str = FastPath(..., description="Unique document ID to delete"),
    settings: Settings = Depends(get_app_settings),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> DocumentDeleteResponse:
    """Delete a document from Qdrant vector store and local storage."""
    try:
        exists = qdrant_store.document_exists(document_id)
        if not exists and document_id not in _DOCUMENT_STATUS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        points_count = qdrant_store.count_pages(document_id)
        if exists:
            qdrant_store.delete_document(document_id)

        # Remove physical file if present
        local_file = Path(settings.local_storage_dir) / f"{document_id}.pdf"
        try:
            if local_file.exists():
                local_file.unlink()
                logger.info(f"Removed local file {local_file}")
        except Exception as file_err:
            logger.warning(f"Could not remove local file {local_file}: {file_err}")

        # Clean background status entry
        _DOCUMENT_STATUS.pop(document_id, None)

        return DocumentDeleteResponse(
            document_id=document_id,
            deleted=True,
            message=f"Document '{document_id}' deleted successfully.",
            qdrant_points_deleted=points_count,
            graph_nodes_deleted=0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document '{document_id}': {e}",
        ) from e


# ==============================================================================
# Endpoint 5: Page Content Detail
# ==============================================================================
@router.get(
    "/{document_id}/pages/{page_number}",
    summary="Get single page content and metadata",
    description="Retrieves the full extracted verbatim text and metadata for an individual page.",
    responses={
        404: {"model": HTTPErrorResponse, "description": "Page or document not found"},
    },
)
async def get_document_page(
    document_id: str = FastPath(..., description="Unique document ID"),
    page_number: int = FastPath(..., ge=1, description="1-indexed page number"),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> dict[str, Any]:
    """Retrieve full verbatim text and metadata for a single document page."""
    try:
        page = qdrant_store.get_page(document_id, page_number)
        if page is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Page {page_number} for document '{document_id}' not found.",
            )
        return {
            "document_id": page.document_id,
            "page_number": page.page_number,
            "filename": page.filename,
            "raw_text": page.raw_text,
            "normalized_text": page.normalized_text,
            "language": page.language,
            "has_tables": page.has_tables,
            "metadata": page.metadata,
            "created_at": page.created_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get page {page_number} for doc '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve page: {e}",
        ) from e


# ==============================================================================
# Endpoint 6: Ingestion Status
# ==============================================================================
@router.get(
    "/{document_id}/status",
    summary="Check document ingestion status",
    description="Returns the processing lifecycle status of a document (especially for background uploads).",
)
async def get_document_status(
    document_id: str = FastPath(..., description="Unique document ID"),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
) -> dict[str, Any]:
    """Check processing status of a document."""
    if document_id in _DOCUMENT_STATUS:
        return {
            "document_id": document_id,
            **_DOCUMENT_STATUS[document_id],
        }

    if qdrant_store.document_exists(document_id):
        page_count = qdrant_store.count_pages(document_id)
        return {
            "document_id": document_id,
            "status": DocumentStatus.COMPLETED,
            "page_count": page_count,
            "error": None,
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document '{document_id}' not found.",
    )
