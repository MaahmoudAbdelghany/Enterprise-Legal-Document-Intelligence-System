"""FastAPI dependency injection providers for LexisGraph.

Provides reusable, testable dependency factories for application settings,
Qdrant multi-vector store, PDF image processor, ColPali embedding engine,
and LLM text extractor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Generator

from loguru import logger

from app.config import Settings, get_settings
from app.core.generation.chain import LegalRAGChain
from app.core.ingestion.colpali_embedder import ColPaliEmbedder
from app.core.ingestion.pdf_processor import PDFProcessor
from app.core.ingestion.text_extractor import TextExtractor
from app.core.retrieval.qdrant_store import QdrantStore

# ------------------------------------------------------------------------------
# Module-level singletons (lazily initialized)
# ------------------------------------------------------------------------------
_qdrant_store: QdrantStore | None = None
_pdf_processor: PDFProcessor | None = None
_colpali_embedder: ColPaliEmbedder | None = None
_text_extractor: TextExtractor | None = None
_legal_rag_chain: LegalRAGChain | None = None


def get_app_settings() -> Settings:
    """Return the global application settings instance."""
    return get_settings()


def get_qdrant_store() -> QdrantStore:
    """Dependency provider for QdrantStore.

    Returns the shared QdrantStore instance, creating it if not already initialized.
    """
    global _qdrant_store
    if _qdrant_store is None:
        settings = get_settings()
        logger.info(
            f"Initializing shared QdrantStore (collection='{settings.qdrant_collection_name}', "
            f"host='{settings.qdrant_host}:{settings.qdrant_port}')"
        )
        _qdrant_store = QdrantStore(
            collection_name=settings.qdrant_collection_name,
            vector_dim=settings.qdrant_vector_dim,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            auto_init=True,
        )
    return _qdrant_store


def get_pdf_processor() -> PDFProcessor:
    """Dependency provider for PDFProcessor.

    Returns the shared PDFProcessor instance configured with application DPI
    and poppler paths.
    """
    global _pdf_processor
    if _pdf_processor is None:
        settings = get_settings()
        logger.info(f"Initializing shared PDFProcessor (dpi={settings.colpali_dpi})")
        _pdf_processor = PDFProcessor(
            dpi=settings.colpali_dpi,
            poppler_path=settings.poppler_path,
        )
    return _pdf_processor


def get_colpali_embedder() -> ColPaliEmbedder:
    """Dependency provider for ColPaliEmbedder.

    Returns the shared ColPaliEmbedder instance. The underlying model weights
    are loaded lazily upon first inference request unless explicitly loaded.
    """
    global _colpali_embedder
    if _colpali_embedder is None:
        settings = get_settings()
        logger.info(
            f"Initializing shared ColPaliEmbedder (model='{settings.colpali_model_name}', "
            f"device='{settings.colpali_device}')"
        )
        _colpali_embedder = ColPaliEmbedder(
            model_name=settings.colpali_model_name,
            device=settings.colpali_device,
            batch_size=settings.colpali_batch_size,
            vector_dim=settings.qdrant_vector_dim,
            auto_load=False,
        )
    return _colpali_embedder


def get_text_extractor() -> TextExtractor:
    """Dependency provider for TextExtractor.

    Returns the shared TextExtractor instance configured with Groq and OpenAI
    credentials from settings.
    """
    global _text_extractor
    if _text_extractor is None:
        settings = get_settings()
        logger.info(
            f"Initializing shared TextExtractor (groq_model='{settings.groq_vision_model}', "
            f"openai_fallback='{settings.openai_model}')"
        )
        _text_extractor = TextExtractor(
            settings=settings,
            groq_api_key=settings.groq_api_key or None,
            groq_vision_model=settings.groq_vision_model,
            openai_api_key=settings.openai_api_key or None,
            openai_model=settings.openai_model,
            temperature=settings.groq_temperature,
            max_retries=settings.groq_max_retries,
        )
    return _text_extractor


def get_legal_rag_chain() -> LegalRAGChain:
    """Dependency provider for LegalRAGChain.

    Returns the shared LegalRAGChain instance configured with application settings.
    """
    global _legal_rag_chain
    if _legal_rag_chain is None:
        settings = get_settings()
        logger.info(
            f"Initializing shared LegalRAGChain (groq_model='{settings.groq_model}', "
            f"openai_fallback='{settings.openai_model}')"
        )
        _legal_rag_chain = LegalRAGChain(settings=settings)
    return _legal_rag_chain


def reset_dependencies() -> None:
    """Reset all cached dependency singletons.

    Primarily used in unit tests to ensure test isolation and clean teardown.
    """
    global _qdrant_store, _pdf_processor, _colpali_embedder, _text_extractor, _legal_rag_chain

    if _qdrant_store is not None:
        try:
            _qdrant_store.close()
        except Exception as e:
            logger.debug(f"Error closing QdrantStore during reset: {e}")
        _qdrant_store = None

    _pdf_processor = None
    _colpali_embedder = None
    _text_extractor = None
    _legal_rag_chain = None
    logger.debug("LexisGraph API dependencies reset.")
