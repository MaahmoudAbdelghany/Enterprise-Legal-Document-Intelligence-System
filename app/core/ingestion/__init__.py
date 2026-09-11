"""Document ingestion, OCR-free page parsing, and ColPali embedding."""

from app.core.ingestion.colpali_embedder import (
    ColPaliEmbedder,
    ColPaliEmbeddingError,
    DeviceConfigurationError,
    EmptyInputError,
    ModelNotLoadedError,
    PageEmbedding,
    QueryEmbedding,
    batch_compute_maxsim,
    compute_maxsim,
)
from app.core.ingestion.pdf_processor import (
    PDFCorruptedError,
    PDFEncryptedError,
    PDFFileNotFoundError,
    PDFMetadata,
    PDFPageImage,
    PDFProcessingError,
    PDFProcessor,
    PopplerNotInstalledError,
)

__all__ = [
    # PDF Processing
    "PDFProcessor",
    "PDFMetadata",
    "PDFPageImage",
    "PDFProcessingError",
    "PDFFileNotFoundError",
    "PDFEncryptedError",
    "PDFCorruptedError",
    "PopplerNotInstalledError",
    # ColPali Multi-Vector Embeddings
    "ColPaliEmbedder",
    "PageEmbedding",
    "QueryEmbedding",
    "ColPaliEmbeddingError",
    "ModelNotLoadedError",
    "DeviceConfigurationError",
    "EmptyInputError",
    "compute_maxsim",
    "batch_compute_maxsim",
]

