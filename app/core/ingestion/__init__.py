"""Document ingestion, OCR-free page parsing, and ColPali embedding."""

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
    "PDFProcessor",
    "PDFMetadata",
    "PDFPageImage",
    "PDFProcessingError",
    "PDFFileNotFoundError",
    "PDFEncryptedError",
    "PDFCorruptedError",
    "PopplerNotInstalledError",
]
