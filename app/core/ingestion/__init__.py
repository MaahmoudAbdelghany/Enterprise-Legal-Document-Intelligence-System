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
from app.core.ingestion.text_extractor import (
    EmptyImageError,
    LLMProviderError,
    NoLLMProviderConfiguredError,
    PageText,
    TextExtractionError,
    TextExtractor,
    detect_language,
    detect_tables,
    encode_image_to_base64,
    normalize_alef,
    normalize_arabic,
    normalize_arabic_digits,
    normalize_taa_marbuta,
    normalize_ya,
    remove_tashkeel,
    remove_tatweel,
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
    # Text Extraction & Arabic Normalization
    "TextExtractor",
    "PageText",
    "TextExtractionError",
    "LLMProviderError",
    "EmptyImageError",
    "NoLLMProviderConfiguredError",
    "normalize_arabic",
    "remove_tashkeel",
    "remove_tatweel",
    "normalize_alef",
    "normalize_taa_marbuta",
    "normalize_ya",
    "normalize_arabic_digits",
    "detect_language",
    "detect_tables",
    "encode_image_to_base64",
]


