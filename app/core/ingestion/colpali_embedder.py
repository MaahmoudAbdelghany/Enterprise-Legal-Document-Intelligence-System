"""ColPali multi-vector visual embedding module for LexisGraph.

Implements OCR-free document embedding using ColPali (ColBERT-style late-interaction
vision-language models) for Arabic and multilingual legal documents.

Each document page is represented as a multi-vector matrix of shape (num_patches, vector_dim),
and each user query is embedded as a multi-vector matrix of shape (num_tokens, vector_dim).
Similarity is computed via the MaxSim (late interaction) operator:
    MaxSim(Q, D) = sum_{i in Q} max_{j in D} (Q_i . D_j)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from loguru import logger
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from app.core.ingestion.pdf_processor import PDFPageImage

from app.config import get_settings


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class ColPaliEmbeddingError(Exception):
    """Base exception for all ColPali embedding errors."""


class ModelNotLoadedError(ColPaliEmbeddingError):
    """Raised when an embedding operation is requested but the model is not loaded."""


class DeviceConfigurationError(ColPaliEmbeddingError):
    """Raised when an unsupported or unavailable computing device is requested."""


class EmptyInputError(ColPaliEmbeddingError):
    """Raised when empty or invalid input is passed to an embedding method."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PageEmbedding:
    """Multi-vector embedding representation of a single document page.

    Attributes:
        page_number: 1-indexed page number within the source document.
        embeddings: 2D numpy array of shape (num_patches, vector_dim) with float32 values.
        vector_dim: Dimensionality of each token/patch vector (typically 128 for ColPali).
    """

    page_number: int
    embeddings: np.ndarray
    vector_dim: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.embeddings, np.ndarray):
            object.__setattr__(self, "embeddings", np.asarray(self.embeddings, dtype=np.float32))
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Page embedding must be a 2D array (patches, dim), got shape {self.embeddings.shape}"
            )
        if self.embeddings.shape[1] != self.vector_dim:
            object.__setattr__(self, "vector_dim", self.embeddings.shape[1])

    @property
    def num_vectors(self) -> int:
        """Return the number of patch vectors for this page."""
        return self.embeddings.shape[0]

    def to_list(self) -> list[list[float]]:
        """Convert embeddings array to nested Python lists for JSON / Qdrant serialization."""
        return self.embeddings.tolist()


@dataclass(frozen=True)
class QueryEmbedding:
    """Multi-vector embedding representation of a natural language search query.

    Attributes:
        query: Original query text.
        embeddings: 2D numpy array of shape (num_tokens, vector_dim) with float32 values.
        vector_dim: Dimensionality of each token vector (typically 128 for ColPali).
    """

    query: str
    embeddings: np.ndarray
    vector_dim: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.embeddings, np.ndarray):
            object.__setattr__(self, "embeddings", np.asarray(self.embeddings, dtype=np.float32))
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Query embedding must be a 2D array (tokens, dim), got shape {self.embeddings.shape}"
            )
        if self.embeddings.shape[1] != self.vector_dim:
            object.__setattr__(self, "vector_dim", self.embeddings.shape[1])

    @property
    def num_vectors(self) -> int:
        """Return the number of token vectors for this query."""
        return self.embeddings.shape[0]

    def to_list(self) -> list[list[float]]:
        """Convert embeddings array to nested Python lists for Qdrant multi-vector search."""
        return self.embeddings.tolist()


# ------------------------------------------------------------------------------
# MaxSim Operator
# ------------------------------------------------------------------------------

def compute_maxsim(query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> float:
    """Compute ColBERT / ColPali late-interaction MaxSim similarity score.

    MaxSim(Q, D) = sum_{i=1}^{N_q} max_{j=1}^{N_d} (Q_i . D_j)

    Assumes unit-normalized vectors so that dot products equal cosine similarities.

    Args:
        query_embeddings: 2D array of shape (N_q, d) for query tokens.
        document_embeddings: 2D array of shape (N_d, d) for document patches.

    Returns:
        Scalar MaxSim similarity score (float).

    Raises:
        ValueError: If input arrays are not 2D or feature dimensions do not match.
    """
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got query ndim={query_embeddings.ndim}, "
            f"doc ndim={document_embeddings.ndim}"
        )
    if query_embeddings.shape[1] != document_embeddings.shape[1]:
        raise ValueError(
            f"Vector dimensions mismatch: query has {query_embeddings.shape[1]} dims, "
            f"document has {document_embeddings.shape[1]} dims"
        )

    # Compute dot product matrix: shape (N_q, N_d)
    similarity_matrix = np.matmul(query_embeddings, document_embeddings.T)

    # For each query token, find the maximum similarity with any document patch
    max_similarities = np.max(similarity_matrix, axis=1)

    # Sum across all query tokens
    return float(np.sum(max_similarities))


def batch_compute_maxsim(
    query_embedding: np.ndarray,
    documents_embeddings: Sequence[np.ndarray],
) -> list[float]:
    """Compute MaxSim scores for one query against multiple document page embeddings.

    Args:
        query_embedding: 2D array of shape (N_q, d).
        documents_embeddings: Sequence of 2D arrays, each of shape (N_d_k, d).

    Returns:
        List of MaxSim score floats matching the order of documents_embeddings.
    """
    scores: list[float] = []
    for doc_emb in documents_embeddings:
        scores.append(compute_maxsim(query_embedding, doc_emb))
    return scores


# ------------------------------------------------------------------------------
# ColPali Embedder Class
# ------------------------------------------------------------------------------

class ColPaliEmbedder:
    """Manages ColPali model lifecycle and multi-vector embedding generation.

    Provides lazy-loading, GPU/CPU auto-resolution, batch page inference, query
    embedding, and normalization for storage in Qdrant multi-vector collections.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        vector_dim: int | None = None,
        auto_load: bool = False,
        custom_encoder: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize ColPali embedder with settings or custom overrides.

        Args:
            model_name: HuggingFace model repo (default from settings).
            device: Computing device ('auto', 'cuda', 'cpu').
            batch_size: Batch size for visual page inference.
            vector_dim: Multi-vector dimensionality (default 128).
            auto_load: If True, eagerly load the model during __init__.
            custom_encoder: Optional custom callable / mock for testing or alternate backends.
        """
        settings = get_settings()
        self._model_name = model_name or settings.colpali_model_name
        self._configured_device = device or settings.colpali_device
        self._batch_size = batch_size or settings.colpali_batch_size
        self._vector_dim = vector_dim or settings.qdrant_vector_dim

        self._resolved_device = self._resolve_device(self._configured_device)
        self._model: Any = None
        self._processor: Any = None
        self._custom_encoder = custom_encoder
        self._is_loaded = False

        logger.info(
            f"Initialized ColPaliEmbedder: model='{self._model_name}', "
            f"device='{self._resolved_device}' (configured: '{self._configured_device}'), "
            f"batch_size={self._batch_size}, vector_dim={self._vector_dim}"
        )

        if auto_load:
            self.load_model()

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return configured model name."""
        return self._model_name

    @property
    def device(self) -> str:
        """Return resolved execution device ('cuda' or 'cpu')."""
        return self._resolved_device

    @property
    def batch_size(self) -> int:
        """Return batch size for page processing."""
        return self._batch_size

    @property
    def vector_dim(self) -> int:
        """Return vector dimension."""
        return self._vector_dim

    @property
    def is_loaded(self) -> bool:
        """Check whether the embedding model is loaded in memory."""
        return self._is_loaded

    # --------------------------------------------------------------------------
    # Model Lifecycle
    # --------------------------------------------------------------------------

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve device dynamically if set to 'auto'."""
        if requested_device in ("cuda", "cpu"):
            return requested_device
        if requested_device == "auto":
            return get_settings().resolved_device
        raise DeviceConfigurationError(
            f"Unsupported device '{requested_device}'. Must be 'auto', 'cuda', or 'cpu'."
        )

    def load_model(self) -> None:
        """Load ColPali model and processor into memory.

        Supports standard HuggingFace / sentence-transformers loading with
        fallback handling. If a custom_encoder was supplied during init,
        it will be used directly.
        """
        if self._is_loaded:
            logger.debug("ColPali model already loaded in memory.")
            return

        if self._custom_encoder is not None:
            logger.info("Using custom / mock encoder provided during initialization.")
            self._is_loaded = True
            return

        logger.info(
            f"Loading ColPali model '{self._model_name}' on device '{self._resolved_device}'..."
        )

        try:
            # First attempt: Try sentence_transformers.MultiVectorEncoder if available
            import sentence_transformers  # noqa: F401
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._resolved_device,
                trust_remote_code=True,
            )
            self._is_loaded = True
            logger.info(f"Successfully loaded '{self._model_name}' via SentenceTransformer.")
            return

        except Exception as st_err:
            logger.warning(
                f"SentenceTransformer load failed or unavailable ({st_err}). "
                "Attempting native ColPali / transformers load..."
            )

        try:
            # Second attempt: Try transformers AutoModel / AutoProcessor
            import torch
            from transformers import AutoModel, AutoProcessor

            torch_dtype = torch.bfloat16 if self._resolved_device == "cuda" else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                device_map=self._resolved_device,
                trust_remote_code=True,
            ).eval()

            self._is_loaded = True
            logger.info(f"Successfully loaded '{self._model_name}' via Transformers.")

        except Exception as tr_err:
            self._is_loaded = False
            error_msg = (
                f"Failed to load ColPali model '{self._model_name}' on device '{self._resolved_device}'. "
                f"Underlying error: {tr_err}. Ensure torch, transformers, and necessary runtime "
                "libraries are installed."
            )
            logger.error(error_msg)
            raise ModelNotLoadedError(error_msg) from tr_err

    def unload_model(self) -> None:
        """Unload model from memory to free VRAM/RAM."""
        import sys

        self._model = None
        self._processor = None
        self._is_loaded = False
        if "torch" in sys.modules and sys.modules["torch"] is not None:
            try:
                torch_mod = sys.modules["torch"]
                if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                    torch_mod.cuda.empty_cache()
            except Exception:
                pass
        logger.info("ColPali model unloaded from memory.")

    # --------------------------------------------------------------------------
    # Embedding Operations
    # --------------------------------------------------------------------------

    @staticmethod
    def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
        """L2-normalize 2D multi-vector array row-wise.

        Args:
            arr: Array of shape (N, d).

        Returns:
            Row-normalized array of shape (N, d).
        """
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    def embed_pages(
        self,
        pages: Sequence[Image.Image | tuple[int, Image.Image] | Any],
    ) -> list[PageEmbedding]:
        """Generate multi-vector embeddings for a sequence of document page images.

        Args:
            pages: Sequence of either PIL Images, (page_number, PIL Image) tuples,
                   or PDFPageImage objects.

        Returns:
            List of PageEmbedding objects corresponding to input pages.

        Raises:
            EmptyInputError: If pages list is empty.
            ModelNotLoadedError: If model is not loaded.
            ColPaliEmbeddingError: On inference failure.
        """
        if not pages:
            raise EmptyInputError("Cannot embed an empty list of pages.")

        if not self._is_loaded:
            self.load_model()

        # Normalize input to (page_number, PIL.Image)
        parsed_pages: list[tuple[int, Image.Image]] = []
        for idx, item in enumerate(pages, start=1):
            if isinstance(item, Image.Image):
                parsed_pages.append((idx, item))
            elif isinstance(item, tuple) and len(item) == 2:
                parsed_pages.append((int(item[0]), item[1]))
            elif hasattr(item, "page_number") and hasattr(item, "image"):
                parsed_pages.append((int(item.page_number), item.image))
            else:
                raise ColPaliEmbeddingError(
                    f"Unsupported page item type: {type(item)}. Expected PIL Image, tuple, or PDFPageImage."
                )

        results: list[PageEmbedding] = []

        # Process in batches
        for batch_start in range(0, len(parsed_pages), self._batch_size):
            batch = parsed_pages[batch_start : batch_start + self._batch_size]
            batch_images = [img for _, img in batch]
            batch_page_nums = [page_num for page_num, _ in batch]

            batch_embeddings = self._encode_images(batch_images)

            for page_num, raw_emb in zip(batch_page_nums, batch_embeddings):
                normalized = self._normalize_embeddings(raw_emb)
                results.append(
                    PageEmbedding(
                        page_number=page_num,
                        embeddings=normalized,
                        vector_dim=normalized.shape[1],
                    )
                )

        return results

    def embed_query(self, query: str) -> QueryEmbedding:
        """Generate multi-vector embedding for a natural language query.

        Args:
            query: Query string (Arabic, English, or mixed legal text).

        Returns:
            QueryEmbedding object containing (num_tokens, vector_dim) array.

        Raises:
            EmptyInputError: If query is empty or whitespace only.
            ModelNotLoadedError: If model is not loaded.
            ColPaliEmbeddingError: On inference failure.
        """
        stripped_query = query.strip() if query else ""
        if not stripped_query:
            raise EmptyInputError("Query cannot be empty or whitespace only.")

        if not self._is_loaded:
            self.load_model()

        raw_emb = self._encode_queries([stripped_query])[0]
        normalized = self._normalize_embeddings(raw_emb)

        return QueryEmbedding(
            query=stripped_query,
            embeddings=normalized,
            vector_dim=normalized.shape[1],
        )

    def embed_queries(self, queries: Sequence[str]) -> list[QueryEmbedding]:
        """Generate multi-vector embeddings for multiple query strings.

        Args:
            queries: Sequence of query strings.

        Returns:
            List of QueryEmbedding objects.
        """
        if not queries:
            raise EmptyInputError("Queries sequence cannot be empty.")

        cleaned_queries: list[str] = []
        for q in queries:
            stripped = q.strip() if q else ""
            if not stripped:
                raise EmptyInputError("Individual query string cannot be empty or whitespace only.")
            cleaned_queries.append(stripped)

        if not self._is_loaded:
            self.load_model()

        raw_embeddings = self._encode_queries(cleaned_queries)
        results: list[QueryEmbedding] = []

        for q, raw_emb in zip(cleaned_queries, raw_embeddings):
            normalized = self._normalize_embeddings(raw_emb)
            results.append(
                QueryEmbedding(
                    query=q,
                    embeddings=normalized,
                    vector_dim=normalized.shape[1],
                )
            )

        return results

    # --------------------------------------------------------------------------
    # Internal Encoding Logic
    # --------------------------------------------------------------------------

    def _encode_images(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Internal helper to route image encoding through active backend."""
        if self._custom_encoder is not None:
            res = self._custom_encoder("images", images)
            return [np.asarray(x, dtype=np.float32) for x in res]

        if hasattr(self._model, "encode"):
            # SentenceTransformer multi-vector encoder
            outputs = self._model.encode(images, batch_size=len(images))
            return [np.asarray(x, dtype=np.float32) for x in outputs]

        if self._processor is not None and self._model is not None:
            # Native ColPali / Transformers inference
            import torch

            inputs = self._processor.process_images(images).to(self._resolved_device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # ColPali embeddings are usually in outputs.last_hidden_state or outputs[0]
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            return [t.cpu().float().numpy() for t in embeddings]

        raise ModelNotLoadedError("No valid model or encoder available for image encoding.")

    def _encode_queries(self, queries: list[str]) -> list[np.ndarray]:
        """Internal helper to route query encoding through active backend."""
        if self._custom_encoder is not None:
            res = self._custom_encoder("queries", queries)
            return [np.asarray(x, dtype=np.float32) for x in res]

        if hasattr(self._model, "encode"):
            # SentenceTransformer multi-vector encoder
            # Prepend query prefix if needed or pass directly
            outputs = self._model.encode(queries, batch_size=len(queries))
            return [np.asarray(x, dtype=np.float32) for x in outputs]

        if self._processor is not None and self._model is not None:
            # Native ColPali / Transformers inference
            import torch

            inputs = self._processor.process_queries(queries).to(self._resolved_device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            embeddings = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            return [t.cpu().float().numpy() for t in embeddings]

        raise ModelNotLoadedError("No valid model or encoder available for query encoding.")
