"""Unit tests for app.core.ingestion.colpali_embedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

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


# ------------------------------------------------------------------------------
# Fixtures & Helpers
# ------------------------------------------------------------------------------

def _create_dummy_image(width: int = 50, height: int = 50, color: str = "white") -> Image.Image:
    """Create a simple PIL RGB image for testing."""
    return Image.new("RGB", (width, height), color=color)


def _mock_encoder_fn(mode: str, items: list) -> list[np.ndarray]:
    """Mock encoding function returning deterministic multi-vector arrays."""
    dim = 128
    if mode == "images":
        # Return 10 patch vectors per image
        return [np.ones((10, dim), dtype=np.float32) for _ in items]
    elif mode == "queries":
        # Return 4 token vectors per query
        return [np.ones((4, dim), dtype=np.float32) for _ in items]
    raise ValueError(f"Unknown mode {mode}")


# ------------------------------------------------------------------------------
# Data Models Tests
# ------------------------------------------------------------------------------

def test_page_embedding_dataclass():
    """Verify PageEmbedding initialization, properties, and to_list()."""
    arr = np.random.randn(8, 128).astype(np.float32)
    page_emb = PageEmbedding(page_number=1, embeddings=arr, vector_dim=128)

    assert page_emb.page_number == 1
    assert page_emb.num_vectors == 8
    assert page_emb.vector_dim == 128
    assert isinstance(page_emb.to_list(), list)
    assert len(page_emb.to_list()) == 8
    assert len(page_emb.to_list()[0]) == 128


def test_page_embedding_invalid_shape():
    """Verify ValueError when page embeddings array is not 2D."""
    arr_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must be a 2D array"):
        PageEmbedding(page_number=1, embeddings=arr_1d)


def test_query_embedding_dataclass():
    """Verify QueryEmbedding initialization, properties, and to_list()."""
    arr = np.random.randn(5, 128).astype(np.float32)
    query_emb = QueryEmbedding(query="شروط العقد", embeddings=arr)

    assert query_emb.query == "شروط العقد"
    assert query_emb.num_vectors == 5
    assert query_emb.vector_dim == 128
    assert isinstance(query_emb.to_list(), list)
    assert len(query_emb.to_list()) == 5


def test_query_embedding_invalid_shape():
    """Verify ValueError when query embeddings array is not 2D."""
    arr_3d = np.zeros((2, 3, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="must be a 2D array"):
        QueryEmbedding(query="بحث", embeddings=arr_3d)


# ------------------------------------------------------------------------------
# MaxSim Operator Tests
# ------------------------------------------------------------------------------

def test_compute_maxsim_exact_math():
    """Verify MaxSim computation matches analytical mathematical result."""
    # Q has 2 tokens of dim 2: Q_0 = [1, 0], Q_1 = [0, 1]
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # D has 3 patches of dim 2: D_0 = [1, 0], D_1 = [0.6, 0.8], D_2 = [0, 1]
    d = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)

    # Similarities:
    # Q_0 . D = [1.0, 0.6, 0.0] -> max is 1.0
    # Q_1 . D = [0.0, 0.8, 1.0] -> max is 1.0
    # MaxSim = 1.0 + 1.0 = 2.0
    score = compute_maxsim(q, d)
    assert pytest.approx(score, rel=1e-5) == 2.0


def test_compute_maxsim_dimension_mismatch():
    """Verify ValueError when feature dimensions between query and doc do not match."""
    q = np.ones((3, 64), dtype=np.float32)
    d = np.ones((5, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="Vector dimensions mismatch"):
        compute_maxsim(q, d)


def test_compute_maxsim_invalid_ndim():
    """Verify ValueError when inputs are not 2D matrices."""
    q = np.ones((64,), dtype=np.float32)
    d = np.ones((5, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 2D arrays"):
        compute_maxsim(q, d)


def test_batch_compute_maxsim():
    """Verify batch_compute_maxsim correctly ranks multiple candidate pages."""
    q = np.array([[1.0, 0.0]], dtype=np.float32)

    # Doc 1 is orthogonal (dot product 0)
    d1 = np.array([[0.0, 1.0]], dtype=np.float32)
    # Doc 2 is partially aligned (dot product 0.5)
    d2 = np.array([[0.5, 0.5]], dtype=np.float32)
    # Doc 3 is perfectly aligned (dot product 1.0)
    d3 = np.array([[1.0, 0.0]], dtype=np.float32)

    scores = batch_compute_maxsim(q, [d1, d2, d3])
    assert len(scores) == 3
    assert pytest.approx(scores[0]) == 0.0
    assert pytest.approx(scores[1]) == 0.5
    assert pytest.approx(scores[2]) == 1.0


# ------------------------------------------------------------------------------
# Embedder Initialization & Device Resolution Tests
# ------------------------------------------------------------------------------

def test_embedder_init_defaults():
    """Verify embedder defaults inherit from application settings."""
    embedder = ColPaliEmbedder()
    assert embedder.model_name == "vidore/colpali-v1.3-hf"
    assert embedder.vector_dim == 128
    assert embedder.batch_size == 4
    assert embedder.is_loaded is False


def test_embedder_init_custom_params():
    """Verify embedder accepts custom configurations."""
    embedder = ColPaliEmbedder(
        model_name="custom/colpali-legal",
        device="cpu",
        batch_size=8,
        vector_dim=64,
    )
    assert embedder.model_name == "custom/colpali-legal"
    assert embedder.device == "cpu"
    assert embedder.batch_size == 8
    assert embedder.vector_dim == 64


def test_embedder_unsupported_device_raises():
    """Verify DeviceConfigurationError when invalid device is configured."""
    with pytest.raises(DeviceConfigurationError, match="Unsupported device 'tpu'"):
        ColPaliEmbedder(device="tpu")


# ------------------------------------------------------------------------------
# Model Loading & Unloading Tests
# ------------------------------------------------------------------------------

def test_embedder_custom_encoder_load():
    """Verify custom encoder immediately marks embedder as loaded."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    assert embedder.is_loaded is False
    embedder.load_model()
    assert embedder.is_loaded is True

    # Calling load_model again returns early
    embedder.load_model()
    assert embedder.is_loaded is True


def test_embedder_unload():
    """Verify unload_model cleans up state and marks is_loaded to False."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    embedder.load_model()
    assert embedder.is_loaded is True

    embedder.unload_model()
    assert embedder.is_loaded is False


def test_embedder_load_failure_raises():
    """Verify ModelNotLoadedError is raised when model packages fail to load."""
    embedder = ColPaliEmbedder(device="cpu")
    with patch.dict("sys.modules", {"sentence_transformers": None, "transformers": None, "torch": None}):
        with patch("app.core.ingestion.colpali_embedder.logger"):
            with pytest.raises(ModelNotLoadedError, match="Failed to load ColPali model"):
                embedder.load_model()



# ------------------------------------------------------------------------------
# Embedding Operations Tests
# ------------------------------------------------------------------------------

def test_embed_pages_empty_raises():
    """Verify EmptyInputError when embedding empty page list."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    with pytest.raises(EmptyInputError, match="Cannot embed an empty list"):
        embedder.embed_pages([])


def test_embed_pages_pil_images():
    """Verify embedding a list of raw PIL Images."""
    embedder = ColPaliEmbedder(batch_size=2, custom_encoder=_mock_encoder_fn)
    images = [_create_dummy_image(), _create_dummy_image(), _create_dummy_image()]

    results = embedder.embed_pages(images)

    assert len(results) == 3
    for idx, page_emb in enumerate(results, start=1):
        assert isinstance(page_emb, PageEmbedding)
        assert page_emb.page_number == idx
        assert page_emb.num_vectors == 10
        assert page_emb.vector_dim == 128
        # Verify L2 normalization: row norms should equal 1.0
        norms = np.linalg.norm(page_emb.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), rtol=1e-5)


def test_embed_pages_tuples():
    """Verify embedding (page_num, Image) tuples preserves page numbers."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    pages = [(5, _create_dummy_image()), (6, _create_dummy_image())]

    results = embedder.embed_pages(pages)

    assert len(results) == 2
    assert results[0].page_number == 5
    assert results[1].page_number == 6


def test_embed_pages_pdf_page_image_objects():
    """Verify embedding objects with page_number and image attributes (PDFPageImage)."""
    mock_page = MagicMock()
    mock_page.page_number = 42
    mock_page.image = _create_dummy_image()

    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    results = embedder.embed_pages([mock_page])

    assert len(results) == 1
    assert results[0].page_number == 42


def test_embed_pages_unsupported_type_raises():
    """Verify ColPaliEmbeddingError when invalid page item type is passed."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    with pytest.raises(ColPaliEmbeddingError, match="Unsupported page item type"):
        embedder.embed_pages(["not_an_image"])  # type: ignore[arg-type]


def test_embed_query():
    """Verify embedding a natural language Arabic query."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    query_str = "ما هي شروط وإجراءات فسخ العقد التجاري؟"

    q_emb = embedder.embed_query(query_str)

    assert isinstance(q_emb, QueryEmbedding)
    assert q_emb.query == query_str
    assert q_emb.num_vectors == 4
    assert q_emb.vector_dim == 128
    norms = np.linalg.norm(q_emb.embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-5)


def test_embed_query_empty_raises():
    """Verify EmptyInputError when query is empty or only whitespace."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    with pytest.raises(EmptyInputError, match="Query cannot be empty"):
        embedder.embed_query("   ")


def test_embed_queries_batch():
    """Verify batch embedding multiple queries."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    queries = ["عقد توريد بضائع", "المادة 12 من قانون العمل", "التعويض عن الضرر"]

    results = embedder.embed_queries(queries)

    assert len(results) == 3
    assert [r.query for r in results] == queries
    for r in results:
        assert r.num_vectors == 4
        assert r.vector_dim == 128


def test_embed_queries_empty_list_raises():
    """Verify EmptyInputError when empty queries list is provided."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    with pytest.raises(EmptyInputError, match="Queries sequence cannot be empty"):
        embedder.embed_queries([])


def test_embed_queries_blank_query_item_raises():
    """Verify EmptyInputError when an individual query item in list is blank."""
    embedder = ColPaliEmbedder(custom_encoder=_mock_encoder_fn)
    with pytest.raises(EmptyInputError, match="Individual query string cannot be empty"):
        embedder.embed_queries(["صحيح", "  "])
