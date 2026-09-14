"""Unit tests for app.core.retrieval.qdrant_store module."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client import models

from app.core.ingestion.colpali_embedder import PageEmbedding, QueryEmbedding
from app.core.ingestion.text_extractor import PageText
from app.core.retrieval.qdrant_store import (
    CollectionInitError,
    DocumentNotFoundError,
    DocumentUpsertError,
    InvalidEmbeddingError,
    QdrantPageRecord,
    QdrantPointPayload,
    QdrantSearchResult,
    QdrantStore,
    QdrantStoreError,
    SearchError,
    _extract_vector_list,
    generate_point_id,
)


# ------------------------------------------------------------------------------
# Helpers & Fixtures
# ------------------------------------------------------------------------------

def _dummy_vectors(num_vectors: int = 4, dim: int = 128) -> list[list[float]]:
    """Generate dummy multi-vector matrix."""
    return np.random.uniform(-1.0, 1.0, size=(num_vectors, dim)).astype(np.float32).tolist()


@pytest.fixture
def memory_store():
    """Provides an ephemeral in-memory QdrantStore instance."""
    store = QdrantStore(
        in_memory=True,
        collection_name="test_legal_docs",
        vector_dim=128,
        auto_init=True,
    )
    yield store
    store.close()


# ------------------------------------------------------------------------------
# Data Models Tests
# ------------------------------------------------------------------------------

def test_qdrant_point_payload_to_from_dict():
    """Verify QdrantPointPayload serialization and deserialization."""
    payload = QdrantPointPayload(
        document_id="contract_001",
        page_number=2,
        filename="contract.pdf",
        raw_text="البند الأول: التعاقد",
        normalized_text="البند الاول: التعاقد",
        language="ar",
        has_tables=True,
        metadata={"jurisdiction": "Egypt", "year": 2024},
    )

    data = payload.to_dict()
    assert data["document_id"] == "contract_001"
    assert data["page_number"] == 2
    assert data["filename"] == "contract.pdf"
    assert data["has_tables"] is True
    assert data["metadata"]["jurisdiction"] == "Egypt"
    assert isinstance(data["created_at"], str)

    restored = QdrantPointPayload.from_dict(data)
    assert restored.document_id == payload.document_id
    assert restored.page_number == payload.page_number
    assert restored.raw_text == payload.raw_text
    assert restored.has_tables == payload.has_tables
    assert restored.metadata == payload.metadata


def test_qdrant_search_result_to_dict():
    """Verify QdrantSearchResult properties and to_dict()."""
    res = QdrantSearchResult(
        point_id="some-uuid",
        document_id="doc_123",
        page_number=3,
        score=0.954,
        raw_text="نص القانون",
        normalized_text="نص القانون",
        language="ar",
        has_tables=False,
        metadata={"article": 45},
        filename="law.pdf",
    )
    d = res.to_dict()
    assert d["point_id"] == "some-uuid"
    assert d["document_id"] == "doc_123"
    assert d["score"] == pytest.approx(0.954)
    assert d["metadata"]["article"] == 45


# ------------------------------------------------------------------------------
# ID Generation & Vector Extraction Tests
# ------------------------------------------------------------------------------

def test_generate_point_id():
    """Verify deterministic UUID generation for page points."""
    pid1 = generate_point_id("doc_alpha", 1)
    pid2 = generate_point_id("doc_alpha", 1)
    pid3 = generate_point_id("doc_alpha", 2)
    pid4 = generate_point_id("doc_beta", 1)

    # Idempotent
    assert pid1 == pid2
    # Distinct for different pages or docs
    assert pid1 != pid3
    assert pid1 != pid4

    # Valid UUID RFC 4122
    parsed_uuid = uuid.UUID(pid1)
    assert str(parsed_uuid) == pid1


def test_extract_vector_list_valid():
    """Verify _extract_vector_list with various valid input representations."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # From numpy array
    v_arr = _extract_vector_list(arr)
    assert len(v_arr) == 2
    assert v_arr[0] == [1.0, 2.0]

    # From PageEmbedding dataclass
    page_emb = PageEmbedding(page_number=1, embeddings=arr, vector_dim=2)
    v_page = _extract_vector_list(page_emb)
    assert v_page == [[1.0, 2.0], [3.0, 4.0]]

    # From QueryEmbedding dataclass
    q_emb = QueryEmbedding(query="بحث", embeddings=arr, vector_dim=2)
    v_q = _extract_vector_list(q_emb)
    assert v_q == [[1.0, 2.0], [3.0, 4.0]]

    # From raw nested list
    raw_list = [[0.1, 0.2], [0.3, 0.4]]
    assert _extract_vector_list(raw_list) == raw_list


def test_extract_vector_list_invalid():
    """Verify _extract_vector_list error handling for invalid structures."""
    # 1D array
    with pytest.raises(InvalidEmbeddingError, match="must be 2D"):
        _extract_vector_list(np.array([1.0, 2.0]))

    # Empty list
    with pytest.raises(InvalidEmbeddingError, match="cannot be empty"):
        _extract_vector_list([])

    # 1D list instead of 2D
    with pytest.raises(InvalidEmbeddingError, match="must be a 2D sequence"):
        _extract_vector_list([1.0, 2.0])

    # Unsupported type
    with pytest.raises(InvalidEmbeddingError, match="Unsupported embedding type"):
        _extract_vector_list("invalid_string_vector")


# ------------------------------------------------------------------------------
# Collection Lifecycle Tests
# ------------------------------------------------------------------------------

def test_store_init_and_lifecycle(memory_store: QdrantStore):
    """Verify collection initialization, existence check, info, and deletion."""
    assert memory_store.collection_exists() is True
    assert memory_store.collection_name == "test_legal_docs"
    assert memory_store.vector_dim == 128

    info = memory_store.get_collection_info()
    assert "status" in info

    # Idempotent init
    memory_store.init_collection(force_recreate=False)
    assert memory_store.collection_exists() is True

    # Force recreate
    memory_store.init_collection(force_recreate=True)
    assert memory_store.collection_exists() is True

    # Delete collection
    assert memory_store.delete_collection() is True
    assert memory_store.collection_exists() is False


def test_store_init_with_custom_client():
    """Verify initializing store with an injected QdrantClient instance."""
    from qdrant_client import QdrantClient

    mock_client = MagicMock(spec=QdrantClient)
    mock_client.collection_exists.return_value = True

    store = QdrantStore(client=mock_client, collection_name="mock_col", auto_init=False)
    assert store.client is mock_client
    assert store.collection_name == "mock_col"


def test_store_init_with_url():
    """Verify initializing store with url parameter invokes QdrantClient correctly."""
    with patch("app.core.retrieval.qdrant_store.QdrantClient") as mock_qc_cls:
        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True
        mock_qc_cls.return_value = mock_instance

        store = QdrantStore(url="http://remote-qdrant:6333", auto_init=True)
        mock_qc_cls.assert_called_once_with(url="http://remote-qdrant:6333", api_key=None)
        assert store.collection_exists() is True


def test_store_context_manager():
    """Verify context manager calls close on exit."""
    with QdrantStore(in_memory=True, collection_name="ctx_test") as store:
        assert store.collection_exists() is True


# ------------------------------------------------------------------------------
# CRUD Tests: Upsert, Retrieve, Count, Delete
# ------------------------------------------------------------------------------

def test_upsert_single_page(memory_store: QdrantStore):
    """Verify upserting a single page and retrieving it."""
    vec = _dummy_vectors(num_vectors=5, dim=128)
    p_text = PageText(
        page_number=1,
        raw_text="عقد بيع ابتدائي",
        normalized_text="عقد بيع ابتدائي",
        language="ar",
        has_tables=False,
        metadata={"lawyer": "Ahmed"},
    )

    pid = memory_store.upsert_page(
        doc_id="contract_01",
        page_number=1,
        embedding=vec,
        page_text=p_text,
        metadata={"client": "Company A"},
        filename="contract_01.pdf",
    )

    assert isinstance(pid, str)
    assert memory_store.count_pages() == 1
    assert memory_store.count_pages(doc_id="contract_01") == 1
    assert memory_store.document_exists("contract_01") is True
    assert memory_store.document_exists("non_existent") is False

    # Retrieve page
    record = memory_store.get_page("contract_01", 1)
    assert record is not None
    assert record.document_id == "contract_01"
    assert record.page_number == 1
    assert record.raw_text == "عقد بيع ابتدائي"
    assert record.filename == "contract_01.pdf"
    assert record.metadata["lawyer"] == "Ahmed"
    assert record.metadata["client"] == "Company A"


def test_upsert_page_validation_errors(memory_store: QdrantStore):
    """Verify input validation errors on upsert_page."""
    vec = _dummy_vectors(num_vectors=3, dim=128)

    # Empty doc_id
    with pytest.raises(DocumentUpsertError, match="document_id cannot be empty"):
        memory_store.upsert_page(doc_id="", page_number=1, embedding=vec)

    # Invalid page number
    with pytest.raises(DocumentUpsertError, match="page_number must be >= 1"):
        memory_store.upsert_page(doc_id="doc1", page_number=0, embedding=vec)

    # Dimension mismatch (dim=64 instead of 128)
    vec_64 = _dummy_vectors(num_vectors=3, dim=64)
    with pytest.raises(InvalidEmbeddingError, match="Embedding dimension 64 does not match"):
        memory_store.upsert_page(doc_id="doc1", page_number=1, embedding=vec_64)


def test_upsert_document_pages_batch(memory_store: QdrantStore):
    """Verify batch upserting an entire multi-page document."""
    num_pages = 4
    embeddings = [
        PageEmbedding(
            page_number=i,
            embeddings=np.random.randn(3, 128).astype(np.float32),
            vector_dim=128,
        )
        for i in range(1, num_pages + 1)
    ]
    texts = [
        PageText(page_number=i, raw_text=f"صفحة {i}", normalized_text=f"صفحة {i}")
        for i in range(1, num_pages + 1)
    ]

    pids = memory_store.upsert_document_pages(
        doc_id="ruling_2024",
        page_embeddings=embeddings,
        page_texts=texts,
        metadata={"court": "Cairo Appellate Court"},
        filename="ruling.pdf",
        batch_size=2,
    )

    assert len(pids) == num_pages
    assert memory_store.count_pages(doc_id="ruling_2024") == num_pages

    # Get all pages sorted
    pages = memory_store.get_document_pages("ruling_2024")
    assert len(pages) == num_pages
    for i, p in enumerate(pages, start=1):
        assert p.page_number == i
        assert p.raw_text == f"صفحة {i}"
        assert p.metadata["court"] == "Cairo Appellate Court"


def test_upsert_document_pages_validation(memory_store: QdrantStore):
    """Verify validation on upsert_document_pages."""
    with pytest.raises(DocumentUpsertError, match="document_id cannot be empty"):
        memory_store.upsert_document_pages(doc_id=" ", page_embeddings=[[[1.0] * 128]])

    with pytest.raises(DocumentUpsertError, match="cannot be empty"):
        memory_store.upsert_document_pages(doc_id="doc1", page_embeddings=[])

    with pytest.raises(InvalidEmbeddingError, match="does not match collection dimension"):
        memory_store.upsert_document_pages(doc_id="doc1", page_embeddings=[[[1.0] * 64]])


def test_get_page_not_found(memory_store: QdrantStore):
    """Verify get_page returns None when point does not exist."""
    assert memory_store.get_page("missing_doc", 99) is None
    assert memory_store.get_document_pages("") == []


def test_list_document_ids(memory_store: QdrantStore):
    """Verify listing distinct document IDs across indexed pages."""
    vec = _dummy_vectors(2, 128)
    memory_store.upsert_page("doc_A", 1, vec)
    memory_store.upsert_page("doc_A", 2, vec)
    memory_store.upsert_page("doc_B", 1, vec)
    memory_store.upsert_page("doc_C", 1, vec)

    doc_ids = memory_store.list_document_ids()
    assert len(doc_ids) == 3
    assert set(doc_ids) == {"doc_A", "doc_B", "doc_C"}


def test_delete_page_and_document(memory_store: QdrantStore):
    """Verify deletion of specific pages and entire documents."""
    vec = _dummy_vectors(2, 128)
    memory_store.upsert_page("doc_del", 1, vec)
    memory_store.upsert_page("doc_del", 2, vec)
    memory_store.upsert_page("doc_del", 3, vec)
    assert memory_store.count_pages("doc_del") == 3

    # Delete single page
    assert memory_store.delete_page("doc_del", 2) is True
    assert memory_store.count_pages("doc_del") == 2
    assert memory_store.get_page("doc_del", 2) is None
    assert memory_store.get_page("doc_del", 1) is not None

    # Delete entire document
    assert memory_store.delete_document("doc_del") is True
    assert memory_store.count_pages("doc_del") == 0
    assert memory_store.document_exists("doc_del") is False

    with pytest.raises(DocumentUpsertError, match="Cannot delete with empty document_id"):
        memory_store.delete_document("")


# ------------------------------------------------------------------------------
# Search Operations Tests (Late-Interaction MaxSim)
# ------------------------------------------------------------------------------

def test_late_interaction_search(memory_store: QdrantStore):
    """Verify ColPali multi-vector MaxSim search returns ranked matches."""
    # Index 3 documents with distinguishable vectors
    vec_target = np.full((4, 128), 0.5, dtype=np.float32)
    vec_other = np.full((4, 128), -0.5, dtype=np.float32)

    memory_store.upsert_page(
        doc_id="doc_target",
        page_number=1,
        embedding=vec_target,
        page_text=PageText(
            page_number=1,
            raw_text="المستند المطابق لبحث المستخدم",
            normalized_text="المستند المطابق لبحث المستخدم",
        ),
    )
    memory_store.upsert_page(
        doc_id="doc_other",
        page_number=1,
        embedding=vec_other,
        page_text=PageText(
            page_number=1,
            raw_text="مستند آخر مختلف تماماً",
            normalized_text="مستند اخر مختلف تماما",
        ),
    )

    # Query with vectors similar to vec_target
    query_vec = np.full((2, 128), 0.5, dtype=np.float32)
    q_emb = QueryEmbedding(query="بحث مطابق", embeddings=query_vec, vector_dim=128)

    results = memory_store.search(query=q_emb, top_k=2, with_vectors=True)
    assert len(results) == 2
    top_result = results[0]
    assert top_result.document_id == "doc_target"
    assert top_result.page_number == 1
    assert "المستند المطابق" in top_result.raw_text
    assert top_result.score > results[1].score
    assert top_result.vector is not None


def test_search_filtering(memory_store: QdrantStore):
    """Verify search filtering by document_id, document_ids, and page_range."""
    vec = _dummy_vectors(3, 128)
    for p in range(1, 6):
        memory_store.upsert_page("doc_filter", p, vec)
    memory_store.upsert_page("doc_filter_2", 1, vec)

    query_vec = _dummy_vectors(2, 128)

    # Filter by exact document_id
    res_single = memory_store.search(query=query_vec, document_id="doc_filter_2")
    assert len(res_single) == 1
    assert res_single[0].document_id == "doc_filter_2"

    # Filter by list of document_ids
    res_multi = memory_store.search(
        query=query_vec,
        document_ids=["doc_filter_2", "non_existent"],
    )
    assert len(res_multi) == 1
    assert res_multi[0].document_id == "doc_filter_2"

    # Filter by page range (pages 2 to 4)
    res_range = memory_store.search(
        query=query_vec,
        document_id="doc_filter",
        page_range=(2, 4),
        top_k=10,
    )
    assert len(res_range) == 3
    page_nums = {r.page_number for r in res_range}
    assert page_nums == {2, 3, 4}


def test_search_with_extra_filter(memory_store: QdrantStore):
    """Verify search combining internal conditions with custom Qdrant Filter."""
    vec = _dummy_vectors(2, 128)
    p_text_ar = PageText(page_number=1, raw_text="عربي", normalized_text="عربي", language="ar")
    p_text_en = PageText(page_number=2, raw_text="English", normalized_text="English", language="en")

    memory_store.upsert_page("doc_lang", 1, vec, page_text=p_text_ar)
    memory_store.upsert_page("doc_lang", 2, vec, page_text=p_text_en)

    custom_filter = models.Filter(
        must=[models.FieldCondition(key="language", match=models.MatchValue(value="ar"))]
    )

    query_vec = _dummy_vectors(2, 128)
    results = memory_store.search(
        query=query_vec,
        document_id="doc_lang",
        extra_filter=custom_filter,
    )
    assert len(results) == 1
    assert results[0].page_number == 1
    assert results[0].language == "ar"


def test_search_validation_errors(memory_store: QdrantStore):
    """Verify error conditions on search."""
    vec = _dummy_vectors(2, 128)

    # top_k < 1
    with pytest.raises(SearchError, match="top_k must be >= 1"):
        memory_store.search(query=vec, top_k=0)

    # dimension mismatch
    vec_64 = _dummy_vectors(2, 64)
    with pytest.raises(InvalidEmbeddingError, match="Query vector dimension 64"):
        memory_store.search(query=vec_64)


# ------------------------------------------------------------------------------
# Error Handling & Edge Cases
# ------------------------------------------------------------------------------

def test_collection_init_error_handling():
    """Verify CollectionInitError is raised when underlying client fails."""
    mock_client = MagicMock()
    mock_client.collection_exists.side_effect = RuntimeError("Connection timeout")

    with pytest.raises(CollectionInitError, match="Failed to check collection existence"):
        QdrantStore(client=mock_client, auto_init=True)


def test_upsert_failure_error_handling():
    """Verify DocumentUpsertError wrapping when upsert call fails."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.upsert.side_effect = RuntimeError("Write failure")

    store = QdrantStore(client=mock_client, auto_init=False)
    vec = _dummy_vectors(2, 128)

    with pytest.raises(DocumentUpsertError, match="Failed to upsert page 1"):
        store.upsert_page("doc1", 1, vec)


def test_search_failure_error_handling():
    """Verify SearchError wrapping when query_points fails."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.query_points.side_effect = RuntimeError("Search execution failed")

    store = QdrantStore(client=mock_client, auto_init=False)
    vec = _dummy_vectors(2, 128)

    with pytest.raises(SearchError, match="Search query failed"):
        store.search(query=vec)


def test_count_and_list_failure_error_handling():
    """Verify QdrantStoreError wrapping on count or scroll errors."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.count.side_effect = RuntimeError("Count failed")
    mock_client.scroll.side_effect = RuntimeError("Scroll failed")

    store = QdrantStore(client=mock_client, auto_init=False)

    with pytest.raises(QdrantStoreError, match="Failed to count pages"):
        store.count_pages()

    with pytest.raises(QdrantStoreError, match="Failed to list document IDs"):
        store.list_document_ids()
