"""Unit tests for app.core.ingestion.pdf_processor."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pypdf import PdfWriter

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
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFSyntaxError


# ------------------------------------------------------------------------------
# Fixtures & Helpers
# ------------------------------------------------------------------------------

def _create_sample_pdf(num_pages: int = 3, title: str = "عقد تجاري", author: str = "شركة التقنية") -> bytes:
    """Helper to generate an in-memory PDF with specified number of blank pages."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    writer.add_metadata({
        "/Title": title,
        "/Author": author,
        "/Creator": "LexisGraph Test Suite",
    })
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _create_encrypted_pdf() -> bytes:
    """Helper to generate an encrypted in-memory PDF."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret_password")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_pdf_processor_init_defaults():
    """Verify processor initializes with settings defaults."""
    processor = PDFProcessor()
    assert processor.dpi == 200
    assert processor.thread_count == 4


def test_pdf_processor_init_custom():
    """Verify processor initializes with custom arguments."""
    processor = PDFProcessor(dpi=300, poppler_path="/usr/bin", thread_count=2)
    assert processor.dpi == 300
    assert processor.poppler_path == "/usr/bin"
    assert processor.thread_count == 2


def test_get_metadata_from_bytes():
    """Verify metadata extraction from raw PDF bytes."""
    pdf_bytes = _create_sample_pdf(num_pages=3, title="عقد توريد", author="المحامي أحمد")
    processor = PDFProcessor()
    meta = processor.get_metadata(pdf_bytes)

    assert isinstance(meta, PDFMetadata)
    assert meta.total_pages == 3
    assert meta.title == "عقد توريد"
    assert meta.author == "المحامي أحمد"
    assert not meta.is_encrypted
    assert meta.file_size_bytes == len(pdf_bytes)


def test_get_metadata_from_file(tmp_path: Path):
    """Verify metadata extraction from a physical file path."""
    pdf_bytes = _create_sample_pdf(num_pages=2)
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(pdf_bytes)

    processor = PDFProcessor()
    meta = processor.get_metadata(pdf_file)
    assert meta.total_pages == 2
    assert meta.file_size_bytes == len(pdf_bytes)


def test_get_metadata_file_not_found(tmp_path: Path):
    """Verify PDFFileNotFoundError when file does not exist."""
    processor = PDFProcessor()
    with pytest.raises(PDFFileNotFoundError):
        processor.get_metadata(tmp_path / "non_existent.pdf")


def test_get_metadata_corrupted_data():
    """Verify PDFCorruptedError when PDF bytes are corrupted."""
    corrupted_bytes = b"NOT_A_VALID_PDF_HEADER_DATA"
    processor = PDFProcessor()
    with pytest.raises(PDFCorruptedError):
        processor.get_metadata(corrupted_bytes)


def test_get_metadata_encrypted():
    """Verify metadata detects encryption."""
    enc_bytes = _create_encrypted_pdf()
    processor = PDFProcessor()
    meta = processor.get_metadata(enc_bytes)
    assert meta.is_encrypted is True


def test_convert_to_images_from_bytes():
    """Verify converting PDF bytes to PIL image tuples with mocked pdf2image."""
    pdf_bytes = _create_sample_pdf(num_pages=2)
    dummy_img1 = Image.new("RGB", (100, 100), color="white")
    dummy_img2 = Image.new("RGB", (100, 100), color="blue")

    with patch("app.core.ingestion.pdf_processor.convert_from_bytes", return_value=[dummy_img1, dummy_img2]) as mock_conv:
        processor = PDFProcessor(dpi=150)
        results = processor.convert_to_images(pdf_bytes)

        assert len(results) == 2
        assert results[0][0] == 1
        assert results[0][1] == dummy_img1
        assert results[1][0] == 2
        assert results[1][1] == dummy_img2
        mock_conv.assert_called_once_with(
            pdf_file=pdf_bytes,
            dpi=150,
            first_page=1,
            last_page=2,
            poppler_path=None,
            thread_count=4,
        )


def test_convert_to_images_from_file(tmp_path: Path):
    """Verify converting PDF file path using convert_from_path."""
    pdf_bytes = _create_sample_pdf(num_pages=1)
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(pdf_bytes)

    dummy_img = Image.new("RGB", (50, 50))
    with patch("app.core.ingestion.pdf_processor.convert_from_path", return_value=[dummy_img]) as mock_conv:
        processor = PDFProcessor()
        results = processor.convert_to_images(pdf_file)

        assert len(results) == 1
        assert results[0] == (1, dummy_img)
        mock_conv.assert_called_once()


def test_convert_to_images_page_range():
    """Verify converting a specific page subset."""
    pdf_bytes = _create_sample_pdf(num_pages=5)
    dummy_imgs = [Image.new("RGB", (10, 10)) for _ in range(2)]

    with patch("app.core.ingestion.pdf_processor.convert_from_bytes", return_value=dummy_imgs):
        processor = PDFProcessor()
        results = processor.convert_to_images(pdf_bytes, first_page=2, last_page=3)

        assert len(results) == 2
        assert results[0][0] == 2
        assert results[1][0] == 3


def test_convert_to_images_invalid_page_range():
    """Verify empty list when start page exceeds end page."""
    pdf_bytes = _create_sample_pdf(num_pages=3)
    processor = PDFProcessor()
    results = processor.convert_to_images(pdf_bytes, first_page=3, last_page=1)
    assert results == []


def test_convert_to_images_encrypted_raises():
    """Verify encrypted PDF raises PDFEncryptedError."""
    enc_bytes = _create_encrypted_pdf()
    processor = PDFProcessor()
    with pytest.raises(PDFEncryptedError):
        processor.convert_to_images(enc_bytes)


def test_convert_to_images_poppler_missing_raises():
    """Verify PopplerNotInstalledError is raised when poppler binary is absent."""
    pdf_bytes = _create_sample_pdf(num_pages=1)

    with patch(
        "app.core.ingestion.pdf_processor.convert_from_bytes",
        side_effect=PDFInfoNotInstalledError("pdftoppm not found"),
    ):
        processor = PDFProcessor()
        with pytest.raises(PopplerNotInstalledError) as exc_info:
            processor.convert_to_images(pdf_bytes)
        assert "Poppler is not installed" in str(exc_info.value)


def test_convert_to_images_syntax_error_raises():
    """Verify PDFSyntaxError converts to PDFCorruptedError."""
    pdf_bytes = _create_sample_pdf(num_pages=1)

    with patch(
        "app.core.ingestion.pdf_processor.convert_from_bytes",
        side_effect=PDFSyntaxError("Malformed stream"),
    ):
        processor = PDFProcessor()
        with pytest.raises(PDFCorruptedError):
            processor.convert_to_images(pdf_bytes)


def test_iter_page_images_lazy_batching():
    """Verify lazy batching yields expected pages in chunks."""
    pdf_bytes = _create_sample_pdf(num_pages=4)

    def fake_render(source, start, end):
        return [(p, Image.new("RGB", (10, 10))) for p in range(start, end + 1)]

    processor = PDFProcessor()
    with patch.object(processor, "_render_range", side_effect=fake_render) as mock_render:
        generator = processor.iter_page_images(pdf_bytes, batch_size=2)
        results = list(generator)

        assert len(results) == 4
        assert [p for p, _ in results] == [1, 2, 3, 4]
        assert mock_render.call_count == 2
        mock_render.assert_any_call(pdf_bytes, 1, 2)
        mock_render.assert_any_call(pdf_bytes, 3, 4)


def test_iter_page_images_encrypted_raises():
    """Verify encrypted PDF in lazy iterator raises PDFEncryptedError."""
    enc_bytes = _create_encrypted_pdf()
    processor = PDFProcessor()
    with pytest.raises(PDFEncryptedError):
        next(processor.iter_page_images(enc_bytes))


def test_save_page_images(tmp_path: Path):
    """Verify saving rendered pages to disk."""
    img1 = Image.new("RGB", (20, 20), color="red")
    img2 = Image.new("RGB", (20, 20), color="green")
    pages = [(1, img1), (2, img2)]

    out_dir = tmp_path / "output_pages"
    processor = PDFProcessor()
    saved = processor.save_page_images(pages, output_dir=out_dir, image_format="PNG", prefix="test_page")

    assert len(saved) == 2
    assert saved[0].exists()
    assert saved[0].name == "test_page_0001.png"
    assert saved[1].exists()
    assert saved[1].name == "test_page_0002.png"


def test_unsupported_source_type():
    """Verify error on unsupported PDF input types."""
    processor = PDFProcessor()
    with pytest.raises(PDFProcessingError):
        processor.get_metadata(12345)  # type: ignore[arg-type]


def test_pdf_page_image_dataclass():
    """Verify PDFPageImage dataclass attributes."""
    img = Image.new("RGB", (100, 200))
    page_img = PDFPageImage(page_number=1, image=img, width=100, height=200, dpi=200)
    assert page_img.page_number == 1
    assert page_img.width == 100
    assert page_img.height == 200
    assert page_img.dpi == 200
