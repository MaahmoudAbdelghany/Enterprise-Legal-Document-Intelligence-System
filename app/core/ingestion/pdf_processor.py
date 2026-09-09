"""PDF processing module for LexisGraph.

Converts Arabic and multilingual legal PDF documents into high-resolution
page images suitable for ColPali OCR-free multi-vector visual embedding,
with support for lazy batching to handle multi-hundred-page documents.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Generator, Sequence

from loguru import logger
from PIL import Image
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

try:
    from pdf2image import convert_from_bytes, convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    )
except ImportError:  # pragma: no cover
    convert_from_bytes = None  # type: ignore[assignment]
    convert_from_path = None  # type: ignore[assignment]
    PDFInfoNotInstalledError = Exception  # type: ignore[misc,assignment]
    PDFPageCountError = Exception  # type: ignore[misc,assignment]
    PDFSyntaxError = Exception  # type: ignore[misc,assignment]

from app.config import get_settings


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Base exception for all PDF processing errors."""


class PDFFileNotFoundError(PDFProcessingError, FileNotFoundError):
    """Raised when the specified PDF file cannot be found."""


class PDFEncryptedError(PDFProcessingError):
    """Raised when the PDF document is password-protected or encrypted."""


class PDFCorruptedError(PDFProcessingError):
    """Raised when the PDF document is malformed or corrupted."""


class PopplerNotInstalledError(PDFProcessingError):
    """Raised when poppler utilities (pdftoppm / pdfinfo) are missing from the system."""


# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PDFMetadata:
    """Extracted metadata and physical properties of a PDF document."""

    total_pages: int
    title: str | None = None
    author: str | None = None
    creator: str | None = None
    producer: str | None = None
    is_encrypted: bool = False
    file_size_bytes: int = 0


@dataclass(frozen=True)
class PDFPageImage:
    """A single rendered page image with associated layout metadata."""

    page_number: int  # 1-indexed
    image: Image.Image
    width: int
    height: int
    dpi: int


# ------------------------------------------------------------------------------
# PDF Processor
# ------------------------------------------------------------------------------

class PDFProcessor:
    """Converts PDF documents to PIL images for downstream visual embedding.

    Features:
    - Lazy page batching for memory-efficient handling of large legal dossiers.
    - Configurable DPI resolution (default 200).
    - Metadata and encryption inspection via lightweight pypdf parsing.
    - Automatic poppler path resolution from configuration or system PATH.
    """

    def __init__(
        self,
        dpi: int | None = None,
        poppler_path: str | Path | None = None,
        thread_count: int = 4,
    ) -> None:
        """Initialize PDFProcessor.

        Args:
            dpi: Rendering DPI. If None, defaults to `Settings.colpali_dpi`.
            poppler_path: Directory containing poppler binaries (pdftoppm, pdfinfo).
            thread_count: Number of worker threads for parallel page rasterization.
        """
        settings = get_settings()
        self.dpi = dpi if dpi is not None else settings.colpali_dpi
        self.poppler_path = (
            str(poppler_path)
            if poppler_path is not None
            else (settings.poppler_path or None)
        )
        self.thread_count = thread_count

    # --------------------------------------------------------------------------
    # Metadata Inspection
    # --------------------------------------------------------------------------

    def get_metadata(self, pdf_source: str | Path | bytes | BinaryIO) -> PDFMetadata:
        """Inspect PDF metadata and page count without full rasterization.

        Args:
            pdf_source: Path to PDF file, raw bytes, or file-like binary stream.

        Returns:
            PDFMetadata containing total pages, author, title, and encryption flag.

        Raises:
            PDFFileNotFoundError: If a file path is provided that does not exist.
            PDFCorruptedError: If the PDF structure is damaged or unreadable.
        """
        stream, file_size = self._get_stream_and_size(pdf_source)
        try:
            reader = PdfReader(stream)
            is_encrypted = reader.is_encrypted
            if is_encrypted:
                try:
                    # Attempt decrypt with empty password for standard open
                    reader.decrypt("")
                except Exception:
                    pass

            try:
                total_pages = len(reader.pages)
                meta = reader.metadata or {}
            except FileNotDecryptedError:
                return PDFMetadata(
                    total_pages=0,
                    is_encrypted=True,
                    file_size_bytes=file_size,
                )

            return PDFMetadata(
                total_pages=total_pages,
                title=meta.get("/Title") if hasattr(meta, "get") else None,
                author=meta.get("/Author") if hasattr(meta, "get") else None,
                creator=meta.get("/Creator") if hasattr(meta, "get") else None,
                producer=meta.get("/Producer") if hasattr(meta, "get") else None,
                is_encrypted=is_encrypted,
                file_size_bytes=file_size,
            )
        except PDFProcessingError:
            raise
        except FileNotDecryptedError:
            return PDFMetadata(
                total_pages=0,
                is_encrypted=True,
                file_size_bytes=file_size,
            )
        except PdfReadError as e:
            raise PDFCorruptedError(f"Corrupted or invalid PDF file: {e}") from e
        except Exception as e:
            raise PDFCorruptedError(f"Unexpected error reading PDF: {e}") from e

    # --------------------------------------------------------------------------
    # Conversion: Direct List
    # --------------------------------------------------------------------------

    def convert_to_images(
        self,
        pdf_source: str | Path | bytes | BinaryIO,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> list[tuple[int, Image.Image]]:
        """Convert PDF pages into a list of (page_number, PIL.Image) tuples.

        Args:
            pdf_source: Path to file, raw bytes, or binary stream.
            first_page: 1-indexed first page to render (inclusive).
            last_page: 1-indexed last page to render (inclusive).

        Returns:
            List of tuples of (1-indexed page_number, PIL.Image).

        Raises:
            PDFFileNotFoundError: If the file does not exist.
            PDFEncryptedError: If document is password-protected.
            PopplerNotInstalledError: If poppler is not found in PATH or config.
            PDFProcessingError: On rasterization failure.
        """
        # Validate metadata & check encryption
        metadata = self.get_metadata(pdf_source)
        if metadata.is_encrypted:
            raise PDFEncryptedError(
                "PDF is encrypted or password-protected and cannot be processed."
            )
        if metadata.total_pages == 0:
            return []

        start_page = max(1, first_page) if first_page else 1
        end_page = min(metadata.total_pages, last_page) if last_page else metadata.total_pages

        if start_page > end_page:
            return []

        logger.info(
            "Converting PDF pages {}..{} (total: {}) at {} DPI",
            start_page,
            end_page,
            metadata.total_pages,
            self.dpi,
        )

        return self._render_range(pdf_source, start_page, end_page)

    # --------------------------------------------------------------------------
    # Conversion: Lazy Batch Generator
    # --------------------------------------------------------------------------

    def iter_page_images(
        self,
        pdf_source: str | Path | bytes | BinaryIO,
        batch_size: int = 10,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> Generator[tuple[int, Image.Image], None, None]:
        """Lazily yield (page_number, PIL.Image) tuples in batches to save memory.

        Crucial for massive legal archives (e.g. 200+ page court records) to prevent
        out-of-memory (OOM) errors during visual embedding.

        Args:
            pdf_source: Path to file, raw bytes, or binary stream.
            batch_size: Number of pages to render per batch chunk.
            first_page: 1-indexed start page.
            last_page: 1-indexed end page.

        Yields:
            (page_number, PIL.Image) tuples.
        """
        metadata = self.get_metadata(pdf_source)
        if metadata.is_encrypted:
            raise PDFEncryptedError("PDF is encrypted and cannot be processed.")
        if metadata.total_pages == 0:
            return

        start_page = max(1, first_page) if first_page else 1
        end_page = min(metadata.total_pages, last_page) if last_page else metadata.total_pages

        curr = start_page
        while curr <= end_page:
            chunk_end = min(curr + batch_size - 1, end_page)
            logger.debug(
                "Processing page batch chunk: pages {}..{} of {}",
                curr,
                chunk_end,
                metadata.total_pages,
            )
            chunk_images = self._render_range(pdf_source, curr, chunk_end)
            for page_num, img in chunk_images:
                yield page_num, img
            curr = chunk_end + 1

    # --------------------------------------------------------------------------
    # Persistence Helper
    # --------------------------------------------------------------------------

    def save_page_images(
        self,
        page_images: Sequence[tuple[int, Image.Image]],
        output_dir: str | Path,
        image_format: str = "PNG",
        prefix: str = "page",
    ) -> list[Path]:
        """Save a collection of page images to a target directory.

        Args:
            page_images: List of (page_number, PIL.Image) tuples.
            output_dir: Directory where images should be saved.
            image_format: Image file format (e.g. "PNG", "JPEG").
            prefix: Filename prefix for generated files.

        Returns:
            List of saved image file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        ext = image_format.lower()
        if ext == "jpeg":
            ext = "jpg"

        for page_num, img in page_images:
            file_path = out_path / f"{prefix}_{page_num:04d}.{ext}"
            img.save(file_path, format=image_format)
            saved_paths.append(file_path)

        logger.info("Saved {} page images to '{}'", len(saved_paths), out_path)
        return saved_paths

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------

    def _render_range(
        self,
        pdf_source: str | Path | bytes | BinaryIO,
        start_page: int,
        end_page: int,
    ) -> list[tuple[int, Image.Image]]:
        """Internal worker to render a slice of pages with pdf2image."""
        try:
            if isinstance(pdf_source, (str, Path)):
                path_obj = Path(pdf_source)
                if not path_obj.is_file():
                    raise PDFFileNotFoundError(f"PDF file not found at '{path_obj}'")
                images = convert_from_path(
                    pdf_path=str(path_obj),
                    dpi=self.dpi,
                    first_page=start_page,
                    last_page=end_page,
                    poppler_path=self.poppler_path,
                    thread_count=self.thread_count,
                )
            else:
                raw_bytes = (
                    pdf_source.getvalue()
                    if isinstance(pdf_source, io.BytesIO)
                    else (
                        pdf_source.read()
                        if hasattr(pdf_source, "read")
                        else pdf_source
                    )
                )
                images = convert_from_bytes(
                    pdf_file=raw_bytes,
                    dpi=self.dpi,
                    first_page=start_page,
                    last_page=end_page,
                    poppler_path=self.poppler_path,
                    thread_count=self.thread_count,
                )

            # Match 1-indexed page numbering
            result: list[tuple[int, Image.Image]] = []
            for idx, img in enumerate(images):
                page_num = start_page + idx
                result.append((page_num, img))
            return result

        except PDFInfoNotInstalledError as e:
            raise PopplerNotInstalledError(
                "Poppler is not installed or not in PATH. Please install poppler-utils "
                "(Linux: apt install poppler-utils, Windows: download poppler binaries "
                "and specify POPPLER_PATH in .env)."
            ) from e
        except (PDFPageCountError, PDFSyntaxError) as e:
            raise PDFCorruptedError(f"Failed to rasterize PDF pages: {e}") from e
        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"Unexpected error during PDF page rendering: {e}") from e

    def _get_stream_and_size(
        self, pdf_source: str | Path | bytes | BinaryIO
    ) -> tuple[BinaryIO, int]:
        """Normalize pdf_source to a seekable BinaryIO stream and byte length."""
        if isinstance(pdf_source, (str, Path)):
            path_obj = Path(pdf_source)
            if not path_obj.is_file():
                raise PDFFileNotFoundError(f"PDF file not found at '{path_obj}'")
            file_bytes = path_obj.read_bytes()
            return io.BytesIO(file_bytes), len(file_bytes)
        elif isinstance(pdf_source, (bytes, bytearray)):
            raw_bytes = bytes(pdf_source)
            return io.BytesIO(raw_bytes), len(raw_bytes)
        elif isinstance(pdf_source, io.BytesIO):
            pdf_source.seek(0)
            data = pdf_source.getvalue()
            return io.BytesIO(data), len(data)
        elif hasattr(pdf_source, "read"):
            data = pdf_source.read()
            return io.BytesIO(data), len(data)
        else:
            raise PDFProcessingError(f"Unsupported PDF source type: {type(pdf_source)}")
