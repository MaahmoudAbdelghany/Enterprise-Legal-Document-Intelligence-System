"""LLM-based text extraction and Arabic normalization module for LexisGraph.

Extracts text from document page images using vision-capable LLMs (Groq Llama 3.2 Vision
with OpenAI GPT-4o fallback), normalizes Arabic script for downstream search and retrieval,
and packages page-level text into structured payloads for Qdrant storage.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models.chat_models import BaseChatModel
    from app.core.ingestion.pdf_processor import PDFPageImage

from app.config import Settings, get_settings


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class TextExtractionError(Exception):
    """Base exception for all text extraction errors."""


class LLMProviderError(TextExtractionError):
    """Raised when an LLM provider fails to process an image or returns an error."""


class EmptyImageError(TextExtractionError):
    """Raised when the provided image is invalid, empty, or zero-sized."""


class NoLLMProviderConfiguredError(TextExtractionError):
    """Raised when neither Groq nor OpenAI API credentials are configured."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PageText:
    """Structured text extracted from a single document page.

    Attributes:
        page_number: 1-indexed page number within the source document.
        raw_text: Verbatim extracted text from the vision LLM.
        normalized_text: Standardized Arabic/multilingual text (tashkeel/tatweel removed, etc.).
        language: Detected primary language ('ar', 'en', 'mixed', or 'unknown').
        has_tables: Whether markdown table structures were detected in the page.
        metadata: Arbitrary additional metadata (token counts, model name, confidence, etc.).
    """

    page_number: int
    raw_text: str
    normalized_text: str
    language: str = "ar"
    has_tables: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert page text model to dictionary for Qdrant payload or JSON responses."""
        return {
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "has_tables": self.has_tables,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageText:
        """Instantiate PageText from dictionary representation."""
        return cls(
            page_number=data.get("page_number", 1),
            raw_text=data.get("raw_text", ""),
            normalized_text=data.get("normalized_text", ""),
            language=data.get("language", "ar"),
            has_tables=data.get("has_tables", False),
            metadata=data.get("metadata", {}),
        )


# ------------------------------------------------------------------------------
# Arabic Text Normalization
# ------------------------------------------------------------------------------

# Unicode regex patterns for Arabic script normalization
_TASHKEEL_REGEX = re.compile(r"[\u064B-\u0652\u0670]")  # Fatha, Damma, Kasra, Shadda, Sukun, etc.
_TATWEEL_REGEX = re.compile(r"\u0640")  # Kashida / Tatweel
_ALEF_REGEX = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آ, أ, إ, ٱ -> ا
_TAA_MARBUTA_REGEX = re.compile(r"\u0629")  # ة -> ه
_ALEF_MAQSURA_REGEX = re.compile(r"\u0649")  # ى -> ي
_ARABIC_LETTERS_REGEX = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTERS_REGEX = re.compile(r"[a-zA-Z]")

# Eastern Arabic numerals to standard Western numerals mapping
_EASTERN_TO_WESTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def remove_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (harakat / tashkeel) from text."""
    return _TASHKEEL_REGEX.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove Arabic kashida / tatweel elongations from text."""
    return _TATWEEL_REGEX.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize various Alef forms (أ, إ, آ, ٱ) into a bare Alef (ا)."""
    return _ALEF_REGEX.sub("\u0627", text)


def normalize_taa_marbuta(text: str) -> str:
    """Normalize Taa Marbuta (ة) to Haa (ه)."""
    return _TAA_MARBUTA_REGEX.sub("\u0647", text)


def normalize_ya(text: str) -> str:
    """Normalize Alef Maqsura (ى) to standard Yaa (ي)."""
    return _ALEF_MAQSURA_REGEX.sub("\u064A", text)


def normalize_arabic_digits(text: str) -> str:
    """Convert Eastern Arabic digits (٠-٩) to standard Western digits (0-9)."""
    return text.translate(_EASTERN_TO_WESTERN_DIGITS)


def normalize_arabic(
    text: str,
    remove_diacritics: bool = True,
    normalize_alef_chars: bool = True,
    normalize_taa: bool = True,
    normalize_alef_maqsura: bool = True,
    strip_tatweel: bool = True,
    convert_digits: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """Normalize Arabic text for robust search, BM25 indexing, and retrieval.

    Args:
        text: Raw text string.
        remove_diacritics: Strip harakat (Fatha, Damma, Kasra, Shadda, Sukun).
        normalize_alef_chars: Unify all Alef variants to bare Alef (ا).
        normalize_taa: Convert Taa Marbuta (ة) to Haa (ه).
        normalize_alef_maqsura: Convert Alef Maqsura (ى) to Yaa (ي).
        strip_tatweel: Strip elongation tatweel (ـ).
        convert_digits: Convert Eastern numerals (٠-٩) to 0-9.
        normalize_whitespace: Collapse consecutive whitespace and strip margins.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    result = text

    if remove_diacritics:
        result = remove_tashkeel(result)

    if strip_tatweel:
        result = remove_tatweel(result)

    if normalize_alef_chars:
        result = normalize_alef(result)

    if normalize_taa:
        result = normalize_taa_marbuta(result)

    if normalize_alef_maqsura:
        result = normalize_ya(result)

    if convert_digits:
        result = normalize_arabic_digits(result)

    if normalize_whitespace:
        # Collapse multiple spaces and tabs into single space, normalize newlines
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

    return result


def detect_language(text: str) -> str:
    """Heuristically detect whether text is Arabic, English, mixed, or unknown.

    Args:
        text: Input string.

    Returns:
        One of 'ar', 'en', 'mixed', or 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"

    arabic_count = len(_ARABIC_LETTERS_REGEX.findall(text))
    latin_count = len(_LATIN_LETTERS_REGEX.findall(text))
    total_letters = arabic_count + latin_count

    if total_letters == 0:
        return "unknown"

    arabic_ratio = arabic_count / total_letters
    latin_ratio = latin_count / total_letters

    if arabic_ratio >= 0.7:
        return "ar"
    if latin_ratio >= 0.7:
        return "en"
    if arabic_ratio >= 0.2 and latin_ratio >= 0.2:
        return "mixed"

    return "ar" if arabic_ratio > latin_ratio else "en"


def detect_tables(text: str) -> bool:
    """Detect whether markdown table formatting exists in extracted text."""
    if not text:
        return False
    # Check for markdown table divider pattern: |---| or |:---|
    return bool(re.search(r"\|(?:\s*:?-+:?\s*\|)+", text))


# ------------------------------------------------------------------------------
# Image Encoding Helper
# ------------------------------------------------------------------------------

def encode_image_to_base64(
    image: Image.Image,
    image_format: str = "JPEG",
    quality: int = 90,
) -> str:
    """Encode a PIL Image into a base64-encoded string.

    Args:
        image: PIL Image instance.
        image_format: Image format ('JPEG' or 'PNG').
        quality: JPEG compression quality (1-100).

    Returns:
        Base64-encoded string of the image.

    Raises:
        EmptyImageError: If image width or height is zero.
    """
    if image is None or image.width == 0 or image.height == 0:
        raise EmptyImageError("Cannot encode empty or zero-dimension image.")

    # Convert RGBA / P mode images to RGB for JPEG compatibility
    working_image = image
    if image_format.upper() in ("JPEG", "JPG"):
        if working_image.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", working_image.size, (255, 255, 255))
            if working_image.mode == "P":
                working_image = working_image.convert("RGBA")
            rgb_img.paste(
                working_image,
                mask=working_image.split()[-1] if working_image.mode in ("RGBA", "LA") else None,
            )
            working_image = rgb_img
        elif working_image.mode != "RGB":
            working_image = working_image.convert("RGB")

    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": image_format}
    if image_format.upper() in ("JPEG", "JPG"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    working_image.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ------------------------------------------------------------------------------
# System Prompts for Legal Vision Extraction
# ------------------------------------------------------------------------------

LEGAL_VISION_SYSTEM_PROMPT = """You are an expert legal document transcription engine specializing in Arabic and bilingual (Arabic/English) legal instruments, contracts, judicial rulings, statutes, and regulatory decrees.
Your sole mission is to extract and transcribe all textual and structured content from the provided document page image with 100% verbatim accuracy.

Strict Extraction Rules:
1. Verbatim Fidelity: Transcribe every word, article number (e.g., المادة الأولى، البند ثانياً), clause, party name, monetary sum, date, and reference code exactly as written.
2. Structure Preservation: Preserve legal document hierarchy using Markdown:
   - Document titles and main sections as `#` or `##` headers.
   - Articles and clauses as numbered or bolded paragraphs.
   - Bullet points, indentations, and list enumerations.
3. Tables and Matrices: Transcribe any tabular data, financial figures, or fee schedules into valid Markdown tables (`| Header 1 | Header 2 |`).
4. Stamps, Seals & Signatures: If stamps, seals, or signature blocks appear, indicate them cleanly in brackets (e.g., `[ختم رسمي: المحكمة التجارية]` or `[توقيع الطرف الأول]`).
5. Zero Hallucination & No Chat: Do NOT summarize, correct, interpret, or omit any text. Do NOT include preambles, explanations, or closing remarks (such as 'Here is the transcription:'). Output ONLY the raw extracted document content.
"""

LEGAL_VISION_USER_PROMPT = "Transcribe all text and structure from this legal document page image exactly as it appears."


# ------------------------------------------------------------------------------
# Text Extractor Engine
# ------------------------------------------------------------------------------

class TextExtractor:
    """Vision-LLM text extraction engine for Arabic and multilingual legal documents.

    Uses Groq vision models (e.g. llama-3.2-11b-vision-preview) as primary high-speed provider,
    with automatic fallback to OpenAI (e.g. gpt-4o) on failure or when Groq is unconfigured.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        primary_llm: BaseChatModel | None = None,
        fallback_llm: BaseChatModel | None = None,
        groq_api_key: str | None = None,
        groq_vision_model: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the TextExtractor.

        Args:
            settings: LexisGraph settings instance. If None, defaults to `get_settings()`.
            primary_llm: Pre-instantiated LangChain chat model to use as primary (useful for DI/testing).
            fallback_llm: Pre-instantiated LangChain chat model to use as fallback.
            groq_api_key: Explicit Groq API key override.
            groq_vision_model: Explicit Groq vision model override.
            openai_api_key: Explicit OpenAI API key override.
            openai_model: Explicit OpenAI model override.
            temperature: LLM temperature (defaults to 0.1 for high extraction fidelity).
            max_retries: Maximum retries on transient API failure.
        """
        self._settings = settings or get_settings()
        self._temperature = temperature if temperature is not None else self._settings.groq_temperature
        self._max_retries = max_retries if max_retries is not None else self._settings.groq_max_retries

        self._groq_api_key = (
            groq_api_key if groq_api_key is not None else self._settings.groq_api_key
        ).strip()
        self._groq_vision_model = (
            groq_vision_model if groq_vision_model is not None else self._settings.groq_vision_model
        ).strip()

        self._openai_api_key = (
            openai_api_key if openai_api_key is not None else self._settings.openai_api_key
        ).strip()
        self._openai_model = (
            openai_model if openai_model is not None else self._settings.openai_model
        ).strip()

        # Injected or initialized models
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm

        self._init_models_if_needed()

    def _init_models_if_needed(self) -> None:
        """Initialize primary and fallback LLM instances if not explicitly provided."""
        # Initialize Groq if key is present and primary_llm not injected
        if self._primary_llm is None and self._groq_api_key:
            try:
                from langchain_groq import ChatGroq

                self._primary_llm = ChatGroq(
                    api_key=self._groq_api_key,
                    model_name=self._groq_vision_model,
                    temperature=self._temperature,
                    max_retries=self._max_retries,
                )
                logger.info(
                    f"Initialized primary vision LLM with Groq model: {self._groq_vision_model}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGroq: {e}")

        # Initialize OpenAI fallback if key is present and fallback_llm not injected
        if self._fallback_llm is None and self._openai_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self._fallback_llm = ChatOpenAI(
                    api_key=self._openai_api_key,
                    model_name=self._openai_model,
                    temperature=self._temperature,
                    max_retries=self._max_retries,
                )
                logger.info(
                    f"Initialized fallback vision LLM with OpenAI model: {self._openai_model}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatOpenAI: {e}")

        # If primary is still None but fallback is available, promote fallback to primary
        if self._primary_llm is None and self._fallback_llm is not None:
            self._primary_llm = self._fallback_llm
            self._fallback_llm = None
            logger.info("Promoted OpenAI model to primary vision LLM (no Groq key configured).")

    @property
    def has_active_provider(self) -> bool:
        """Check if at least one vision LLM provider is configured and available."""
        return self._primary_llm is not None or self._fallback_llm is not None

    def build_vision_messages(self, b64_image: str) -> list[BaseMessage]:
        """Construct multimodal LangChain messages for vision LLM inference."""
        return [
            SystemMessage(content=LEGAL_VISION_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": LEGAL_VISION_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ]
            ),
        ]

    def _resolve_image_and_page_number(
        self,
        page: PDFPageImage | Image.Image,
        page_number: int | None = None,
    ) -> tuple[Image.Image, int]:
        """Extract PIL.Image and 1-indexed page number from input."""
        # Check if page is PDFPageImage
        if hasattr(page, "image") and hasattr(page, "page_number"):
            img = page.image
            num = page.page_number if page_number is None else page_number
        elif isinstance(page, Image.Image):
            img = page
            num = page_number if page_number is not None else 1
        else:
            raise EmptyImageError(f"Unsupported page input type: {type(page).__name__}")

        if img.width == 0 or img.height == 0:
            raise EmptyImageError("Page image has invalid dimensions (width=0 or height=0).")

        return img, num

    def extract_page_text(
        self,
        page: PDFPageImage | Image.Image,
        page_number: int | None = None,
    ) -> PageText:
        """Extract and normalize text from a single document page image (synchronous).

        Args:
            page: Either a `PDFPageImage` instance or a PIL `Image.Image`.
            page_number: Optional 1-indexed page number override.

        Returns:
            Structured `PageText` instance with raw and normalized text.

        Raises:
            EmptyImageError: If the image is empty or invalid.
            NoLLMProviderConfiguredError: If no LLM provider is available.
            LLMProviderError: If both primary and fallback LLMs fail.
        """
        if not self.has_active_provider:
            raise NoLLMProviderConfiguredError(
                "No vision LLM provider configured. Please provide GROQ_API_KEY or OPENAI_API_KEY."
            )

        img, num = self._resolve_image_and_page_number(page, page_number)
        b64_img = encode_image_to_base64(img)
        messages = self.build_vision_messages(b64_img)

        raw_text = ""
        provider_used = "primary"

        # Try primary LLM
        try:
            assert self._primary_llm is not None
            response = self._primary_llm.invoke(messages)
            raw_text = str(response.content)
        except Exception as primary_err:
            logger.warning(
                f"Primary vision LLM failed on page {num}: {primary_err}. Attempting fallback..."
            )
            # Try fallback LLM if available
            if self._fallback_llm is not None:
                try:
                    response = self._fallback_llm.invoke(messages)
                    raw_text = str(response.content)
                    provider_used = "fallback"
                except Exception as fallback_err:
                    logger.error(
                        f"Fallback vision LLM also failed on page {num}: {fallback_err}"
                    )
                    raise LLMProviderError(
                        f"Both primary and fallback vision LLMs failed on page {num}. "
                        f"Primary: {primary_err} | Fallback: {fallback_err}"
                    ) from fallback_err
            else:
                raise LLMProviderError(
                    f"Primary vision LLM failed on page {num} and no fallback configured: {primary_err}"
                ) from primary_err

        # Normalize and construct PageText
        normalized_text = normalize_arabic(raw_text)
        language = detect_language(raw_text)
        has_tables = detect_tables(raw_text)

        return PageText(
            page_number=num,
            raw_text=raw_text,
            normalized_text=normalized_text,
            language=language,
            has_tables=has_tables,
            metadata={
                "provider": provider_used,
                "image_width": img.width,
                "image_height": img.height,
            },
        )

    async def aextract_page_text(
        self,
        page: PDFPageImage | Image.Image,
        page_number: int | None = None,
    ) -> PageText:
        """Extract and normalize text from a single document page image (asynchronous).

        Args:
            page: Either a `PDFPageImage` instance or a PIL `Image.Image`.
            page_number: Optional 1-indexed page number override.

        Returns:
            Structured `PageText` instance with raw and normalized text.
        """
        if not self.has_active_provider:
            raise NoLLMProviderConfiguredError(
                "No vision LLM provider configured. Please provide GROQ_API_KEY or OPENAI_API_KEY."
            )

        img, num = self._resolve_image_and_page_number(page, page_number)
        b64_img = encode_image_to_base64(img)
        messages = self.build_vision_messages(b64_img)

        raw_text = ""
        provider_used = "primary"

        try:
            assert self._primary_llm is not None
            response = await self._primary_llm.ainvoke(messages)
            raw_text = str(response.content)
        except Exception as primary_err:
            logger.warning(
                f"Primary vision LLM failed asynchronously on page {num}: {primary_err}. Attempting fallback..."
            )
            if self._fallback_llm is not None:
                try:
                    response = await self._fallback_llm.ainvoke(messages)
                    raw_text = str(response.content)
                    provider_used = "fallback"
                except Exception as fallback_err:
                    logger.error(
                        f"Fallback vision LLM also failed on page {num}: {fallback_err}"
                    )
                    raise LLMProviderError(
                        f"Both primary and fallback vision LLMs failed on page {num}. "
                        f"Primary: {primary_err} | Fallback: {fallback_err}"
                    ) from fallback_err
            else:
                raise LLMProviderError(
                    f"Primary vision LLM failed on page {num} and no fallback configured: {primary_err}"
                ) from primary_err

        normalized_text = normalize_arabic(raw_text)
        language = detect_language(raw_text)
        has_tables = detect_tables(raw_text)

        return PageText(
            page_number=num,
            raw_text=raw_text,
            normalized_text=normalized_text,
            language=language,
            has_tables=has_tables,
            metadata={
                "provider": provider_used,
                "image_width": img.width,
                "image_height": img.height,
            },
        )

    def extract_document_text(
        self,
        pages: Sequence[PDFPageImage | Image.Image],
    ) -> list[PageText]:
        """Synchronously extract text from all pages in a document.

        Args:
            pages: Sequence of `PDFPageImage` or PIL `Image.Image` objects.

        Returns:
            List of `PageText` objects ordered by page number.
        """
        results: list[PageText] = []
        for idx, page in enumerate(pages, start=1):
            results.append(self.extract_page_text(page, page_number=idx))
        return results

    async def aextract_document_text(
        self,
        pages: Sequence[PDFPageImage | Image.Image],
        max_concurrency: int = 3,
    ) -> list[PageText]:
        """Asynchronously extract text from all pages in a document with bounded concurrency.

        Args:
            pages: Sequence of `PDFPageImage` or PIL `Image.Image` objects.
            max_concurrency: Maximum number of concurrent vision LLM requests.

        Returns:
            List of `PageText` objects ordered by page number.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_extract(
            page: PDFPageImage | Image.Image,
            page_num: int,
        ) -> PageText:
            async with semaphore:
                return await self.aextract_page_text(page, page_number=page_num)

        tasks = [
            _bounded_extract(page, page_num=idx)
            for idx, page in enumerate(pages, start=1)
        ]
        return await asyncio.gather(*tasks)
