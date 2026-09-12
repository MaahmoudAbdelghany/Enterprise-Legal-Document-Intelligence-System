"""Unit tests for app.core.ingestion.text_extractor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from app.config import Settings
from app.core.ingestion.pdf_processor import PDFPageImage
from app.core.ingestion.text_extractor import (
    EmptyImageError,
    LLMProviderError,
    NoLLMProviderConfiguredError,
    PageText,
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


# ------------------------------------------------------------------------------
# Fixtures & Helpers
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple 100x100 RGB image for testing."""
    return Image.new("RGB", (100, 100), color=(255, 255, 255))


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a simple 100x100 RGBA image with transparency for testing."""
    return Image.new("RGBA", (100, 100), color=(200, 200, 200, 128))


@pytest.fixture
def sample_page_image(sample_image: Image.Image) -> PDFPageImage:
    """Create a PDFPageImage instance for testing."""
    return PDFPageImage(
        page_number=3,
        image=sample_image,
        width=100,
        height=100,
        dpi=200,
    )


@pytest.fixture
def mock_primary_llm() -> MagicMock:
    """Mock LangChain chat model for primary vision extractor."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(
        content="# عقد توريد تجاري\n\nالمادة ١: يلتزم الطرف الأول بتوريد البضائع المتفق عليها."
    )
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="# عقد توريد تجاري\n\nالمادة ١: يلتزم الطرف الأول بتوريد البضائع المتفق عليها."
        )
    )
    return mock


@pytest.fixture
def mock_fallback_llm() -> MagicMock:
    """Mock LangChain chat model for fallback vision extractor."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(
        content="OpenAI Fallback: عقد إيجار تجاري موثق"
    )
    mock.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="OpenAI Fallback: عقد إيجار تجاري موثق"
        )
    )
    return mock


# ------------------------------------------------------------------------------
# Arabic Normalization & NLP Helper Tests
# ------------------------------------------------------------------------------

def test_remove_tashkeel():
    """Verify that Arabic diacritics (harakat) are completely stripped."""
    text_with_harakat = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    result = remove_tashkeel(text_with_harakat)
    assert result == "بسم الله الرحمن الرحيم"
    assert "ِ" not in result
    assert "ّ" not in result
    assert "ْ" not in result


def test_remove_tatweel():
    """Verify that elongation kashida is removed."""
    elongated = "شــــركــــة المــــحــــامــــاة"
    assert remove_tatweel(elongated) == "شركة المحاماة"


def test_normalize_alef():
    """Verify that all Alef variants (أ, إ, آ, ٱ) are unified to bare Alef."""
    text = "إبرام اتفاقية أتعاب آمنة ٱستثنائية"
    result = normalize_alef(text)
    assert result == "ابرام اتفاقية اتعاب امنة استثنائية"


def test_normalize_taa_marbuta():
    """Verify that Taa Marbuta is standardized to Haa."""
    text = "شركة تجارية مساهمة"
    result = normalize_taa_marbuta(text)
    assert result == "شركه تجاريه مساهمه"


def test_normalize_ya():
    """Verify that Alef Maqsura (ى) is standardized to standard Yaa (ي)."""
    text = "يحتوي المستند على دعوى قضائية لدى المحكمة"
    result = normalize_ya(text)
    assert result == "يحتوي المستند علي دعوي قضائية لدي المحكمة"


def test_normalize_arabic_digits():
    """Verify that Eastern Arabic numerals are mapped to Western digits."""
    text = "المادة ١٤ من القانون رقم ٢٠٢٣ لسنة ١٩٩٨"
    result = normalize_arabic_digits(text)
    assert result == "المادة 14 من القانون رقم 2023 لسنة 1998"


def test_normalize_arabic_full_pipeline():
    """Verify end-to-end Arabic normalization pipeline."""
    raw_legal_text = (
        "  الْمَادَّةُ ١:   يَلْتَزِمُ الطَّرَفُ الْأَوَّلُ بِتَسْلِيمِ   "
        "الْبَضَائِعِ إِلَى شَـــرِكَةِ الْمُشْتَرِي.  \n\n\n  "
    )
    cleaned = normalize_arabic(raw_legal_text)
    assert cleaned == "الماده 1: يلتزم الطرف الاول بتسليم البضائع الي شركه المشتري."


def test_normalize_arabic_empty():
    """Verify that empty or None text gracefully returns empty string."""
    assert normalize_arabic("") == ""
    assert normalize_arabic(None) == ""  # type: ignore[arg-type]


# ------------------------------------------------------------------------------
# Language & Table Detection Tests
# ------------------------------------------------------------------------------

def test_detect_language_arabic():
    """Verify Arabic detection on legal text."""
    arabic_text = "حكمت المحكمة بإلزام المدعى عليه بأداء المبلغ المطالب به"
    assert detect_language(arabic_text) == "ar"


def test_detect_language_english():
    """Verify English detection on legal text."""
    english_text = "The commercial contract shall be governed by international law."
    assert detect_language(english_text) == "en"


def test_detect_language_mixed():
    """Verify detection of mixed bilingual legal text."""
    mixed_text = "عقد بيع أسهم Share Purchase Agreement بين الطرفين"
    assert detect_language(mixed_text) == "mixed"


def test_detect_language_empty_and_numbers():
    """Verify unknown status on empty or number-only text."""
    assert detect_language("") == "unknown"
    assert detect_language("   ") == "unknown"
    assert detect_language("123456789") == "unknown"


def test_detect_tables():
    """Verify detection of markdown tables."""
    text_with_table = (
        "| البند | الوصف | القيمة |\n|---|---|---|\n| 1 | أتعاب | 5000 |"
    )
    text_without_table = "هذا نص عادي لا يحتوي على أي جداول بيانات."
    assert detect_tables(text_with_table) is True
    assert detect_tables(text_without_table) is False
    assert detect_tables("") is False


# ------------------------------------------------------------------------------
# Image Encoding Tests
# ------------------------------------------------------------------------------

def test_encode_image_to_base64_rgb(sample_image: Image.Image):
    """Verify encoding of standard RGB image."""
    b64 = encode_image_to_base64(sample_image)
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_encode_image_to_base64_rgba(sample_rgba_image: Image.Image):
    """Verify encoding handles RGBA with transparency conversion to RGB."""
    b64 = encode_image_to_base64(sample_rgba_image, image_format="JPEG")
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_encode_image_to_base64_zero_dimensions():
    """Verify that zero-sized images raise EmptyImageError."""
    bad_img = Image.new("RGB", (0, 100))
    with pytest.raises(EmptyImageError, match="zero-dimension"):
        encode_image_to_base64(bad_img)

    with pytest.raises(EmptyImageError, match="zero-dimension"):
        encode_image_to_base64(None)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------
# PageText Data Model Tests
# ------------------------------------------------------------------------------

def test_page_text_to_dict_and_from_dict():
    """Verify serialization and deserialization of PageText."""
    pt = PageText(
        page_number=2,
        raw_text="عقد أتعاب",
        normalized_text="عقد اتعاب",
        language="ar",
        has_tables=False,
        metadata={"provider": "groq", "confidence": 0.98},
    )
    d = pt.to_dict()
    assert d["page_number"] == 2
    assert d["raw_text"] == "عقد أتعاب"
    assert d["normalized_text"] == "عقد اتعاب"
    assert d["metadata"]["provider"] == "groq"

    restored = PageText.from_dict(d)
    assert restored.page_number == pt.page_number
    assert restored.raw_text == pt.raw_text
    assert restored.normalized_text == pt.normalized_text
    assert restored.metadata == pt.metadata


# ------------------------------------------------------------------------------
# TextExtractor Initialization Tests
# ------------------------------------------------------------------------------

def test_text_extractor_no_providers():
    """Verify behavior when no LLM providers are configured."""
    settings = Settings(
        groq_api_key="",
        openai_api_key="",
    )
    extractor = TextExtractor(settings=settings)
    assert not extractor.has_active_provider

    dummy_img = Image.new("RGB", (50, 50))
    with pytest.raises(NoLLMProviderConfiguredError):
        extractor.extract_page_text(dummy_img)


def test_text_extractor_injected_primary(mock_primary_llm: MagicMock):
    """Verify extractor with injected primary LLM."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    assert extractor.has_active_provider


def test_text_extractor_init_with_groq_and_openai():
    """Verify extractor initializes ChatGroq and ChatOpenAI when keys provided."""
    with patch("langchain_groq.ChatGroq") as mock_groq_cls, patch("langchain_openai.ChatOpenAI") as mock_openai_cls:
        extractor = TextExtractor(
            groq_api_key="gsk_test_groq_key_123",
            openai_api_key="sk-test-openai-key-456",
        )
        assert mock_groq_cls.called
        assert mock_openai_cls.called
        assert extractor.has_active_provider


def test_text_extractor_openai_only_promoted():
    """Verify that when only OpenAI is configured, it is promoted to primary."""
    with patch("langchain_openai.ChatOpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        extractor = TextExtractor(
            groq_api_key="",
            openai_api_key="sk-test-openai-key-456",
        )
        assert extractor.has_active_provider
        assert extractor._primary_llm == mock_instance
        assert extractor._fallback_llm is None


# ------------------------------------------------------------------------------
# Text Extraction Execution Tests
# ------------------------------------------------------------------------------

def test_extract_page_text_success_image(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify successful page extraction with PIL.Image input."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    result = extractor.extract_page_text(sample_image, page_number=1)

    assert result.page_number == 1
    assert "عقد توريد تجاري" in result.raw_text
    assert "الماده 1:" in result.normalized_text
    assert result.language == "ar"
    assert result.metadata["provider"] == "primary"
    assert mock_primary_llm.invoke.called


def test_extract_page_text_success_pdf_page_image(
    mock_primary_llm: MagicMock,
    sample_page_image: PDFPageImage,
):
    """Verify successful page extraction with PDFPageImage input preserving page_number."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    result = extractor.extract_page_text(sample_page_image)

    assert result.page_number == 3  # Inherited from PDFPageImage
    assert result.metadata["image_width"] == 100
    assert result.metadata["image_height"] == 100


def test_extract_page_text_invalid_input_type(mock_primary_llm: MagicMock):
    """Verify error on unsupported page object."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    with pytest.raises(EmptyImageError, match="Unsupported page input type"):
        extractor.extract_page_text("not an image")  # type: ignore[arg-type]


def test_extract_page_text_zero_dimension(mock_primary_llm: MagicMock):
    """Verify error when page image has 0 dimension."""
    bad_img = Image.new("RGB", (0, 50))
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    with pytest.raises(EmptyImageError, match="invalid dimensions"):
        extractor.extract_page_text(bad_img)


def test_extract_page_text_fallback_on_primary_failure(
    mock_primary_llm: MagicMock,
    mock_fallback_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify fallback LLM is invoked when primary fails."""
    mock_primary_llm.invoke.side_effect = RuntimeError("Groq Rate Limit exceeded")

    extractor = TextExtractor(
        primary_llm=mock_primary_llm,
        fallback_llm=mock_fallback_llm,
    )

    result = extractor.extract_page_text(sample_image, page_number=2)
    assert result.metadata["provider"] == "fallback"
    assert "OpenAI Fallback" in result.raw_text
    assert mock_fallback_llm.invoke.called


def test_extract_page_text_both_fail_raises_error(
    mock_primary_llm: MagicMock,
    mock_fallback_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify LLMProviderError is raised when both primary and fallback fail."""
    mock_primary_llm.invoke.side_effect = RuntimeError("Groq 500 error")
    mock_fallback_llm.invoke.side_effect = RuntimeError("OpenAI 500 error")

    extractor = TextExtractor(
        primary_llm=mock_primary_llm,
        fallback_llm=mock_fallback_llm,
    )

    with pytest.raises(LLMProviderError, match="Both primary and fallback"):
        extractor.extract_page_text(sample_image, page_number=1)


def test_extract_page_text_primary_fails_no_fallback(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify LLMProviderError is raised when primary fails and no fallback is set."""
    mock_primary_llm.invoke.side_effect = RuntimeError("Network timeout")

    extractor = TextExtractor(primary_llm=mock_primary_llm, fallback_llm=None)

    with pytest.raises(LLMProviderError, match="no fallback configured"):
        extractor.extract_page_text(sample_image, page_number=1)


# ------------------------------------------------------------------------------
# Async and Batch Extraction Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aextract_page_text_success(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify asynchronous page extraction."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    result = await extractor.aextract_page_text(sample_image, page_number=1)

    assert result.page_number == 1
    assert "عقد توريد تجاري" in result.raw_text
    assert mock_primary_llm.ainvoke.called


@pytest.mark.asyncio
async def test_aextract_page_text_fallback(
    mock_primary_llm: MagicMock,
    mock_fallback_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify asynchronous fallback on primary failure."""
    mock_primary_llm.ainvoke.side_effect = RuntimeError("Groq Async Error")

    extractor = TextExtractor(
        primary_llm=mock_primary_llm,
        fallback_llm=mock_fallback_llm,
    )
    result = await extractor.aextract_page_text(sample_image, page_number=1)
    assert result.metadata["provider"] == "fallback"
    assert "OpenAI Fallback" in result.raw_text


@pytest.mark.asyncio
async def test_aextract_page_text_no_provider(sample_image: Image.Image):
    """Verify async extraction raises when no provider configured."""
    settings = Settings(groq_api_key="", openai_api_key="")
    extractor = TextExtractor(settings=settings)
    with pytest.raises(NoLLMProviderConfiguredError):
        await extractor.aextract_page_text(sample_image)


@pytest.mark.asyncio
async def test_aextract_page_text_both_fail(
    mock_primary_llm: MagicMock,
    mock_fallback_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify async extraction raises LLMProviderError when both fail."""
    mock_primary_llm.ainvoke.side_effect = RuntimeError("Groq Fail")
    mock_fallback_llm.ainvoke.side_effect = RuntimeError("OpenAI Fail")

    extractor = TextExtractor(
        primary_llm=mock_primary_llm,
        fallback_llm=mock_fallback_llm,
    )
    with pytest.raises(LLMProviderError, match="Both primary and fallback"):
        await extractor.aextract_page_text(sample_image)


@pytest.mark.asyncio
async def test_aextract_page_text_primary_fails_no_fallback(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify async extraction raises when primary fails and no fallback exists."""
    mock_primary_llm.ainvoke.side_effect = RuntimeError("Groq timeout")
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    with pytest.raises(LLMProviderError, match="no fallback configured"):
        await extractor.aextract_page_text(sample_image)


def test_extract_document_text_batch(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify synchronous batch document extraction."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    pages = [sample_image, sample_image, sample_image]
    results = extractor.extract_document_text(pages)

    assert len(results) == 3
    assert [r.page_number for r in results] == [1, 2, 3]


@pytest.mark.asyncio
async def test_aextract_document_text_batch(
    mock_primary_llm: MagicMock,
    sample_image: Image.Image,
):
    """Verify asynchronous bounded batch document extraction."""
    extractor = TextExtractor(primary_llm=mock_primary_llm)
    pages = [sample_image, sample_image, sample_image, sample_image]
    results = await extractor.aextract_document_text(pages, max_concurrency=2)

    assert len(results) == 4
    assert [r.page_number for r in results] == [1, 2, 3, 4]
