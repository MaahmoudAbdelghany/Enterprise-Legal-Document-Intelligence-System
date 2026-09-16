"""Unit tests for app.core.generation.chain."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable

from app.config import Settings
from app.core.generation.chain import (
    ContextFormattingError,
    LegalGenerationError,
    LegalRAGChain,
    LegalRAGInput,
    LegalRAGOutput,
    NoLLMProviderConfiguredError,
    SourceDocument,
    create_default_legal_prompt,
    create_legal_rag_chain,
    format_legal_context,
    get_generation_llm,
    normalize_source_document,
)
from app.core.retrieval.qdrant_store import QdrantSearchResult


# ------------------------------------------------------------------------------
# Mock Chat Models for Tests
# ------------------------------------------------------------------------------

class MockLegalChatModel(BaseChatModel):
    """Custom mock ChatModel implementing LangChain BaseChatModel for testing."""

    response_text: str = "يلتزم الطرف الأول بتسليم العقار في الموعد المحدد."
    model_name: str = "mock-legal-llm"

    @property
    def _llm_type(self) -> str:
        return "mock-legal-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tokens = ["يلتزم ", "الطرف ", "الأول ", "بتسليم ", "العقار."]
        for token in tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        tokens = ["يلتزم ", "الطرف ", "الأول ", "بتسليم ", "العقار."]
        for token in tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))


class FailingLegalChatModel(BaseChatModel):
    """Mock ChatModel that simulates network/API failures."""

    model_name: str = "failing-llm"

    @property
    def _llm_type(self) -> str:
        return "failing-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("Groq API rate limit exceeded")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("Groq API rate limit exceeded")

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise RuntimeError("Groq streaming connection lost")
        yield  # make it a generator  # type: ignore

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise RuntimeError("Groq async streaming connection lost")
        yield  # make it an async generator  # type: ignore


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> MockLegalChatModel:
    """Provide a mock LangChain chat model."""
    return MockLegalChatModel()


@pytest.fixture
def sample_sources() -> list[SourceDocument]:
    """Provide a sample list of SourceDocument instances."""
    return [
        SourceDocument(
            document_id="contract-001",
            page_number=1,
            score=0.92,
            filename="commercial_agreement.pdf",
            text_snippet="المادة الأولى: اتفق الطرفان على توريد الأجهزة الكهربائية.",
            metadata={"type": "commercial_contract"},
            has_tables=False,
        ),
        SourceDocument(
            document_id="contract-001",
            page_number=2,
            score=0.88,
            filename="commercial_agreement.pdf",
            text_snippet="المادة الثانية: تكون مدة هذا العقد سنة ميلادية واحدة تبدأ من تاريخ توقيعه.",
            metadata={"type": "commercial_contract"},
            has_tables=True,
        ),
    ]


@pytest.fixture
def sample_qdrant_result() -> QdrantSearchResult:
    """Provide a sample QdrantSearchResult."""
    return QdrantSearchResult(
        point_id="pt-123",
        document_id="ruling-456",
        page_number=3,
        score=0.95,
        raw_text="حكمت المحكمة بإلزام المدعى عليه بدفع المبلغ المطالب به.",
        filename="court_judgment.pdf",
        has_tables=False,
        metadata={"court": "Commercial Court"},
    )


# ------------------------------------------------------------------------------
# SourceDocument & Normalization Tests
# ------------------------------------------------------------------------------

class TestSourceDocument:
    """Tests for SourceDocument data model and normalization helpers."""

    def test_source_document_init_and_serialization(self) -> None:
        doc = SourceDocument(
            document_id="doc_1",
            page_number=4,
            score=0.85,
            filename="file.pdf",
            text_snippet="نص المادة",
            metadata={"law": "civil"},
            has_tables=True,
        )
        data = doc.to_dict()
        assert data["document_id"] == "doc_1"
        assert data["page_number"] == 4
        assert data["score"] == 0.85
        assert data["filename"] == "file.pdf"
        assert data["has_tables"] is True
        assert data["metadata"] == {"law": "civil"}

        restored = SourceDocument.from_dict(data)
        assert restored.document_id == doc.document_id
        assert restored.page_number == doc.page_number
        assert restored.score == doc.score
        assert restored.text_snippet == doc.text_snippet
        assert restored.has_tables == doc.has_tables

    def test_source_document_from_qdrant_result(
        self, sample_qdrant_result: QdrantSearchResult
    ) -> None:
        doc = SourceDocument.from_qdrant_result(sample_qdrant_result)
        assert doc.document_id == "ruling-456"
        assert doc.page_number == 3
        assert doc.score == 0.95
        assert doc.filename == "court_judgment.pdf"
        assert "حكمت المحكمة" in doc.text_snippet
        assert doc.metadata == {"court": "Commercial Court"}

    def test_normalize_source_document_variants(
        self, sample_qdrant_result: QdrantSearchResult
    ) -> None:
        # SourceDocument instance
        doc = SourceDocument(document_id="d1", page_number=1, text_snippet="test")
        assert normalize_source_document(doc) is doc

        # QdrantSearchResult
        res_doc = normalize_source_document(sample_qdrant_result)
        assert res_doc.document_id == "ruling-456"

        # Dict representation
        dict_doc = normalize_source_document(
            {"document_id": "d2", "page_number": 5, "text_snippet": "raw snippet"}
        )
        assert dict_doc.document_id == "d2"
        assert dict_doc.page_number == 5

        # Plain string
        str_doc = normalize_source_document("مستند نصي بسيط")
        assert str_doc.document_id == "inline_context"
        assert str_doc.text_snippet == "مستند نصي بسيط"

        # Arbitrary object fallback
        obj_doc = normalize_source_document(12345)
        assert obj_doc.document_id == "unknown"
        assert obj_doc.text_snippet == "12345"


# ------------------------------------------------------------------------------
# Context Formatting Tests
# ------------------------------------------------------------------------------

class TestContextFormatting:
    """Tests for format_legal_context."""

    def test_format_empty_and_none_context(self) -> None:
        text, docs = format_legal_context(None)
        assert "لا يوجد سياق" in text
        assert docs == []

        text, docs = format_legal_context([])
        assert "لا يوجد سياق" in text
        assert docs == []

        text, docs = format_legal_context("")
        assert "لا يوجد سياق" in text
        assert docs == []

    def test_format_raw_string_context(self) -> None:
        raw_text = "المادة ١٠: يحق للمشتري فحص المبيع خلال أسبوع."
        text, docs = format_legal_context(raw_text)
        assert text == raw_text
        assert len(docs) == 1
        assert docs[0].text_snippet == raw_text
        assert docs[0].document_id == "inline_text"

    def test_format_sequence_of_sources(self, sample_sources: list[SourceDocument]) -> None:
        text, docs = format_legal_context(sample_sources)
        assert len(docs) == 2
        assert "commercial_agreement.pdf" in text
        assert "المادة الأولى" in text
        assert "المادة الثانية" in text
        assert "الصفحة: 1" in text
        assert "الصفحة: 2" in text
        assert "درجة الصلة: 0.920" in text
        assert "يحتوي على جداول" in text


# ------------------------------------------------------------------------------
# Prompt Template Tests
# ------------------------------------------------------------------------------

class TestPromptTemplates:
    """Tests for create_default_legal_prompt."""

    def test_default_legal_prompt_structure(self) -> None:
        prompt = create_default_legal_prompt()
        variables = prompt.input_variables
        assert "context" in variables
        assert "question" in variables

        formatted = prompt.format(context="سياق تجريبي", question="ما هي الالتزامات؟")
        assert "سياق تجريبي" in formatted
        assert "ما هي الالتزامات؟" in formatted
        assert "مستشار وباحث قانوني" in formatted

    def test_custom_system_prompt(self) -> None:
        custom_sys = "أنت محكم تجاري دولي متخصص في فض المنازعات."
        prompt = create_default_legal_prompt(system_prompt=custom_sys)
        formatted = prompt.format(context="سياق", question="سؤال")
        assert "أنت محكم تجاري دولي" in formatted

    def test_prompt_with_chat_history(self) -> None:
        prompt = create_default_legal_prompt(include_chat_history=True)
        assert "chat_history" in prompt.input_variables or "chat_history" in getattr(
            prompt, "optional_variables", []
        )


# ------------------------------------------------------------------------------
# LLM Factory Tests
# ------------------------------------------------------------------------------

class TestLLMFactory:
    """Tests for get_generation_llm."""

    def test_raise_when_no_keys_configured(self) -> None:
        settings = Settings(
            groq_api_key="",
            openai_api_key="",
        )
        with pytest.raises(NoLLMProviderConfiguredError, match="No LLM provider configured"):
            get_generation_llm(settings=settings)

    def test_force_groq_without_key_raises(self) -> None:
        settings = Settings(groq_api_key="", openai_api_key="valid-openai-key")
        with pytest.raises(NoLLMProviderConfiguredError, match="Groq API key is not configured"):
            get_generation_llm(settings=settings, force_provider="groq")

    def test_force_openai_without_key_raises(self) -> None:
        settings = Settings(groq_api_key="valid-groq-key", openai_api_key="")
        with pytest.raises(NoLLMProviderConfiguredError, match="OpenAI API key is not configured"):
            get_generation_llm(settings=settings, force_provider="openai")

    def test_instantiate_groq_when_key_present(self) -> None:
        settings = Settings(groq_api_key="gsk_test_key_12345")
        with patch("langchain_groq.ChatGroq") as mock_chat_groq:
            mock_instance = MagicMock()
            mock_chat_groq.return_value = mock_instance

            llm, model_id = get_generation_llm(settings=settings, force_provider="groq")
            assert llm == mock_instance
            assert "groq:" in model_id
            mock_chat_groq.assert_called_once()

    def test_instantiate_openai_when_key_present(self) -> None:
        settings = Settings(groq_api_key="", openai_api_key="sk-test-openai-key")
        with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
            mock_instance = MagicMock()
            mock_chat_openai.return_value = mock_instance

            llm, model_id = get_generation_llm(settings=settings, force_provider="openai")
            assert llm == mock_instance
            assert "openai:" in model_id
            mock_chat_openai.assert_called_once()


# ------------------------------------------------------------------------------
# LegalRAGChain Invocation & Streaming Tests
# ------------------------------------------------------------------------------

class TestLegalRAGChain:
    """Tests for LegalRAGChain core operations."""

    def test_chain_initialization(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        assert chain.llm == mock_llm
        assert chain.model_name == "mock-legal-llm"
        assert isinstance(chain.get_runnable(), Runnable)

    def test_create_legal_rag_chain_factory(self, mock_llm: MockLegalChatModel) -> None:
        chain = create_legal_rag_chain(llm=mock_llm)
        assert isinstance(chain, LegalRAGChain)
        assert chain.llm == mock_llm

    def test_invoke_with_rag_input(
        self, mock_llm: MockLegalChatModel, sample_sources: list[SourceDocument]
    ) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        rag_input = LegalRAGInput(
            question="ما هو التزام الطرف الأول؟",
            context=sample_sources,
        )

        output = chain.invoke(rag_input)
        assert isinstance(output, LegalRAGOutput)
        assert output.answer == "يلتزم الطرف الأول بتسليم العقار في الموعد المحدد."
        assert len(output.sources) == 2
        assert output.question == "ما هو التزام الطرف الأول؟"
        assert output.language == "ar"
        assert output.model_used == "mock-legal-llm"
        assert output.latency_seconds > 0.0

        output_dict = output.to_dict()
        assert output_dict["question"] == "ما هو التزام الطرف الأول؟"
        assert len(output_dict["sources"]) == 2

    def test_invoke_with_dict_input(
        self, mock_llm: MockLegalChatModel, sample_sources: list[SourceDocument]
    ) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        output = chain.invoke({
            "question": "ما هي مدة العقد؟",
            "context": sample_sources,
        })
        assert output.question == "ما هي مدة العقد؟"
        assert len(output.sources) == 2
        assert "تسليم العقار" in output.answer

    def test_invoke_with_english_response(self) -> None:
        en_llm = MockLegalChatModel(
            response_text="The first party is obligated to deliver the goods on time."
        )
        chain = LegalRAGChain(llm=en_llm)
        output = chain.invoke({"question": "What is the obligation?", "context": "Plain text context"})
        assert output.language == "en"
        assert "obligated" in output.answer

    def test_invoke_empty_question_raises_value_error(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        with pytest.raises(ValueError, match="question cannot be empty"):
            chain.invoke({"question": "   ", "context": []})

    def test_invoke_invalid_input_type_raises_value_error(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        with pytest.raises(ValueError, match="Expected LegalRAGInput or dict"):
            chain.invoke(12345)  # type: ignore

    def test_invoke_failure_wraps_in_legal_generation_error(self) -> None:
        failing_chain = LegalRAGChain(llm=FailingLegalChatModel())
        with pytest.raises(LegalGenerationError, match="Legal generation failed"):
            failing_chain.invoke({"question": "سؤال", "context": "سياق"})

    @pytest.mark.asyncio
    async def ainvoke_test(
        self, mock_llm: MockLegalChatModel, sample_sources: list[SourceDocument]
    ) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        rag_input = LegalRAGInput(
            question="ما هو التزام الطرف الأول؟",
            context=sample_sources,
        )
        output = await chain.ainvoke(rag_input)
        assert isinstance(output, LegalRAGOutput)
        assert output.answer == "يلتزم الطرف الأول بتسليم العقار في الموعد المحدد."
        assert len(output.sources) == 2
        assert output.latency_seconds > 0.0

    @pytest.mark.asyncio
    async def ainvoke_failure_wraps_in_legal_generation_error(self) -> None:
        failing_chain = LegalRAGChain(llm=FailingLegalChatModel())
        with pytest.raises(LegalGenerationError, match="Async legal generation failed"):
            await failing_chain.ainvoke({"question": "سؤال", "context": "سياق"})

    def test_stream(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        tokens = list(chain.stream({"question": "سؤال", "context": "سياق"}))
        assert len(tokens) > 0
        assert "يلتزم " in tokens

    def test_stream_failure_wraps_in_legal_generation_error(self) -> None:
        failing_chain = LegalRAGChain(llm=FailingLegalChatModel())
        with pytest.raises(LegalGenerationError, match="Streaming generation failed"):
            list(failing_chain.stream({"question": "سؤال", "context": "سياق"}))

    @pytest.mark.asyncio
    async def astream_test(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        collected = []
        async for token in chain.astream({"question": "سؤال", "context": "سياق"}):
            collected.append(token)
        assert len(collected) > 0
        assert "يلتزم " in collected

    @pytest.mark.asyncio
    async def astream_failure_wraps_in_legal_generation_error(self) -> None:
        failing_chain = LegalRAGChain(llm=FailingLegalChatModel())
        with pytest.raises(LegalGenerationError, match="Async streaming generation failed"):
            async for _ in failing_chain.astream({"question": "سؤال", "context": "سياق"}):
                pass

    @pytest.mark.asyncio
    async def astream_events_test(self, mock_llm: MockLegalChatModel) -> None:
        chain = LegalRAGChain(llm=mock_llm)
        events = []
        async for ev in chain.astream_events({"question": "سؤال", "context": "سياق"}):
            events.append(ev)
        assert len(events) > 0
        event_names = [e["event"] for e in events]
        assert "on_chain_start" in event_names
        assert "on_chain_end" in event_names
