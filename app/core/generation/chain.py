"""LangChain LCEL RAG generation chain for LexisGraph.

Provides an enterprise-grade LangChain Expression Language (LCEL) retrieval-augmented
generation chain specialized in Arabic and multilingual legal document intelligence.
Handles context formatting, bilingual legal prompts, primary Groq inference with
OpenAI fallback, streaming tokens, and structured source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnablePassthrough
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    from app.core.retrieval.qdrant_store import QdrantSearchResult

from app.config import Settings, get_settings


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class LegalGenerationError(Exception):
    """Base exception for legal generation chain errors."""


class NoLLMProviderConfiguredError(LegalGenerationError):
    """Raised when neither Groq nor OpenAI API keys are configured."""


class ContextFormattingError(LegalGenerationError):
    """Raised when formatting retrieved documents into context fails."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceDocument:
    """Represents a retrieved document page cited in the legal answer.

    Attributes:
        document_id: Unique identifier of the document.
        page_number: 1-indexed page number within the document.
        score: Relevance or similarity score (e.g., MaxSim or RRF score).
        filename: Original file name of the document.
        text_snippet: Text excerpt or full verbatim text from the page.
        metadata: Arbitrary metadata dictionary (court, date, parties, etc.).
        has_tables: Whether the cited page includes tabular information.
    """

    document_id: str
    page_number: int
    score: float = 0.0
    filename: str = ""
    text_snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    has_tables: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize source document to dictionary."""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "score": self.score,
            "filename": self.filename,
            "text_snippet": self.text_snippet,
            "metadata": dict(self.metadata),
            "has_tables": self.has_tables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceDocument:
        """Instantiate SourceDocument from a dictionary."""
        return cls(
            document_id=str(data.get("document_id", "")),
            page_number=int(data.get("page_number", 1)),
            score=float(data.get("score", 0.0)),
            filename=str(data.get("filename", "")),
            text_snippet=str(data.get("text_snippet", data.get("raw_text", ""))),
            metadata=dict(data.get("metadata", {})),
            has_tables=bool(data.get("has_tables", False)),
        )

    @classmethod
    def from_qdrant_result(cls, result: Any) -> SourceDocument:
        """Instantiate SourceDocument from a QdrantSearchResult or similar object."""
        snippet = getattr(result, "raw_text", "") or getattr(result, "normalized_text", "")
        return cls(
            document_id=str(getattr(result, "document_id", "")),
            page_number=int(getattr(result, "page_number", 1)),
            score=float(getattr(result, "score", 0.0)),
            filename=str(getattr(result, "filename", "")),
            text_snippet=snippet,
            metadata=dict(getattr(result, "metadata", {})),
            has_tables=bool(getattr(result, "has_tables", False)),
        )


@dataclass(frozen=True)
class LegalRAGInput:
    """Input parameters for the legal RAG generation chain.

    Attributes:
        question: User query or legal research prompt.
        context: Retrieved documents, search results, or pre-formatted context string.
        chat_history: Optional conversation history for multi-turn dialogues.
        language: Preferred output language ('ar', 'en', or 'auto').
        legal_domain: Optional legal domain specialization (e.g. 'commercial', 'labor').
        system_instructions: Optional user-provided custom system instructions.
    """

    question: str
    context: Sequence[SourceDocument | Any] | str = field(default_factory=tuple)
    chat_history: Sequence[BaseMessage | dict[str, str]] | None = None
    language: Literal["ar", "en", "auto"] = "auto"
    legal_domain: str = "general"
    system_instructions: str | None = None


@dataclass(frozen=True)
class LegalRAGOutput:
    """Structured output generated by the legal RAG chain.

    Attributes:
        answer: Generated legal answer text.
        sources: List of SourceDocument instances cited or referenced.
        question: Original query submitted by the user.
        language: Language of the generated answer.
        model_used: Identifier of the LLM model that performed generation.
        latency_seconds: Total inference and pipeline elapsed time in seconds.
        metadata: Additional diagnostic or telemetry metadata.
    """

    answer: str
    sources: list[SourceDocument] = field(default_factory=list)
    question: str = ""
    language: str = "ar"
    model_used: str = ""
    latency_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert output to a JSON-serializable dictionary."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "question": self.question,
            "language": self.language,
            "model_used": self.model_used,
            "latency_seconds": round(self.latency_seconds, 4),
            "metadata": dict(self.metadata),
        }


# ------------------------------------------------------------------------------
# Context Formatting
# ------------------------------------------------------------------------------

def normalize_source_document(item: Any) -> SourceDocument:
    """Normalize heterogeneous source items into a typed SourceDocument.

    Supports:
        - SourceDocument instances
        - QdrantSearchResult instances
        - Dicts matching either schema
        - Plain text strings (converted to a generic document)

    Args:
        item: Raw item from vector search or context list.

    Returns:
        Standardized SourceDocument instance.
    """
    if isinstance(item, SourceDocument):
        return item
    if hasattr(item, "document_id") and hasattr(item, "page_number"):
        return SourceDocument.from_qdrant_result(item)
    if isinstance(item, dict):
        return SourceDocument.from_dict(item)
    if isinstance(item, str):
        return SourceDocument(
            document_id="inline_context",
            page_number=1,
            text_snippet=item.strip(),
        )
    return SourceDocument(
        document_id="unknown",
        page_number=1,
        text_snippet=str(item).strip(),
    )


def format_legal_context(
    sources: Sequence[SourceDocument | Any] | str | None,
) -> tuple[str, list[SourceDocument]]:
    """Format heterogeneous retrieval results into structured prompt context.

    Each page is formatted with clear bounding markers indicating document name,
    document ID, and page number to guide LLM citation grounding.

    Args:
        sources: Sequence of sources or raw pre-formatted context string.

    Returns:
        Tuple of (formatted_context_string, normalized_sources_list).
    """
    if sources is None:
        return "(لا يوجد سياق مستندي متاح)", []

    if isinstance(sources, str):
        cleaned_str = sources.strip()
        if not cleaned_str:
            return "(لا يوجد سياق مستندي متاح)", []
        doc = SourceDocument(
            document_id="inline_text",
            page_number=1,
            text_snippet=cleaned_str,
        )
        return cleaned_str, [doc]

    if not sources:
        return "(لا يوجد سياق مستندي متاح)", []

    normalized_list: list[SourceDocument] = []
    formatted_chunks: list[str] = []

    for idx, raw_item in enumerate(sources, start=1):
        doc = normalize_source_document(raw_item)
        normalized_list.append(doc)

        header_parts: list[str] = []
        if doc.filename:
            header_parts.append(f"المستند: {doc.filename}")
        elif doc.document_id:
            header_parts.append(f"المعرف: {doc.document_id}")
        else:
            header_parts.append(f"المرجع: #{idx}")

        header_parts.append(f"الصفحة: {doc.page_number}")
        if doc.score > 0.0:
            header_parts.append(f"درجة الصلة: {doc.score:.3f}")
        if doc.has_tables:
            header_parts.append("يحتوي على جداول")

        header_line = " | ".join(header_parts)
        text_body = doc.text_snippet.strip() or "(النص غير متوفر لهذه الصفحة)"

        chunk = (
            f"--- [مرجع {idx}] ({header_line}) ---\n"
            f"{text_body}\n"
            f"--- [نهاية مرجع {idx}] ---"
        )
        formatted_chunks.append(chunk)

    return "\n\n".join(formatted_chunks), normalized_list


# ------------------------------------------------------------------------------
# Prompt Templates
# ------------------------------------------------------------------------------

DEFAULT_LEGAL_SYSTEM_PROMPT = """أنت مستشار وباحث قانوني رقمي فائق الذكاء، متخصص في تحليل وتفسير المستندات والعقود واللوائح والقرارات القضائية باللغتين العربية والإنجليزية.

مهمتك الأساسية:
تقديم إجابات قانونية بالغة الدقة والاحترافية بالاعتماد الحصري والصارم على "السياق المستندي المرفق" فقط.

القواعد الإلزامية التي يجب الالتزام بها دون استثناء:
1. الأمانة والتحري الصارم (Strict Grounding):
   - استخرج إجابتك حصراً من النصوص المذكورة في سياق المستندات أدناه.
   - يُحظر تماماً اختلاق أو تخمين أو افتراض أي بنود أو وقائع أو التزامات غير منصوص عليها صراحة في السياق.
   - إذا كان السياق لا يحتوي على إجابة السؤال كلياً أو جزئياً، فصرّح بوضوح ومهنية:
     "المعلومة المطلوبة غير مذكورة في المستندات المرفقة"، واذكر فقط ما يتصل بها من حقائق إن وُجدت.

2. التوثيق والإسناد الدقيق (Precise Citations):
   - عند ذكر أي شرط، التزام، غرامة، أو حكم، اذكر المصدر بدقة بين قوسين، على سبيل المثال:
     (اسم المستند / المعرف: [اسم المستند]، الصفحة: [رقم الصفحة]، المادة/البند: [رقم المادة إن وُجد]).

3. الصياغة القانونية الرصينة:
   - استخدم المصطلحات القانونية المعتمدة (مثل: الطرف الأول، الطرف الثاني، الديباجة، الالتزامات الجوهرية، التعويض الاتفاقي، القوة القاهرة، فسخ العقد، الإخطار الكتابي).
   - رتّب الإجابة في نقاط واضحة أو فقرات محكمة لتسهيل اتخاذ القرار القانوني.

4. لغة الإجابة:
   - أجب باللغة العربية الفصحى القانونية افتراضياً، أو باللغة الإنجليزية إذا كان السؤال مطروحاً بالإنجليزية.
"""

DEFAULT_HUMAN_PROMPT = """السياق المستندي المرفق:
{context}

السؤال القانوني:
{question}

الإجابة القانونية الموثقة:"""


def create_default_legal_prompt(
    system_prompt: str | None = None,
    include_chat_history: bool = False,
) -> ChatPromptTemplate:
    """Create a LangChain ChatPromptTemplate configured for legal document QA.

    Args:
        system_prompt: Optional custom system prompt overriding default.
        include_chat_history: If True, inserts MessagesPlaceholder for chat_history.

    Returns:
        ChatPromptTemplate ready for LCEL binding.
    """
    sys_text = system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_LEGAL_SYSTEM_PROMPT
    messages: list[Any] = [
        SystemMessagePromptTemplate.from_template(sys_text),
    ]

    if include_chat_history:
        messages.append(MessagesPlaceholder(variable_name="chat_history", optional=True))

    messages.append(HumanMessagePromptTemplate.from_template(DEFAULT_HUMAN_PROMPT))

    return ChatPromptTemplate.from_messages(messages)


# ------------------------------------------------------------------------------
# LLM Factory & Resolution
# ------------------------------------------------------------------------------

def get_generation_llm(
    settings: Settings | None = None,
    streaming: bool = False,
    force_provider: Literal["groq", "openai", "auto"] = "auto",
) -> tuple[BaseChatModel, str]:
    """Resolve and instantiate the primary or fallback LangChain ChatModel.

    Priority:
        1. Groq ChatGroq (`llama-3.3-70b-versatile` by default)
        2. OpenAI ChatOpenAI (`gpt-4o` fallback)

    Args:
        settings: Application Settings (default loaded from singleton).
        streaming: Enable token streaming mode on the model.
        force_provider: Explicitly enforce 'groq', 'openai', or 'auto'.

    Returns:
        Tuple of (BaseChatModel instance, model_identifier_string).

    Raises:
        NoLLMProviderConfiguredError: If chosen or available provider has no API key.
    """
    cfg = settings or get_settings()

    # Determine provider
    use_groq = False
    use_openai = False

    if force_provider == "groq":
        use_groq = True
    elif force_provider == "openai":
        use_openai = True
    else:  # auto
        if cfg.has_groq:
            use_groq = True
        elif cfg.has_openai:
            use_openai = True

    if use_groq:
        if not cfg.groq_api_key.strip():
            raise NoLLMProviderConfiguredError(
                "Groq API key is not configured in settings. Set GROQ_API_KEY environment variable."
            )
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                groq_api_key=cfg.groq_api_key,
                model_name=cfg.groq_model,
                temperature=cfg.groq_temperature,
                max_retries=cfg.groq_max_retries,
                streaming=streaming,
            )
            logger.info(f"Initialized ChatGroq with model '{cfg.groq_model}' (streaming={streaming})")
            return llm, f"groq:{cfg.groq_model}"
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {e}")
            if cfg.has_openai and force_provider == "auto":
                logger.warning("Attempting automatic fallback to OpenAI ChatOpenAI...")
                use_openai = True
            else:
                raise LegalGenerationError(f"Failed to instantiate Groq LLM: {e}") from e

    if use_openai:
        if not cfg.openai_api_key.strip():
            raise NoLLMProviderConfiguredError(
                "OpenAI API key is not configured in settings. Set OPENAI_API_KEY environment variable."
            )
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=cfg.openai_api_key,
                model=cfg.openai_model,
                temperature=cfg.groq_temperature,
                streaming=streaming,
            )
            logger.info(f"Initialized ChatOpenAI with model '{cfg.openai_model}' (streaming={streaming})")
            return llm, f"openai:{cfg.openai_model}"
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise LegalGenerationError(f"Failed to instantiate OpenAI LLM: {e}") from e

    raise NoLLMProviderConfiguredError(
        "No LLM provider configured. Please set GROQ_API_KEY or OPENAI_API_KEY in .env."
    )


# ------------------------------------------------------------------------------
# Legal RAG LCEL Chain Class
# ------------------------------------------------------------------------------

class LegalRAGChain:
    """Enterprise LCEL RAG generation chain for legal intelligence.

    Implements the core LangChain Expression Language pipeline:
        `prompt | llm | output_parser`
    with automatic input normalization, context formatting, source attribution,
    streaming token generation, and bilingual legal reasoning.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        prompt: BasePromptTemplate | None = None,
        settings: Settings | None = None,
        streaming: bool = False,
    ) -> None:
        """Initialize LegalRAGChain with LLM, prompt, and settings.

        Args:
            llm: Optional pre-configured LangChain BaseChatModel instance.
            prompt: Optional custom BasePromptTemplate instance.
            settings: Application settings.
            streaming: Whether the chain LLM should enable token streaming.
        """
        self._settings = settings or get_settings()
        self._streaming = streaming

        if llm is not None:
            self._llm = llm
            self._model_name = getattr(llm, "model_name", getattr(llm, "model", str(type(llm).__name__)))
        else:
            self._llm, self._model_name = get_generation_llm(self._settings, streaming=streaming)

        self._prompt = prompt or create_default_legal_prompt()
        self._parser = StrOutputParser()

        # Build raw LCEL execution graph
        self._runnable: Runnable = self._prompt | self._llm | self._parser
        logger.debug(f"LegalRAGChain initialized with model '{self._model_name}'.")

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def llm(self) -> BaseChatModel:
        """Return the active LangChain chat model."""
        return self._llm

    @property
    def prompt(self) -> BasePromptTemplate:
        """Return the active prompt template."""
        return self._prompt

    @property
    def model_name(self) -> str:
        """Return active model name or identifier."""
        return self._model_name

    def get_runnable(self) -> Runnable:
        """Return the underlying LangChain LCEL Runnable."""
        return self._runnable

    # --------------------------------------------------------------------------
    # Internal Input Parsing
    # --------------------------------------------------------------------------

    def _prepare_payload(
        self,
        input_data: LegalRAGInput | dict[str, Any],
    ) -> tuple[dict[str, Any], list[SourceDocument], str]:
        """Normalize raw input into LCEL variables and extracted sources.

        Args:
            input_data: LegalRAGInput or dictionary.

        Returns:
            Tuple of (lcel_inputs_dict, normalized_sources_list, question_string).
        """
        if isinstance(input_data, LegalRAGInput):
            q = input_data.question
            ctx_raw = input_data.context
            chat_hist = input_data.chat_history
        elif isinstance(input_data, dict):
            q = str(input_data.get("question", ""))
            ctx_raw = input_data.get("context", ())
            chat_hist = input_data.get("chat_history")
        else:
            raise ValueError(f"Expected LegalRAGInput or dict, got {type(input_data)}")

        if not q or not q.strip():
            raise ValueError("Query question cannot be empty.")

        formatted_context, sources = format_legal_context(ctx_raw)

        chain_inputs: dict[str, Any] = {
            "context": formatted_context,
            "question": q.strip(),
        }
        if chat_hist is not None:
            chain_inputs["chat_history"] = list(chat_hist)

        return chain_inputs, sources, q.strip()

    # --------------------------------------------------------------------------
    # Synchronous Execution
    # --------------------------------------------------------------------------

    def invoke(
        self,
        input_data: LegalRAGInput | dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> LegalRAGOutput:
        """Execute the LCEL RAG chain synchronously.

        Args:
            input_data: LegalRAGInput instance or dictionary.
            config: Optional LangChain RunnableConfig (callbacks, tags, metadata).

        Returns:
            Structured LegalRAGOutput with answer and sources.
        """
        start_time = time.perf_counter()
        payload, sources, question = self._prepare_payload(input_data)

        try:
            answer_text = self._runnable.invoke(payload, config=config)
            elapsed = time.perf_counter() - start_time

            # Basic language detection heuristic
            lang = "ar"
            if any("\u0600" <= c <= "\u06ff" for c in answer_text):
                lang = "ar"
            elif any(c.isascii() and c.isalpha() for c in answer_text):
                lang = "en"

            return LegalRAGOutput(
                answer=answer_text.strip(),
                sources=sources,
                question=question,
                language=lang,
                model_used=self._model_name,
                latency_seconds=elapsed,
            )
        except Exception as e:
            logger.error(f"Error during LegalRAGChain.invoke: {e}")
            raise LegalGenerationError(f"Legal generation failed: {e}") from e

    def stream(
        self,
        input_data: LegalRAGInput | dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> Iterator[str]:
        """Stream generated answer tokens synchronously.

        Args:
            input_data: LegalRAGInput instance or dictionary.
            config: Optional LangChain RunnableConfig.

        Yields:
            Incremental string tokens as they are produced by the LLM.
        """
        payload, _, _ = self._prepare_payload(input_data)
        try:
            for chunk in self._runnable.stream(payload, config=config):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Error during LegalRAGChain.stream: {e}")
            raise LegalGenerationError(f"Streaming generation failed: {e}") from e

    # --------------------------------------------------------------------------
    # Asynchronous Execution
    # --------------------------------------------------------------------------

    async def ainvoke(
        self,
        input_data: LegalRAGInput | dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> LegalRAGOutput:
        """Execute the LCEL RAG chain asynchronously.

        Args:
            input_data: LegalRAGInput instance or dictionary.
            config: Optional LangChain RunnableConfig.

        Returns:
            Structured LegalRAGOutput with answer and sources.
        """
        start_time = time.perf_counter()
        payload, sources, question = self._prepare_payload(input_data)

        try:
            answer_text = await self._runnable.ainvoke(payload, config=config)
            elapsed = time.perf_counter() - start_time

            lang = "ar"
            if any("\u0600" <= c <= "\u06ff" for c in answer_text):
                lang = "ar"
            elif any(c.isascii() and c.isalpha() for c in answer_text):
                lang = "en"

            return LegalRAGOutput(
                answer=answer_text.strip(),
                sources=sources,
                question=question,
                language=lang,
                model_used=self._model_name,
                latency_seconds=elapsed,
            )
        except Exception as e:
            logger.error(f"Error during LegalRAGChain.ainvoke: {e}")
            raise LegalGenerationError(f"Async legal generation failed: {e}") from e

    async def astream(
        self,
        input_data: LegalRAGInput | dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated answer tokens asynchronously.

        Args:
            input_data: LegalRAGInput instance or dictionary.
            config: Optional LangChain RunnableConfig.

        Yields:
            Incremental string tokens as they are produced by the LLM.
        """
        payload, _, _ = self._prepare_payload(input_data)
        try:
            async for chunk in self._runnable.astream(payload, config=config):
                if chunk:
                    yield chunk
        except Exception as e:
            logger.error(f"Error during LegalRAGChain.astream: {e}")
            raise LegalGenerationError(f"Async streaming generation failed: {e}") from e

    async def astream_events(
        self,
        input_data: LegalRAGInput | dict[str, Any],
        version: str = "v2",
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events (tokens, prompt formatting, tool calls) asynchronously.

        Args:
            input_data: LegalRAGInput instance or dictionary.
            version: LangChain event streaming schema version ('v1' or 'v2').
            config: Optional LangChain RunnableConfig.

        Yields:
            Event dictionaries matching LangChain standard event streaming format.
        """
        payload, _, _ = self._prepare_payload(input_data)
        try:
            async for event in self._runnable.astream_events(payload, version=version, config=config):
                yield event
        except Exception as e:
            logger.error(f"Error during LegalRAGChain.astream_events: {e}")
            raise LegalGenerationError(f"Event streaming failed: {e}") from e


# ------------------------------------------------------------------------------
# Factory Function
# ------------------------------------------------------------------------------

def create_legal_rag_chain(
    llm: BaseChatModel | None = None,
    prompt: BasePromptTemplate | None = None,
    settings: Settings | None = None,
    streaming: bool = False,
) -> LegalRAGChain:
    """Convenience factory function to instantiate a LegalRAGChain.

    Args:
        llm: Optional pre-configured LangChain BaseChatModel instance.
        prompt: Optional custom prompt template.
        settings: Application settings.
        streaming: Enable token streaming mode.

    Returns:
        Configured LegalRAGChain instance.
    """
    return LegalRAGChain(
        llm=llm,
        prompt=prompt,
        settings=settings,
        streaming=streaming,
    )
