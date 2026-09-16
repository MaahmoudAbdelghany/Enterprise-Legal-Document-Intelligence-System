"""LangChain LCEL generation chains, prompts, and streaming pipelines."""

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
)

__all__ = [
    "ContextFormattingError",
    "LegalGenerationError",
    "LegalRAGChain",
    "LegalRAGInput",
    "LegalRAGOutput",
    "NoLLMProviderConfiguredError",
    "SourceDocument",
    "create_default_legal_prompt",
    "create_legal_rag_chain",
    "format_legal_context",
    "get_generation_llm",
]
