"""Unit tests for app.core.generation.prompts."""

from __future__ import annotations

from typing import Any
import pytest

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate

from app.core.generation.chain import LegalRAGChain, LegalRAGInput
from app.core.generation.prompts import (
    COMPLIANCE_AUDIT_SYSTEM_PROMPT,
    COMPLIANCE_AUDIT_SYSTEM_PROMPT_EN,
    CONTRACT_ANALYSIS_SYSTEM_PROMPT,
    CONTRACT_ANALYSIS_SYSTEM_PROMPT_EN,
    COURT_RULING_SYSTEM_PROMPT,
    COURT_RULING_SYSTEM_PROMPT_EN,
    DEFAULT_HUMAN_PROMPT,
    DEFAULT_HUMAN_PROMPT_EN,
    DEFAULT_LEGAL_SYSTEM_PROMPT,
    DEFAULT_LEGAL_SYSTEM_PROMPT_EN,
    EXTRACTION_HUMAN_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT_EN,
    HUMAN_PROMPTS_AR,
    HUMAN_PROMPTS_EN,
    LEGAL_SUMMARIZATION_SYSTEM_PROMPT,
    LEGAL_SUMMARIZATION_SYSTEM_PROMPT_EN,
    STRICT_GROUNDING_RULES_AR,
    STRICT_GROUNDING_RULES_EN,
    SUMMARIZATION_HUMAN_PROMPT,
    SYSTEM_PROMPTS_AR,
    SYSTEM_PROMPTS_EN,
    create_compliance_audit_prompt,
    create_contract_analysis_prompt,
    create_court_ruling_prompt,
    create_default_legal_prompt,
    create_legal_prompt,
    create_legal_summary_prompt,
    get_human_prompt,
    get_system_prompt,
)
from tests.test_generation_chain import MockLegalChatModel


# ------------------------------------------------------------------------------
# Test Prompt Content & Grounding Rules
# ------------------------------------------------------------------------------

def test_strict_grounding_rules_arabic() -> None:
    """Verify Arabic grounding directives contain mandatory legal constraints."""
    assert "Strict Grounding" in STRICT_GROUNDING_RULES_AR
    assert "المعلومة المطلوبة غير مذكورة في المستندات المرفقة" in STRICT_GROUNDING_RULES_AR
    assert "Precise Citations" in STRICT_GROUNDING_RULES_AR
    assert "المستند:" in STRICT_GROUNDING_RULES_AR
    assert "الصفحة:" in STRICT_GROUNDING_RULES_AR


def test_strict_grounding_rules_english() -> None:
    """Verify English grounding directives contain mandatory legal constraints."""
    assert "Strict Grounding" in STRICT_GROUNDING_RULES_EN
    assert "The requested information is not mentioned in the provided documents." in STRICT_GROUNDING_RULES_EN
    assert "Precise Citations" in STRICT_GROUNDING_RULES_EN


def test_default_system_prompts_contain_grounding() -> None:
    """Verify built-in prompts integrate strict grounding rules."""
    assert STRICT_GROUNDING_RULES_AR in DEFAULT_LEGAL_SYSTEM_PROMPT
    assert STRICT_GROUNDING_RULES_EN in DEFAULT_LEGAL_SYSTEM_PROMPT_EN
    assert STRICT_GROUNDING_RULES_AR in CONTRACT_ANALYSIS_SYSTEM_PROMPT
    assert STRICT_GROUNDING_RULES_AR in COURT_RULING_SYSTEM_PROMPT
    assert STRICT_GROUNDING_RULES_AR in COMPLIANCE_AUDIT_SYSTEM_PROMPT
    assert STRICT_GROUNDING_RULES_AR in LEGAL_SUMMARIZATION_SYSTEM_PROMPT
    assert STRICT_GROUNDING_RULES_AR in EXTRACTION_SYSTEM_PROMPT


def test_system_prompt_catalog_completeness() -> None:
    """Ensure catalog has entries for all legal task types in both languages."""
    expected_tasks = {"qa", "contract", "ruling", "compliance", "summary", "extraction"}
    assert set(SYSTEM_PROMPTS_AR.keys()) == expected_tasks
    assert set(SYSTEM_PROMPTS_EN.keys()) == expected_tasks
    assert set(HUMAN_PROMPTS_AR.keys()) == expected_tasks
    assert set(HUMAN_PROMPTS_EN.keys()) == expected_tasks


# ------------------------------------------------------------------------------
# Test Prompt Resolution Helpers
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("task_type", ["qa", "contract", "ruling", "compliance", "summary", "extraction"])
def test_get_system_prompt_arabic(task_type: Any) -> None:
    """Verify retrieval of Arabic system prompts."""
    prompt = get_system_prompt(task_type=task_type, language="ar")
    assert prompt == SYSTEM_PROMPTS_AR[task_type]
    assert len(prompt) > 50


@pytest.mark.parametrize("task_type", ["qa", "contract", "ruling", "compliance", "summary", "extraction"])
def test_get_system_prompt_english(task_type: Any) -> None:
    """Verify retrieval of English system prompts."""
    prompt = get_system_prompt(task_type=task_type, language="en")
    assert prompt == SYSTEM_PROMPTS_EN[task_type]
    assert len(prompt) > 50


def test_get_system_prompt_custom_override() -> None:
    """Verify custom system prompt takes precedence."""
    custom = "أنت مستشار قانوني خاص بشركات التأمين."
    res = get_system_prompt(task_type="qa", language="ar", custom_override=custom)
    assert res == custom


def test_get_human_prompt_resolution() -> None:
    """Verify human prompt retrieval and overrides."""
    ar_qa = get_human_prompt("qa", "ar")
    assert "{context}" in ar_qa
    assert "{question}" in ar_qa

    en_qa = get_human_prompt("qa", "en")
    assert "Attached Document Context:" in en_qa

    summary_ar = get_human_prompt("summary", "ar")
    assert "المستندات القانونية المراد تلخيصها:" in summary_ar

    custom = "السياق:\n{context}\nالاستفسار:\n{question}"
    res = get_human_prompt("qa", "ar", custom_override=custom)
    assert res == custom


# ------------------------------------------------------------------------------
# Test Prompt Builders
# ------------------------------------------------------------------------------

def test_create_legal_prompt_basic() -> None:
    """Verify create_legal_prompt creates a valid ChatPromptTemplate."""
    prompt = create_legal_prompt(task_type="qa", language="ar")
    assert isinstance(prompt, ChatPromptTemplate)
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables

    formatted = prompt.format_messages(
        context="محتوى المادة 5",
        question="ما هي شروط العقد؟",
    )
    assert len(formatted) == 2
    assert isinstance(formatted[0], SystemMessage)
    assert isinstance(formatted[1], HumanMessage)
    assert "محتوى المادة 5" in formatted[1].content
    assert "ما هي شروط العقد؟" in formatted[1].content


def test_create_legal_prompt_with_history() -> None:
    """Verify create_legal_prompt includes chat_history placeholder."""
    prompt = create_legal_prompt(task_type="qa", language="ar", include_chat_history=True)
    assert "chat_history" in prompt.input_variables or "chat_history" in prompt.optional_variables

    history = [
        HumanMessage(content="سؤال سابق"),
        AIMessage(content="إجابة سابقة"),
    ]
    formatted = prompt.format_messages(
        context="سياق تجريبي",
        question="سؤال لاحق",
        chat_history=history,
    )
    assert len(formatted) == 4
    assert isinstance(formatted[0], SystemMessage)
    assert formatted[1] == history[0]
    assert formatted[2] == history[1]
    assert isinstance(formatted[3], HumanMessage)


def test_create_legal_prompt_with_additional_instructions() -> None:
    """Verify additional_instructions are appended to the system message."""
    extra = "يرجى حصر المسؤوليات المالية في جدول مقارن."
    prompt = create_legal_prompt(
        task_type="contract",
        language="ar",
        additional_instructions=extra,
    )
    formatted = prompt.format_messages(context="سياق", question="سؤال")
    assert extra in formatted[0].content


def test_create_default_legal_prompt() -> None:
    """Verify create_default_legal_prompt matches standard API."""
    prompt = create_default_legal_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    formatted = prompt.format_messages(context="بند 1", question="سؤال")
    assert len(formatted) == 2


def test_specialized_builders() -> None:
    """Verify all specialized prompt factory functions."""
    contract_p = create_contract_analysis_prompt(language="ar")
    assert "Contract Review Specialist" in contract_p.format_messages(context="c", question="q")[0].content

    ruling_p = create_court_ruling_prompt(language="ar")
    assert "حيثيات وأسباب الحكم" in ruling_p.format_messages(context="c", question="q")[0].content

    compliance_p = create_compliance_audit_prompt(language="ar")
    assert "Legal Compliance & Risk Auditor" in compliance_p.format_messages(context="c", question="q")[0].content

    summary_p = create_legal_summary_prompt(language="ar")
    assert "التلخيص التنفيذي" in summary_p.format_messages(context="c", question="q")[0].content


# ------------------------------------------------------------------------------
# Test Integration with LegalRAGChain
# ------------------------------------------------------------------------------

def test_legal_rag_chain_with_custom_prompt() -> None:
    """Verify LegalRAGChain executes successfully with specialized prompts."""
    custom_prompt = create_contract_analysis_prompt(language="ar")
    mock_llm = MockLegalChatModel(response_text="تم فحص العقد: لا توجد شروط جزائية مجحفة.")

    chain = LegalRAGChain(llm=mock_llm, prompt=custom_prompt)
    output = chain.invoke(
        LegalRAGInput(
            question="حلل الشروط الجزائية في العقد",
            context="المادة 12: غرامة التأخير 1% أسبوعياً.",
        )
    )

    assert output.answer == "تم فحص العقد: لا توجد شروط جزائية مجحفة."
    assert len(output.sources) == 1
    assert output.language == "ar"
