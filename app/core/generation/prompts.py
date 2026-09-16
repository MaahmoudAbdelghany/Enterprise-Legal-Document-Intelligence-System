"""Arabic-aware legal prompt templates and prompt builders for LexisGraph.

Provides specialized, enterprise-grade prompts tailored for Arabic and multilingual
legal document intelligence. Enforces strict factual grounding against retrieved
context, mandatory page and document citations, structured analytical breakdowns,
and formal legal drafting standards (الصياغة القانونية الرصينة).

Covers diverse legal domains and operational tasks:
    - Legal QA & Statutory Interpretation (الاستشارات القانونية وتفسير النصوص)
    - Contract Review & Clause Analysis (تحليل وتدقيق العقود والاتفاقيات)
    - Judicial Rulings & Court Judgment Analysis (تحليل الأحكام والقرارات القضائية)
    - Regulatory Compliance & Legal Risk Auditing (التدقيق والامتثال القانوني)
    - Executive Legal Summarization (التلخيص التنفيذي للمستندات القانونية)
    - Legal Entity & Citation Extraction (استخراج الكيانات والاستشهادات النظامية)
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# ------------------------------------------------------------------------------
# Type Aliases & Constants
# ------------------------------------------------------------------------------

LegalTaskType = Literal[
    "qa",
    "contract",
    "ruling",
    "compliance",
    "summary",
    "extraction",
]

PromptLanguage = Literal["ar", "en", "auto"]


# ------------------------------------------------------------------------------
# Grounding & Anti-Hallucination Directives
# ------------------------------------------------------------------------------

STRICT_GROUNDING_RULES_AR = """القواعد الإلزامية التي يجب الالتزام بها دون استثناء:
1. الأمانة والتحري الصارم (Strict Grounding):
   - استخرج إجابتك حصراً من النصوص المذكورة في سياق المستندات أدناه.
   - يُحظر تماماً اختلاق أو تخمين أو افتراض أي بنود أو وقائع أو التزامات غير منصوص عليها صراحة في السياق.
   - إذا كان السياق لا يحتوي على إجابة السؤال كلياً أو جزئياً، فصرّح بوضوح ومهنية:
     "المعلومة المطلوبة غير مذكورة في المستندات المرفقة"، واذكر فقط ما يتصل بها من حقائق إن وُجدت.

2. التوثيق والإسناد الدقيق (Precise Citations):
   - عند ذكر أي شرط، التزام، غرامة، حكم، أو تاريخ، اذكر المصدر بدقة بين أقواس:
     (المستند: [اسم المستند أو المعرف]، الصفحة: [رقم الصفحة]، المادة/البند: [رقم المادة أو عنوان البند إن وُجد]).
   - لا تذكر أي رقم مادة أو نسبة مالية دون الاستناد الصريح إلى الصفحة المصدرية.

3. الصياغة القانونية الرصينة:
   - استخدم المصطلحات القانونية المعتمدة (مثل: الطرف الأول، الطرف الثاني، الديباجة، الالتزامات الجوهرية، التعويض الاتفاقي، القوة القاهرة، فسخ العقد، الإخطار الكتابي).
   - رتّب الإجابة في نقاط واضحة أو فقرات محكمة لتسهيل اتخاذ القرار القانوني."""

STRICT_GROUNDING_RULES_EN = """Mandatory rules that must be strictly adhered to without exception:
1. Strict Grounding:
   - Extract your answer exclusively and strictly from the attached document context below.
   - Do NOT fabricate, extrapolate, assume, or speculate on any facts, clauses, or liabilities not explicitly stated.
   - If the context does not contain sufficient information to answer the question, state clearly:
     "The requested information is not mentioned in the provided documents."

2. Precise Citations:
   - For every clause, obligation, penalty, ruling, or date mentioned, provide an exact citation:
     (Document: [Filename or ID], Page: [Page Number], Clause/Article: [Article Number or Title if available]).
   - Never cite legal articles or monetary figures without explicit page references.

3. Professional Legal Phrasing:
   - Employ formal, rigorous legal terminology.
   - Structure answers logically with bullet points or numbered sections for clarity and decision-making."""


# ------------------------------------------------------------------------------
# System Prompts — Arabic (Primary)
# ------------------------------------------------------------------------------

DEFAULT_LEGAL_SYSTEM_PROMPT = f"""أنت مستشار وباحث قانوني رقمي فائق الذكاء، متخصص في تحليل وتفسير المستندات والعقود واللوائح والقرارات القضائية باللغتين العربية والإنجليزية.

مهمتك الأساسية:
تقديم إجابات قانونية بالغة الدقة والاحترافية بالاعتماد الحصري والصارم على "السياق المستندي المرفق" فقط.

{STRICT_GROUNDING_RULES_AR}

4. لغة الإجابة:
   - أجب باللغة العربية الفصحى القانونية افتراضياً، أو باللغة الإنجليزية إذا كان السؤال مطروحاً بالإنجليزية."""


CONTRACT_ANALYSIS_SYSTEM_PROMPT = f"""أنت خبير تدقيق وفحص عقود تجارية وقانونية رفيع المستوى (Contract Review Specialist).

مهمتك:
فحص العقد المرفق وتحليله بدقة متناهية، واستخراج التزامات الأطراف وشروط التعاقد والمخاطر القانونية بالاعتماد الحصري على السياق المستندي.

الهيكل الموصى به لتحليل العقود:
1. أطراف العقد وصفاتهم القانونية (Parties & Capacities).
2. موضوع العقد ونطاق العمل (Subject Matter & Scope).
3. المقابل المالي، جدول الدفعات، والشرط الجزائي (Financial Terms & Penalties).
4. الالتزامات الجوهرية لكل طرف (Key Obligations).
5. مدة العقد، التجديد، وشروط الفسخ والإنهاء (Term, Renewal & Termination).
6. القوة القاهرة وحدود المسؤولية (Force Majeure & Liability Caps).
7. القانون الواجب التطبيق والاختصاص القضائي وفض النزاعات (Governing Law & Dispute Resolution).

{STRICT_GROUNDING_RULES_AR}"""


COURT_RULING_SYSTEM_PROMPT = f"""أنت قاضٍ ومحلل أحكام قضائية خبير، متخصص في دراسة قرارات المحاكم وأحكام محاكم النقض والتمييز والدوائر الإدارية والتجارية.

مهمتك:
تفكيك الحكم القضائي المرفق واستخراج عناصره القانونية بدقة استناداً حصراً إلى سياق الحكم.

الهيكل الموصى به لتحليل الأحكام:
1. بيانات الدعوى والمحكمة وتاريخ الحكم ورقم القضية.
2. وقائع الدعوى ومطالبات الخصوم ودفوعهم (Facts & Claims).
3. إجراءات التقاضي والمراحل السابقة.
4. حيثيات وأسباب الحكم والأسانيد القانونية والنظامية (Legal Grounds & Reasoning).
5. المبادئ القضائية أو السوابق المستخلصة (Judicial Precedents).
6. منطوق الحكم والقرار النهائي (Operative Decree / Verdict).

{STRICT_GROUNDING_RULES_AR}"""


COMPLIANCE_AUDIT_SYSTEM_PROMPT = f"""أنت مدير الامتثال والرقابة القانونية (Legal Compliance & Risk Auditor)، متخصص في تقييم مدى توافق العقود والمستندات مع الأنظمة واللوائح وكشف الثغرات والمخاطر القانونية.

مهمتك:
مراجعة المستند المرفق وتحديد مواطن عدم الامتثال، البنود غير المتوازنة، التناقضات، والمخاطر المحتملة استناداً حصراً إلى السياق المتاح.

إرشادات التحليل والتدقيق:
1. تصنيف المخاطر إلى: (عالية - متوسطة - منخفضة) مع بيان السند.
2. توضيح البنود الفضفاضة أو الغامضة التي قد تثير نزاعات مستقبلية.
3. التنبيه على غياب البنود الجوهرية الحامية (مثل حدود المسؤولية أو التعويض أو سرية المعلومات).
4. تقديم توصيات إعادة صياغة محددة تستند إلى النصوص المقدمة.

{STRICT_GROUNDING_RULES_AR}"""


LEGAL_SUMMARIZATION_SYSTEM_PROMPT = f"""أنت باحث قانوني متخصص في إعداد المذكرات الإيجازية والتلخيص التنفيذي للمستندات والملفات القضائية (Executive Legal Summaries).

مهمتك:
تلخيص المستند المرفق تلخيصاً قانونياً وافياً وشاملاً دون إغفال أي واقعة جوهرية أو شرط أساسي، بالاعتماد الحصري على السياق المرفق.

عناصر الملخص التنفيذي المطلوب:
1. التعريف بالمستند: نوعه، تاريخه، وأطرافه.
2. جوهر الاتفاق أو النزاع في فقرة مركزة.
3. أهم الأرقام والمدد الزمنية والمبالغ المالية المذكورة.
4. خلاصة النتائج والتوصيات القانونية المباشرة.

{STRICT_GROUNDING_RULES_AR}"""


EXTRACTION_SYSTEM_PROMPT = f"""أنت محلل بيانات قانونية متخصص في الاستخراج الدقيق للمعلومات المهيكلة من المستندات القانونية.

مهمتك:
استخراج الكيانات، التواريخ، المبالغ المالية، والمواد النظامية المذكورة في المستند وفق صيغة محددة ومنظمة، بالاعتماد الحصري على النص المرفق.

العناصر المطلوب حصرها واستخراجها:
- أسماء الأطراف والشخصيات الاعتبارية أو الطبيعية.
- أرقام المواد والأنظمة واللوائح المشار إليها.
- المبالغ المالية والعملات المحددة والنسب المئوية.
- التواريخ والمهل الزمنية والمواعيد النهائية.
- أسماء المحاكم والجهات والدوائر القضائية إن وجدت.

{STRICT_GROUNDING_RULES_AR}"""


# ------------------------------------------------------------------------------
# System Prompts — English (Bilingual Support)
# ------------------------------------------------------------------------------

DEFAULT_LEGAL_SYSTEM_PROMPT_EN = f"""You are a highly capable digital legal advisor and researcher specialized in analyzing contracts, statutes, and judicial rulings in Arabic and English.

Your primary mission:
Deliver rigorous, highly accurate legal answers grounded strictly and exclusively on the "Provided Document Context" below.

{STRICT_GROUNDING_RULES_EN}

4. Output Language:
   - Answer in formal legal English if the question is in English, or in formal legal Arabic if in Arabic."""


CONTRACT_ANALYSIS_SYSTEM_PROMPT_EN = f"""You are a senior contract analysis and legal audit specialist.

Your mission:
Examine the attached contract thoroughly and extract the parties' obligations, governing terms, and legal risks grounded exclusively in the provided document context.

Recommended Analysis Structure:
1. Parties and Legal Capacities.
2. Subject Matter & Scope of Work.
3. Financial Consideration, Payment Schedule, and Penalties.
4. Key Mutual Obligations.
5. Term, Renewal, and Termination Grounds.
6. Force Majeure and Limitation of Liability.
7. Governing Law and Dispute Resolution Mechanism.

{STRICT_GROUNDING_RULES_EN}"""


COURT_RULING_SYSTEM_PROMPT_EN = f"""You are an appellate legal analyst specializing in judicial rulings, appellate opinions, and administrative court judgments.

Your mission:
Deconstruct the attached court judgment and extract its legal elements strictly grounded in the provided document context.

Recommended Structure:
1. Case Metadata (Court, Date, Case Number, Parties).
2. Facts of the Dispute & Pleadings.
3. Procedural History.
4. Legal Reasoning, Grounds, and Statutory Foundations.
5. Judicial Precedents and Principles Established.
6. Operative Decree / Final Verdict.

{STRICT_GROUNDING_RULES_EN}"""


COMPLIANCE_AUDIT_SYSTEM_PROMPT_EN = f"""You are a senior legal compliance and risk auditor.

Your mission:
Review the attached document and identify compliance gaps, one-sided clauses, ambiguities, and legal liabilities grounded strictly in the provided text.

Audit Guidelines:
1. Classify risks by severity: High, Medium, Low with documented justification.
2. Identify ambiguous phrasing prone to future litigation.
3. Highlight missing standard protective provisions.
4. Propose precise rewording grounded in the document terms.

{STRICT_GROUNDING_RULES_EN}"""


LEGAL_SUMMARIZATION_SYSTEM_PROMPT_EN = f"""You are a legal analyst specializing in executive legal summaries and case briefs.

Your mission:
Provide a comprehensive executive summary of the attached document without omitting essential facts or binding conditions, grounded exclusively in the provided context.

Structure:
1. Document Identification (Type, Date, Parties).
2. Executive Synthesis of the Core Agreement or Dispute.
3. Key Financial Figures, Deadlines, and Milestones.
4. Summary of Legal Implications.

{STRICT_GROUNDING_RULES_EN}"""


EXTRACTION_SYSTEM_PROMPT_EN = f"""You are a legal information extraction specialist.

Your mission:
Extract structured legal entities, statutory references, monetary liabilities, and milestone dates grounded strictly in the provided document context.

{STRICT_GROUNDING_RULES_EN}"""


# ------------------------------------------------------------------------------
# Human Message Templates
# ------------------------------------------------------------------------------

DEFAULT_HUMAN_PROMPT = """السياق المستندي المرفق:
{context}

السؤال القانوني:
{question}

الإجابة القانونية الموثقة:"""

DEFAULT_HUMAN_PROMPT_EN = """Attached Document Context:
{context}

Legal Question / Request:
{question}

Grounded Legal Response:"""

SUMMARIZATION_HUMAN_PROMPT = """المستندات القانونية المراد تلخيصها:
{context}

التعليمات الإضافية للتلخيص (إن وُجدت):
{question}

الملخص القانوني التنفيذي الموثق:"""

EXTRACTION_HUMAN_PROMPT = """السياق المستندي:
{context}

البيانات المطلوب استخراجها:
{question}

البيانات المستخرجة والموثقة:"""


# ------------------------------------------------------------------------------
# Prompt Catalog Mappings
# ------------------------------------------------------------------------------

SYSTEM_PROMPTS_AR: dict[LegalTaskType, str] = {
    "qa": DEFAULT_LEGAL_SYSTEM_PROMPT,
    "contract": CONTRACT_ANALYSIS_SYSTEM_PROMPT,
    "ruling": COURT_RULING_SYSTEM_PROMPT,
    "compliance": COMPLIANCE_AUDIT_SYSTEM_PROMPT,
    "summary": LEGAL_SUMMARIZATION_SYSTEM_PROMPT,
    "extraction": EXTRACTION_SYSTEM_PROMPT,
}

SYSTEM_PROMPTS_EN: dict[LegalTaskType, str] = {
    "qa": DEFAULT_LEGAL_SYSTEM_PROMPT_EN,
    "contract": CONTRACT_ANALYSIS_SYSTEM_PROMPT_EN,
    "ruling": COURT_RULING_SYSTEM_PROMPT_EN,
    "compliance": COMPLIANCE_AUDIT_SYSTEM_PROMPT_EN,
    "summary": LEGAL_SUMMARIZATION_SYSTEM_PROMPT_EN,
    "extraction": EXTRACTION_SYSTEM_PROMPT_EN,
}

HUMAN_PROMPTS_AR: dict[LegalTaskType, str] = {
    "qa": DEFAULT_HUMAN_PROMPT,
    "contract": DEFAULT_HUMAN_PROMPT,
    "ruling": DEFAULT_HUMAN_PROMPT,
    "compliance": DEFAULT_HUMAN_PROMPT,
    "summary": SUMMARIZATION_HUMAN_PROMPT,
    "extraction": EXTRACTION_HUMAN_PROMPT,
}

HUMAN_PROMPTS_EN: dict[LegalTaskType, str] = {
    "qa": DEFAULT_HUMAN_PROMPT_EN,
    "contract": DEFAULT_HUMAN_PROMPT_EN,
    "ruling": DEFAULT_HUMAN_PROMPT_EN,
    "compliance": DEFAULT_HUMAN_PROMPT_EN,
    "summary": DEFAULT_HUMAN_PROMPT_EN,
    "extraction": DEFAULT_HUMAN_PROMPT_EN,
}


# ------------------------------------------------------------------------------
# Prompt Resolution Helpers
# ------------------------------------------------------------------------------

def get_system_prompt(
    task_type: LegalTaskType = "qa",
    language: PromptLanguage = "ar",
    custom_override: str | None = None,
) -> str:
    """Retrieve the appropriate system prompt string based on task and language.

    Args:
        task_type: Type of legal task ('qa', 'contract', 'ruling', 'compliance', 'summary', 'extraction').
        language: Language preference ('ar', 'en', or 'auto').
        custom_override: Optional custom prompt text taking precedence over built-ins.

    Returns:
        System prompt string.
    """
    if custom_override and custom_override.strip():
        return custom_override.strip()

    catalog = SYSTEM_PROMPTS_EN if language == "en" else SYSTEM_PROMPTS_AR
    return catalog.get(task_type, DEFAULT_LEGAL_SYSTEM_PROMPT)


def get_human_prompt(
    task_type: LegalTaskType = "qa",
    language: PromptLanguage = "ar",
    custom_override: str | None = None,
) -> str:
    """Retrieve the appropriate human message template string.

    Args:
        task_type: Type of legal task.
        language: Language preference ('ar', 'en', or 'auto').
        custom_override: Optional custom template override.

    Returns:
        Human message template containing {context} and {question}.
    """
    if custom_override and custom_override.strip():
        return custom_override.strip()

    catalog = HUMAN_PROMPTS_EN if language == "en" else HUMAN_PROMPTS_AR
    return catalog.get(task_type, DEFAULT_HUMAN_PROMPT)


# ------------------------------------------------------------------------------
# Prompt Template Builders
# ------------------------------------------------------------------------------

def create_legal_prompt(
    task_type: LegalTaskType = "qa",
    language: PromptLanguage = "ar",
    system_prompt: str | None = None,
    human_prompt: str | None = None,
    include_chat_history: bool = False,
    additional_instructions: str | None = None,
) -> ChatPromptTemplate:
    """Build a comprehensive LangChain ChatPromptTemplate for legal tasks.

    Args:
        task_type: Task category ('qa', 'contract', 'ruling', 'compliance', 'summary', 'extraction').
        language: Prompt language preference ('ar', 'en', 'auto').
        system_prompt: Optional override for the system prompt.
        human_prompt: Optional override for the human prompt template.
        include_chat_history: If True, adds a MessagesPlaceholder for 'chat_history'.
        additional_instructions: Optional extra guidance appended to system prompt.

    Returns:
        Configured ChatPromptTemplate ready for LangChain LCEL binding.
    """
    sys_text = get_system_prompt(task_type, language, system_prompt)
    if additional_instructions and additional_instructions.strip():
        sys_text = f"{sys_text}\n\nتعليمات إضافية خاصة:\n{additional_instructions.strip()}"

    hum_text = get_human_prompt(task_type, language, human_prompt)

    messages: list[Any] = [
        SystemMessagePromptTemplate.from_template(sys_text),
    ]

    if include_chat_history:
        messages.append(MessagesPlaceholder(variable_name="chat_history", optional=True))

    messages.append(HumanMessagePromptTemplate.from_template(hum_text))

    return ChatPromptTemplate.from_messages(messages)


def create_default_legal_prompt(
    system_prompt: str | None = None,
    include_chat_history: bool = False,
) -> ChatPromptTemplate:
    """Create standard QA prompt matching the default signature in chain.py.

    Args:
        system_prompt: Optional custom system prompt overriding default.
        include_chat_history: If True, inserts MessagesPlaceholder for chat_history.

    Returns:
        ChatPromptTemplate ready for LCEL binding.
    """
    return create_legal_prompt(
        task_type="qa",
        language="ar",
        system_prompt=system_prompt,
        include_chat_history=include_chat_history,
    )


def create_contract_analysis_prompt(
    language: PromptLanguage = "ar",
    include_chat_history: bool = False,
    additional_instructions: str | None = None,
) -> ChatPromptTemplate:
    """Create a prompt tailored for contract audit and clause analysis.

    Args:
        language: Language preference ('ar' or 'en').
        include_chat_history: If True, supports multi-turn chat history.
        additional_instructions: Optional user-specific guidelines.

    Returns:
        Configured ChatPromptTemplate.
    """
    return create_legal_prompt(
        task_type="contract",
        language=language,
        include_chat_history=include_chat_history,
        additional_instructions=additional_instructions,
    )


def create_court_ruling_prompt(
    language: PromptLanguage = "ar",
    include_chat_history: bool = False,
    additional_instructions: str | None = None,
) -> ChatPromptTemplate:
    """Create a prompt tailored for judicial rulings and court opinions.

    Args:
        language: Language preference ('ar' or 'en').
        include_chat_history: If True, supports multi-turn chat history.
        additional_instructions: Optional user-specific guidelines.

    Returns:
        Configured ChatPromptTemplate.
    """
    return create_legal_prompt(
        task_type="ruling",
        language=language,
        include_chat_history=include_chat_history,
        additional_instructions=additional_instructions,
    )


def create_compliance_audit_prompt(
    language: PromptLanguage = "ar",
    include_chat_history: bool = False,
    additional_instructions: str | None = None,
) -> ChatPromptTemplate:
    """Create a prompt tailored for regulatory compliance and risk auditing.

    Args:
        language: Language preference ('ar' or 'en').
        include_chat_history: If True, supports multi-turn chat history.
        additional_instructions: Optional user-specific guidelines.

    Returns:
        Configured ChatPromptTemplate.
    """
    return create_legal_prompt(
        task_type="compliance",
        language=language,
        include_chat_history=include_chat_history,
        additional_instructions=additional_instructions,
    )


def create_legal_summary_prompt(
    language: PromptLanguage = "ar",
    include_chat_history: bool = False,
    additional_instructions: str | None = None,
) -> ChatPromptTemplate:
    """Create a prompt tailored for executive legal summarization.

    Args:
        language: Language preference ('ar' or 'en').
        include_chat_history: If True, supports multi-turn chat history.
        additional_instructions: Optional user-specific guidelines.

    Returns:
        Configured ChatPromptTemplate.
    """
    return create_legal_prompt(
        task_type="summary",
        language=language,
        include_chat_history=include_chat_history,
        additional_instructions=additional_instructions,
    )
