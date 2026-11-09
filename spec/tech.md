# Tech — LexisGraph

> Living document. Changes go through `sdd-replan` (mode C). Versions are the ones locked in `uv.lock` (verified 2026-11-09).

## Stack
| Layer | Technology | Version | Role |
|:---|:---|:---|:---|
| Language | Python | 3.11 (runtime) · `>=3.10` (declared) | Everything |
| Package manager | uv | — | Dependency locking (`uv.lock`) and running tools |
| API | FastAPI + Uvicorn | 0.141.1 | REST + SSE streaming under `/api/v1` |
| Validation / settings | Pydantic v2 + pydantic-settings | 2.13.5 | Domain schemas, structured LLM output, centralized config |
| LLM orchestration | LangChain (LCEL) | core 1.6.1 | RAG chain, prompts, streaming |
| LLM providers | Groq (primary), OpenAI (fallback) | groq 0.37.1 | Generation + vision text extraction; later: local open models (Ollama / vLLM) |
| Vector store | Qdrant | client 1.19.0 · server v1.12.1 | Multi-vector (ColPali) today; text vectors + hybrid from Phase 3 |
| Visual retrieval | ColPali v1.3 (transformers / torch) | transformers 5.16.1 · torch 2.14.0 | Late-interaction page embeddings — **complex pages only** from Phase 3 |
| Keyword search | rank-bm25 | — | BM25 for hybrid retrieval (Phase 3) |
| PDF | pdf2image, pypdf, Pillow | — | Page rendering and parsing |
| PII | Presidio (analyzer/anonymizer) | — | Arabic PII masking (Phase 4) |
| Evaluation | RAGAS + custom field-accuracy runner | — | Quality, citation, cost, latency (Phase 2–3) |
| UI | Streamlit | to be added | Arabic RTL pilot UI (Phase 2) |
| Infra | Docker Compose | — | Local Qdrant (+ Neo4j, deferred) |
| Logging | loguru | — | Structured logs incl. cost/latency |
| Testing | pytest, pytest-asyncio, pytest-cov | — | Unit + API tests |
| Quality tools | ruff, black, mypy | — | Lint, format, types |

**Deferred (installed but not to be extended):** Neo4j / `langchain-neo4j`, NeMo Guardrails, AWS ECS.

## Invariants (testable)
1. **Single config source** — all settings are read through `app/config.py` (`Settings` / `get_settings`). No `os.environ` / `os.getenv` elsewhere in `app/`, and no API keys or secrets hard-coded anywhere.
2. **Cited or "not found"** — every Q&A answer and every extracted field carries its source (document ID + page number), or an explicit not-found value. No uncited factual output reaches the API or UI.
3. **Validated LLM output** — structured LLM output (e.g. extracted fields) is parsed into a Pydantic model; invalid output is retried or returned as a structured error, never passed through raw.
4. **Extract once** — vision/text extraction runs at most once per page at ingestion and its result is cached/stored; query and extraction requests never re-run vision extraction on already-ingested pages.
5. **Measured cost** — every LLM call records model, input/output tokens, estimated USD cost, and latency (from Feature 2.4 onward).
6. **Model choice is configuration, not code** — which model serves which task (cheap vs strong tier, provider) is set in configuration; changing it requires no code change (from Phase 3/4).
7. **GPU optional** — the system must ingest and answer with ColPali disabled on a CPU-only machine (from Phase 3).
8. **No raw PII to cloud** — when PII masking is enabled, no unmasked personal data is sent to an external LLM provider (from Phase 4).
9. **Offline tests** — the test suite never calls real external services (LLM APIs, Qdrant, Neo4j, Hugging Face downloads); boundaries are mocked or faked. `pytest` passes with no network and no API keys.
10. **Green suite** — every feature lands with tests for its requirements, and the full test suite passes.
11. **Clean new code** — files created or modified by a feature pass `ruff check` and `black --check` (existing lint/format/type debt in untouched files is tracked in the backlog and not a blocker).
12. **Arabic normalization is shared** — Arabic text normalization uses the single implementation in `app/core/ingestion/text_extractor.py`; no ad-hoc duplicates.

## Folder Conventions
```
app/
  config.py              # centralized settings (invariant 1)
  main.py                # FastAPI app factory + router registration
  api/routes/            # HTTP layer only (documents, query, health, …)
  api/dependencies.py    # DI for services
  models/schemas.py      # Pydantic API/domain schemas
  core/ingestion/        # PDF → images → text/embeddings
  core/retrieval/        # Qdrant, BM25, hybrid fusion
  core/generation/       # prompts, chains, model routing / providers
  core/extraction/       # clause extraction (Phase 2)
  core/security/         # PII masking, audit (Phase 4)
  core/knowledge_graph/  # deferred
ui/                      # Streamlit pilot UI (Phase 2)
evaluation/              # eval corpus, ground truth, runners
scripts/                 # CLI entry points (eval, maintenance)
tests/                   # pytest, mirrors app/ (test_<module>.py)
infrastructure/          # deployment assets (Phase 5)
spec/                    # SDD source of truth
```
- Routes stay thin: business logic lives in `app/core/`.
- Tests mirror modules: `tests/test_<module>.py`.

## Testing
- Framework: pytest (`asyncio_mode = auto`), coverage via pytest-cov (configured in `pyproject.toml`).
- Baseline (2026-11-09): **234 tests pass**, coverage 87%.
- External boundaries (LLM, Qdrant, model downloads) are mocked — see invariant 9.
- Evaluation (accuracy/cost/latency) is a separate gate run by the evaluation runner (Phase 2+), not part of the unit suite.

## Deployment
- **Today:** local development — `docker compose up -d qdrant` + Uvicorn.
- **Pilot (Phase 5):** Docker image + compose profile on a client cloud VM or on-premise server; cloud LLM with PII masking by default, local open model as an option.
- **Deferred:** AWS ECS / CloudFormation.
