# Roadmap — LexisGraph

> **Living document.** Updated only through `sdd-replan` (structure) or a feature commit (checkbox tick). Phases are small; features are small steps with their main paths.
> Execution mode: **phase** by default; phases/features touching sensitive areas (see `spec/sdd-profile.md`) run in **feature mode**.

---

## Phase 1 — Foundation & Core Pipeline *(nearly complete)*
**Goal:** PDF → page images → embeddings → Qdrant → cited LLM answer, exposed via FastAPI.

- [x] **Feature 1.1**: Project scaffolding & centralized settings (`pyproject.toml`, `app/config.py`, `.env.example`)
- [x] **Feature 1.2**: Local infrastructure via Docker Compose (`docker-compose.yml`: Qdrant, Neo4j, app)
- [x] **Feature 1.3**: PDF processing to page images (`app/core/ingestion/pdf_processor.py`)
- [x] **Feature 1.4**: ColPali multi-vector embeddings + MaxSim (`app/core/ingestion/colpali_embedder.py`)
- [x] **Feature 1.5**: Vision-LLM text extraction + Arabic normalization (`app/core/ingestion/text_extractor.py`)
- [x] **Feature 1.6**: Qdrant multi-vector store CRUD (`app/core/retrieval/qdrant_store.py`)
- [x] **Feature 1.7**: Bilingual legal RAG chain with streaming (`app/core/generation/chain.py`, `prompts.py`)
- [x] **Feature 1.8**: Pydantic domain schemas (`app/models/schemas.py`)
- [x] **Feature 1.9**: Document, query (incl. SSE), and health API routes (`app/api/routes/`)
- [x] **Feature 1.10**: FastAPI entry point (`app/main.py`)
- [ ] **Feature 1.11**: End-to-end test — upload PDF → query → cited answer (`tests/`), external services mocked at the boundary

---

## Phase 2 — Sellable Demo
**Goal:** A lawyer uploads a contract and gets the 8-field clause table, cited answers, and an Excel export in a Streamlit UI — with first measurements of accuracy, cost, and latency.

- [ ] **Feature 2.1**: Evaluation corpus seed — ~10 Arabic contracts (public templates + synthetic variations + 3–5 anonymised scans) with ground truth for the 8 fields and ~30 Q&A pairs (`evaluation/`)
- [ ] **Feature 2.2**: Clause extraction service — 8-field Pydantic schema, structured LLM output, per-field source page, explicit "not found" (`app/core/extraction/`)
- [ ] **Feature 2.3**: Extraction API + Excel export endpoint (`app/api/routes/`)
- [ ] **Feature 2.4**: Cost & latency instrumentation — tokens, USD cost, and duration per document and per request (`app/core/`, logs)
- [ ] **Feature 2.5**: Evaluation runner — field accuracy, citation correctness, cost, latency report (`evaluation/`, `scripts/`)
- [ ] **Feature 2.6**: Streamlit pilot UI (Arabic RTL) — upload, clause table, cited Q&A, Excel download (`ui/`)

---

## Phase 3 — Quality & Cost Optimisation
**Goal:** Reach the success-metric targets at the lowest cost; make the system run without a GPU.

- [ ] **Feature 3.1**: Text-first indexing — cached page text + text embeddings in Qdrant (`app/core/retrieval/`)
- [ ] **Feature 3.2**: BM25 keyword search + Reciprocal Rank Fusion hybrid retriever (`app/core/retrieval/`)
- [ ] **Feature 3.3**: Page-complexity routing — ColPali only for complex pages (tables, stamps, poor scans) (`app/core/ingestion/`)
- [ ] **Feature 3.4**: Two-tier model routing — cheap model for extraction/classification, strong model for complex synthesis, configured per task (`app/config.py`, `app/core/generation/`)
- [ ] **Feature 3.5**: Model selection benchmark on the Arabic evaluation set; record the decision as an ADR (`evaluation/`, `spec/adr/`)

---

## Phase 4 — Security & Data Sovereignty *(sensitive — feature mode)*
**Goal:** No raw personal data leaves the client boundary; the same system runs on cloud or on-premise models.

- [ ] **Feature 4.1**: LLM provider abstraction — Groq / OpenAI / local, selectable by configuration (`app/core/generation/`)
- [ ] **Feature 4.2**: Arabic PII masking before every cloud LLM call (Presidio) (`app/core/security/`)
- [ ] **Feature 4.3**: Local open-model provider (Ollama / vLLM) for on-premise deployments (`app/core/generation/`)
- [ ] **Feature 4.4**: Secrets hardening and audit log of documents sent to external providers (`app/core/security/`)

---

## Phase 5 — Pilot Deployment
**Goal:** Install and operate the pilot at a client (cloud VM or on-premise) with a repeatable process.

- [ ] **Feature 5.1**: Multi-stage `Dockerfile` (the current compose file references one that does not exist yet)
- [ ] **Feature 5.2**: Pilot compose profile without Neo4j; optional local-model service
- [ ] **Feature 5.3**: CI with GitHub Actions — lint → test → build
- [ ] **Feature 5.4**: Pilot runbook & handover pack (install, operate, known issues)

---

## Deferred (not scheduled — revisit via `sdd-replan`)
- Neo4j knowledge graph (entity extraction, graph retrieval, `/graph` routes)
- Adaptive query router
- NeMo Guardrails
- Full AWS ECS / CloudFormation production stack
- Playbook comparison & risk flagging
- Confidentiality and non-compete fields
- WhatsApp / customer-service channel, ERP integrations
