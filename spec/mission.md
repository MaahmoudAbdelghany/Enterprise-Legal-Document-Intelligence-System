# Mission — LexisGraph

> Living document. Changes go through `sdd-replan` (mode C). Source inputs: grill-me decisions and the FDE business strategy (2026-11-09).

## Product (What)
LexisGraph is an **Arabic-first contract intelligence assistant**. It ingests Arabic (and bilingual Arabic/English) contracts — including scanned PDFs with tables, stamps, and signatures — and produces:
1. A **structured table of key clauses** extracted from each contract, every value linked to its source page.
2. **Question answering** over the contract archive, where every answer cites its source page or explicitly says the information was not found.

## Problem (Why)
Legal and procurement teams in Arab companies manage large volumes of Arabic contracts, many of them scanned. Reviewing a contract manually takes hours, renewal and penalty deadlines are missed, and knowledge is locked in individual employees' heads. Global legal-AI tools are English-first, weak on scanned Arabic documents, and send data abroad — which conflicts with data-protection laws and client sensitivity in Egypt, Saudi Arabia, and the UAE.

## Target Users (Who)
- **Primary (first segment):** in-house **legal departments and procurement teams** in medium and large companies (e.g. real estate, contracting), **starting in Egypt**.
- **Later:** law firms and in-house legal teams in Saudi Arabia and the UAE; banks/insurance (customer-service policy bot).
- **End users are non-technical** (lawyers, procurement officers). The buyer is the department head / CxO who needs proven ROI.

## Core Goal
Deliver a **sellable demo and paid pilot**: a lawyer uploads a contract and, within two minutes, gets an accurate clause table plus cited answers — at a cost of a few cents per contract.

## Guiding Principles
1. **Lowest cost × highest performance** — every technical and product decision is judged by: "is this the cheapest way to reach the required quality?"
2. **Pareto (80/20)** for prioritisation — build the 20% that delivers 80% of customer value first.
3. **Lean** for execution — ship the smallest thing that teaches us something, measure, then improve. Optimisation is driven by evaluation numbers, never by guesswork.
4. **Trust over cleverness** — cite sources, say "not found" instead of hallucinating, keep a human in the loop.
5. **Data sovereignty by design** — cloud first with PII masking, swappable to on-premise models.

## Scope
### In scope (current horizon)
- Contract ingestion (digital and scanned PDFs, Arabic and bilingual).
- Extraction of **8 key fields**: parties & contract type · start & end dates · auto-renewal & notice period · value & payment terms · penalties & liquidated damages · guarantees & letters of guarantee · termination conditions · governing law & dispute resolution.
- Cited question answering over uploaded contracts.
- Excel export of the clause table.
- Streamlit pilot UI (Arabic, right-to-left).
- Text-first hybrid retrieval (embeddings + BM25), ColPali only for complex pages.
- Two-tier model routing (cheap model / strong model), selected by Arabic evaluation results.
- Arabic PII masking before any cloud LLM call; swappable model provider including local open models.
- Evaluation harness measuring accuracy, citation correctness, cost, and latency.

### Out of scope (deferred — see roadmap "Deferred")
- Neo4j knowledge graph.
- Full AWS ECS production deployment.
- Comparing contracts against a company playbook / risk scoring.
- Confidentiality and non-compete field extraction.
- WhatsApp / customer-service channels, ERP integrations.
- Multi-tenant SaaS, billing, user management beyond the pilot.

## Success Metrics (demo / pilot acceptance)
| Metric | Target |
|:---|:---|
| Field extraction accuracy (on the evaluation set) | ≥ 85% |
| Citation correctness (cited page actually contains the answer/value) | ≥ 90% |
| Cost per contract (~20 pages, end to end) | < USD 0.10 |
| Processing time per contract (~20 pages) | < 2 minutes |
| Unknown information | System answers "not found" instead of inventing a value |

Targets are revisited after the first baseline measurement (via `sdd-replan`).

## Evaluation Data Policy
A mixed corpus: published public contract templates, synthetic variations generated from them, and 3–5 scanned contracts with all personal data removed. No real client data in the repository.

## Tone
Professional, precise, and trustworthy. Arabic-first in the user interface; outcomes over architecture in every demo.
