# SDD Project Profile — LexisGraph

> Project-specific settings read by every global `sdd-*` skill. Part of the constitution: change it via `sdd-replan`.
> Values marked `(discovered)` were inferred from the repo; values marked `(confirmed)` were confirmed by the human.

## Commands
| Purpose | Command |
|:---|:---|
| Install / sync deps | `uv sync --all-extras` (discovered — unverified in this run; the existing `.venv` was used) |
| Full test suite | `uv run pytest` (discovered — verified via `.venv/bin/python -m pytest`: 234 passed) |
| Targeted tests | `uv run pytest tests/test_<module>.py -o addopts="" -v` (discovered) |
| Lint | `uv run ruff check app tests` (discovered — verified: runs; **baseline 167 existing errors**) |
| Format check | `uv run black --check app tests` (discovered — verified: runs; **baseline 25 files need reformatting**) |
| Type check | `uv run mypy app` (discovered — verified: runs; **baseline 17 existing errors in 6 files**) |
| Build | `not applicable` until Feature 5.1 adds a `Dockerfile` |
| Run app (manual checks) | `docker compose up -d qdrant` then `uv run uvicorn app.main:app --reload` → http://localhost:8000/docs (discovered from README) |
| Extra gates (evals, e2e, security scans) | Evaluation runner (added in Feature 2.5) — required for any feature that changes extraction, retrieval, prompts, or model routing |

> Lint / format / type gates apply to **files created or modified by the feature** (`tech.md` invariant 11). The existing baseline debt is tracked in `spec/backlog/`.

## Main Branch & Remote
- Main branch: `main` (discovered)
- Remote: `origin` → https://github.com/MaahmoudAbdelghany/Enterprise-Legal-Document-Intelligence-System (discovered)
- Merge style: `merge --no-ff` (assumption — confirm)

## Execution Mode
- Default: `phase` (assumption — not objected)

## Sensitive Areas (force feature mode + mandatory deep review)
- PII masking & personal-data handling · secrets and API keys · LLM provider layer (decides where data is sent) (confirmed)
- Paths: `app/core/security/**`, `app/config.py` (secret fields), `app/core/generation/**` provider/model-routing code, `.env.example`
- Roadmap Phase 4 runs entirely in feature mode.

## Deep Review Lenses
Four parallel read-only subagents:
1. Security & privacy — OWASP Top 10 for LLMs (prompt injection, data leakage, insecure output handling), secrets, PII sent to providers
2. Architecture & `tech.md` invariants — single config source, cited-or-not-found, extract-once, model choice as configuration
3. Spec drift (requirements ↔ validation ↔ code ↔ roadmap ↔ docs)
4. Test quality & cost — fabricated/over-mocked tests, missing Arabic edge cases, unmeasured LLM cost/latency

## Key Invariants to Check in Review
- All config via `app/config.py`; no hard-coded secrets (invariant 1)
- Every answer/field cites document + page or says not found (invariant 2)
- LLM structured output validated by Pydantic (invariant 3)
- No vision re-extraction at query time (invariant 4)
- Every LLM call logs tokens, cost, latency (invariant 5)
- Tests run offline with no API keys (invariant 9)

## Feature Folder Pattern
- `spec/features/feature-<phase><nn>-<slug>/`

## Language
- Artifact language (specs, code, commits, changelog): `English` — no Arabic summaries (confirmed)
- Conversation language (messages to the human): `Egyptian Arabic, right-to-left` (follow the user's global formatting rules for Arabic questions)
- Approval words: `approved, go ahead, proceed, موافق, ابدأ, start` (assumption — not objected)

## Changelog
- File: `CHANGELOG.md` · Audience: `founder, pilot clients, and partners (non-technical)` · Title line: `LexisGraph`
