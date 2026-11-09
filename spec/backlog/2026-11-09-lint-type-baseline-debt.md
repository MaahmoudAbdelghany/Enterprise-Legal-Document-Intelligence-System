# Lint, format, and type-check baseline debt

- **Date:** 2026-11-09
- **Type:** Tech debt
- **Source:** `sdd-constitution` discovery

## Context
Running the profile commands on `main` (commit `efc143f`):
- `ruff check app tests` → 167 errors (97 auto-fixable)
- `black --check app tests` → 25 files would be reformatted
- `mypy app` → 17 errors in 6 files (e.g. `app/api/routes/documents.py:189` arg-type)

The test suite is green (234 passed). Per `tech.md` invariant 11, only files touched by a feature must be clean, so this debt does not block features.

## Proposal
A dedicated cleanup feature (mechanical `ruff --fix` + `black`, then manual mypy fixes) scheduled before Phase 5 CI (Feature 5.3), so CI can enforce all three gates on the whole codebase.

## Status
Open — schedule via `sdd-replan`.
