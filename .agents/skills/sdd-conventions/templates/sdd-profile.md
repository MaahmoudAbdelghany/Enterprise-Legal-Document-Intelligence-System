# SDD Project Profile — <Project Name>

> Project-specific settings read by every global `sdd-*` skill. Part of the constitution: change it via `sdd-replan`.
> Values marked `(discovered)` were inferred from the repo; values marked `(confirmed)` were confirmed by the human.

## Commands
| Purpose | Command |
|:---|:---|
| Install / sync deps | `<e.g. npm ci · uv sync --all-extras --dev · poetry install>` |
| Full test suite | `<e.g. npm test · uv run pytest -v>` |
| Targeted tests | `<e.g. npx vitest run <file> · uv run pytest <file> -v>` |
| Lint | `<e.g. npm run lint · uv run ruff check .>` |
| Format check | `<command or "not configured">` |
| Type check | `<e.g. npx tsc --noEmit · mypy · "not configured">` |
| Build | `<e.g. npm run build · "not applicable">` |
| Run app (manual checks) | `<e.g. npm run dev → http://localhost:3000>` |
| Extra gates (evals, e2e, security scans) | `<command + when it applies>` |

## Main Branch & Remote
- Main branch: `<main | master>`
- Remote: `<origin URL or "none">`
- Merge style: `<PR preferred | merge --no-ff>`

## Execution Mode
- Default: `phase` | `feature`

## Sensitive Areas (force feature mode + mandatory deep review)
- `<e.g. auth / payments / DB schema & migrations / PII / booking writes / state machine>`
- Paths: `<e.g. src/auth/**, prisma/schema.prisma>`

## Deep Review Lenses
Four parallel read-only subagents (adapt to the project):
1. `<Security & privacy — e.g. OWASP, secrets, PII handling>`
2. `<Architecture & tech.md invariants>`
3. `Spec drift (requirements ↔ validation ↔ code ↔ roadmap ↔ docs)`
4. `Test quality (fabricated/over-mocked tests, missing edge cases)`

## Key Invariants to Check in Review
- `<e.g. all config via src/config.ts · p95 latency < 300ms · no direct DB access from UI>` (mirror of the most critical `spec/tech.md` rules)

## Feature Folder Pattern
- `spec/features/feature-<phase><nn>-<slug>/` `<or custom>`

## Language
- Artifact language (specs, code, commits, changelog): `English`
- Conversation language (messages to the human): `<e.g. Egyptian Arabic, right-to-left>`
- Approval words: `approved, go ahead, proceed, <local equivalents>`

## Changelog
- File: `CHANGELOG.md` · Audience: `<stakeholders>` · Title line: `<product name>`

