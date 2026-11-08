---
name: sdd-validation
description: Defines and executes Spec-Driven Development validation for any project. SPEC mode writes a feature's validation.md (called by sdd-feature-spec / sdd-phase); EXECUTE mode runs every check and records evidence (called by sdd-implement). Covers unit, integration, error/edge-case, regression, build/lint/type, manual/visual, and API/CLI tiers using only the project's real tools (from spec/sdd-profile.md), forbids fabricated tests, and requires a technical justification for every Not Applicable tier.
---

# sdd-validation — Prove Success With Real Evidence

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE).

> Validation is broader than testing. Tests are one form of evidence; validation defines what counts as complete proof that each requirement was met.

`Requirement → What does success mean? → What evidence proves it? → Tests / Manual / API / CLI / Build / Lint / Regression`

## Mode A — SPEC (write `validation.md`)

### A1. Real capabilities only
Use PROFILE "Commands", and verify them against the repo (manifests, lockfiles, CI workflows, test folders, scripts). Never require a tool that isn't installed. Adding one is a tech-stack change → `sdd-replan`.

### A2. Tiers (only when applicable)
| Tier | Use when | Focus |
|:---|:---|:---|
| 1 Unit | Isolated logic (parsers, schemas, pure functions, rules) | Known input → expected output, no I/O |
| 2 Integration | Components collaborate (route → service → storage, UI → API) | Data flow and state across boundaries |
| 3 Error & Edge | Always consider | Abnormal input, outages, timeouts, limits, graceful fallbacks |
| 4 Regression | **Mandatory** when existing files, schemas, tables, or core services change | Full suite green |
| 5 Build/Lint/Type | Always, using whatever is configured | PROFILE lint / type / build |
| 6 Manual/Visual | UI, UX, audio, responsive layouts, copy/i18n | Exact launch command, steps, expected result |
| 7 API/CLI/App | Exposed entry points | Real command or request → live output |
| Extra gates | PROFILE "Extra gates" conditions are met | e.g. evals, e2e, security scans |

### A3. Write the file
Use `templates/validation.md`. Include a **traceability table** (every R/NFR → check IDs, and back). Make checks concrete: test path/name, input, expected output, or exact command plus expected output. Performance/cost NFRs from `tech.md` get measurable checks when the feature touches those paths.

## Mode B — EXECUTE
1. Run the targeted tests, then the full suite, lint, type check, and build (as configured), then any extra gates that apply.
2. Run non-interactive Tier 7 commands. Ask first before anything that contacts real users, sends real messages, or spends significant paid budget.
3. Tier 6: prepare exact instructions for the human. Never claim manual checks as passed.
4. Fill in §5 Execution Results (✅ / ❌ / ⏭ N/A plus evidence) and tick only checks that genuinely passed.
5. ❌ → fix the code (or the spec, under the no-drift rule) and re-run.

## Strict Rules ⚖️
1. **Never fabricate a tier**: no hollow, contrived, or over-mocked tests just to fill a checklist.
2. **Not Applicable needs a technical justification** (e.g. "isolated regex utility: no inter-component calls, no I/O, no state"). "Not needed" is forbidden.
3. **Only real, discovered commands**: never invent scripts. If a command doesn't exist, fix the spec.
4. **Evidence or it didn't happen**: never report a check as passed unless you ran it in this session.

