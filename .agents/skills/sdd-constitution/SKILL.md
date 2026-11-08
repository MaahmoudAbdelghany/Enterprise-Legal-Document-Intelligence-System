---
name: sdd-constitution
description: Creates or regenerates a project's Spec-Driven Development constitution (spec/mission.md, spec/tech.md, spec/roadmap.md, spec/sdd-profile.md, optional spec/architecture.md). Greenfield mode interviews the human; brownfield mode reverse-engineers the existing codebase, README, TODOs, git history, and old plans. Use when the user says "create the constitution", "adopt SDD in this project", "set up SDD", or "regenerate the specs", or when spec/ or spec/sdd-profile.md is missing.
---

# sdd-constitution — Mission, Tech Stack, Roadmap, Profile

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV).

> The constitution captures agreement between the human and the agent, and among the humans. It is a living document; later changes go through `sdd-replan`.

## Modes
- **Greenfield** (little or no code): interview-driven.
- **Brownfield** (existing code): discover first, then interview only for gaps.
- **Refresh** (`spec/` exists but has drifted): discover, write a drift report, then apply it via `sdd-replan` mode C.

Work on `replan/constitution` if git is available. **Never refactor application code in this skill.**

## Step 1 — Discover (brownfield / refresh)
Separate **verified facts** (with file references) from **assumptions**:
- README, manifests and lockfiles, `.env.example` (variable names only), CI workflows, Dockerfiles.
- Source layout, entry points, tests, scripts, and lint/format/type configs → the commands for the profile.
- `git log --oneline -50`, TODO/FIXME comments, issue lists, old plan documents (input only; afterwards move them to `spec/archive/` with a README note, if the human agrees).

## Step 2 — Interview (3–8 questions, each with a recommended answer)
Only what the code can't tell you: mission tone, audience, problem, out-of-scope items; stack constraints (company standards, hosting, budget, compliance); non-negotiable invariants; sensitive areas; roadmap granularity (recommended: small phases with small features); default execution mode (phase recommended); conversation language and approval words.

## Step 3 — Write
| File | Content |
|:---|:---|
| `spec/mission.md` | Product (what), Problem (why), Target users (who), Core goal, Scope in/out, success metrics, tone |
| `spec/tech.md` | Stack table (layer · tech · version · role), numbered **testable invariants**, folder conventions, testing stack, deployment |
| `spec/roadmap.md` | Living-document header, phases with goals, `- [ ] **Feature X.Y**: title (main paths)` in small steps; brownfield: built work is `- [x]` |
| `spec/sdd-profile.md` | From `sdd-conventions/templates/sdd-profile.md`: real commands, main branch and remote, mode, sensitive areas, review lenses, key invariants, folder pattern, language, changelog |
| `spec/architecture.md` | Optional: component diagram (Mermaid), data flows, ADR links |

Also create `spec/backlog/README.md` (an empty index table). Don't add extra documents unless asked.

## Step 4 — Self-Check
- [ ] The stack and commands match the repo (or the human's choices); every profile command was run once, or marked unverified.
- [ ] Invariants are concrete and verifiable.
- [ ] The next roadmap item is small, with clear prerequisites.
- [ ] Facts vs assumptions are marked; assumptions are listed for the human.

## Step 5 — 🛑 Human Review
Present a summary, assumptions, and open questions (conversation language). Apply corrections through the agent, keeping all the files consistent. After approval: commit `docs(spec): establish project constitution`, and ask before merging. Next: `sdd-phase` (default) or `sdd-feature-spec`.

