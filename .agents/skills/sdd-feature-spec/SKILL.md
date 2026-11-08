---
name: sdd-feature-spec
description: FEATURE MODE planning in Spec-Driven Development for any project, for a single feature. Phase mode (sdd-phase) is the default; use this skill when the user explicitly wants one feature ("spec feature X.Y", "small feature", "one feature at a time"), for sensitive areas listed in spec/sdd-profile.md, or for a small ad-hoc idea. sdd-phase also reuses its Step 6 quality rules. Runs the clean-slate check, estimates size, interviews the human with 3-6 questions (each with a recommended answer), writes plan.md, requirements.md and validation.md, then stops at Gate 1. Never writes application code.
---

# sdd-feature-spec — Plan a Feature (ends at Gate 1)

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE). Templates are in `sdd-conventions/templates/`.

> The agent is the muscle; the spec is the brain. This skill produces the brain and never writes application code.

## Inputs
- **`next`** (default): the first unchecked `- [ ] Feature X.Y` in `spec/roadmap.md`.
- **`new`**: the human describes something that isn't on the roadmap.

## Step 1 — Clean Slate
CONV §7. Dirty tree or wrong branch → report and ask. Never stash, discard, or switch on your own. No constitution/profile → run `sdd-constitution` first.

## Step 2 — Load Context From Disk
`spec/mission.md`, `spec/tech.md` (invariants), `spec/roadmap.md`, `spec/sdd-profile.md`, relevant `spec/architecture.md` / `spec/adr/` / `spec/security/`, the 1–2 most recent feature folders (to match their style), the source files involved, and related `spec/backlog/` items. Never `spec/archive/`.

## Step 3 — Prerequisites
Prerequisite roadmap items are checked **and** their code actually exists. Otherwise stop and report.

## Step 4 — Estimate Size (agent decides; human may override)
| Size | Signals | Action |
|:---|:---|:---|
| Small | 1–2 execution groups, one module, no new dependency or invariant | One folder. `new` mode: add as the next item in the best-fitting phase |
| Medium | 3–5 groups, a few modules | One folder, implement group by group |
| Big | >5 groups, independent deliverables, a new subsystem, or a constitution change | Propose a breakdown into a phase/features → `sdd-replan` mode B → after approval, spec the first feature (or run `sdd-phase`) |

## Step 5 — Spec Interview (always, 3–6 questions)
Each question has a recommended answer first. Cover what is genuinely open: scope boundaries, key technical decisions and trade-offs, conflicts with the constitution or code, the validation approach, and the size (if unclear). Anything already decided → record it in `plan.md` Decisions with its source. On real ambiguity, ask; never guess.

## Step 6 — Write the Three Files
Folder: CONV §3 / PROFILE "Feature Folder Pattern".
1. `plan.md` ← `templates/plan.md`
2. `requirements.md` ← `templates/requirements.md`
3. `validation.md` ← `sdd-validation` spec mode + `templates/validation.md`

Quality bar: every R/NFR ↔ at least one validation check; the constraints list the relevant `tech.md` invariants; execution groups are small and testable; right level of detail (CONV §8). `new` mode Small/Medium → add the roadmap line. Optional: a summary in the conversation language.

## Step 7 — Self-Check
- [ ] No application code touched.
- [ ] The three files are consistent with each other.
- [ ] Open questions are empty or explicitly listed.
- [ ] Roadmap edits are limited to this feature's line.

## Step 8 — 🛑 GATE 1
Present (conversation language): the size and why, the decisions table, assumptions and risks, file links. Changes → update all affected files consistently, then present again. Only an approval word unlocks `sdd-implement`. Recommend committing the spec (`docs(spec): add Feature X.Y spec`) and implementing in a fresh context.

