---
name: sdd-review
description: Post-validation review gate in Spec-Driven Development for any project (feature mode). Use after sdd-validation execute mode, or when the user asks to "review the feature", "deep review", or "check before commit". Reads validation evidence, inspects the git diff against the 5 core questions, optionally runs a parallel deep review with subagents (lenses from spec/sdd-profile.md), writes review.md, and stops at Gate 2. After approval, commits, runs sdd-changelog, pushes, and stops at Gate 3. sdd-phase reuses its 5 questions and deep review at phase level.
---

# sdd-review — Review, Gate 2, Commit, Gate 3

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE).

## Step 1 — Read the Validation Evidence
`validation.md` §5 is filled in; all automated checks are ✅; every tier is either covered or N/A with a justification; extra gates ran where they apply. Anything red or missing → **HALT**, back to `sdd-implement`.

## Step 2 — Diff Review: the 5 Core Questions
`git status --short`, `git diff --stat`, `git diff`, `git diff --staged`.

| # | Question | Fails when… |
|:---|:---|:---|
| 1 | Does it work? | Stubs, TODO, placeholder returns, mock-only illusions, unrun paths |
| 2 | Does it match the spec? | Missing R/NFR, altered logic, **scope creep** |
| 3 | Does it respect the architecture? | Breaks `tech.md` invariants, ADRs, or PROFILE "Key Invariants"; unapproved dependencies; layer violations; circular imports |
| 4 | Is the diff free of unrelated changes? | Formatting churn, unrelated files, debug output, scratch files, deleted comments |
| 5 | Is existing behaviour unbroken? | Full-suite failures, incompatible signature changes |

Also check **spec sync**: every changed file appears in `plan.md` "Expected Files", and every spec item has code.

## Step 3 — Deep Review (conditional)
Run it when: the feature is Medium/Big; it touches PROFILE "Sensitive Areas"; it is the last feature of a phase; or the human asks.
Launch **parallel read-only subagents**, one per PROFILE "Deep Review Lenses" item (default: Security & privacy · Architecture & invariants · Spec drift · Test quality). Give each the feature folder, the diff summary, and its lens. Each returns `severity · file:line · issue · suggested fix`. Fix high and medium (spec first if intent changes); low → fix, `sdd-research-backlog`, or accept with a reason.

## Step 4 — `review.md` and Loop
Write `spec/features/<feature>/review.md` from `templates/review-report.md`. Any ❌ → fix → re-run validation → repeat.

## Step 5 — 🛑 GATE 2
Present (conversation language): what was built and why, the validation summary, the 5 verdicts, deep-review handling, spec changes, the manual checks for the human, and the proposed commit message. Changes → spec first, code, re-validate, re-review, present again.

## Step 6 — After Approval
1. Tick the roadmap box; set the `plan.md` Status to Committed.
2. `sdd-changelog`.
3. On `phase/<N>-<slug>`: stage only the feature's files, commit `feat(<scope>): implement Feature X.Y <title>`, then `git push -u origin phase/<N>-<slug>` (if there is a remote).
4. Never merge to the main branch here.

## Step 7 — 🛑 GATE 3
Quick replan check (`sdd-replan` mode A): is the next item still right, is the phase complete (→ mode D), has a backlog item become urgent? Recommend a fresh context, then stop.

