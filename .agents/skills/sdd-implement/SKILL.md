---
name: sdd-implement
description: Implements an approved feature spec in Spec-Driven Development for any project. Used in feature mode after Gate 1 ("implement", "go ahead", or the approval word on a feature spec, or "implement group N of Feature X.Y"), and by the sdd-phase orchestrator inside each fresh per-feature subagent (no commit, no review, report back). Re-reads the spec from disk, implements execution group by execution group, keeps spec and code in sync, then runs sdd-validation execute mode and, in feature mode, sdd-review.
---

# sdd-implement — Build From the Spec

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE).

## Preconditions (stop if any fail)
- The feature spec exists and was explicitly approved (Gate 1, or Phase Gate 1).
- The correct `phase/<N>-<slug>` branch is checked out.
- The working tree contains only expected changes.

## Step 1 — Load From Disk
The feature's `plan.md`, `requirements.md`, `validation.md`; `spec/tech.md` invariants; PROFILE (commands, key invariants); the source files listed in "Expected Files". The spec is the contract, not the conversation.

## Step 2 — Implement by Execution Group
For each group, in order (or only the groups the human named):
1. Implement only what the group and the requirements ask. **No scope creep**; extra ideas go to `sdd-research-backlog`.
2. Respect every constraint, `tech.md` invariant, and PROFILE "Key Invariants" item.
3. Write the tests that `validation.md` lists for the group. They must exercise real behaviour (see `sdd-validation` Rule 1).
4. Run the targeted tests (PROFILE) and fix failures before moving on.
5. Preserve existing comments and docstrings you didn't intentionally change.

For PROFILE "Sensitive Areas", prefer one group at a time with a short report between groups.

## Step 3 — Keep Spec and Code in Sync
- Small, obvious spec corrections (paths, helpers within scope) → update the spec files and list the change in the report.
- A change to intent, scope, or an invariant → stop, explain, propose the spec change, and wait.
Never leave code that the spec doesn't describe, or spec items that were silently skipped.

## Step 4 — Validate (and Review)
1. Run `sdd-validation` **execute mode**.
2. **Feature mode** → `sdd-review` (ends at Gate 2).
   **Phase-mode subagent** → skip review and return the report to the orchestrator. If still red after 2 fix attempts, or an intent/scope/invariant change is needed, stop and report.

## Step 5 — Report (conversation language)
What was done per group and why (R-IDs), spec changes, and the manual checks the human must run (exact commands). No commit, push, or roadmap tick here.

