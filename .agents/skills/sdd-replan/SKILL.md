---
name: sdd-replan
description: Replanning in Spec-Driven Development for any project, between features and phases. Mode A quick check before the next feature/phase; mode B roadmap changes (add, merge, split, reorder, break down big features, schedule backlog items); mode C change of direction or constitution update (mission/tech/profile) with an impact analysis first; mode D phase close and merge (in phase mode only the retrospective); mode E improving the SDD process and skills themselves. Works on replan/<topic> branches and always analyses before editing.
---

# sdd-replan — Run Slow to Run Fast

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE).

| Mode | Trigger | Branch | Edit before approval? |
|:---|:---|:---|:---|
| A Quick check | Gate 3, start of a feature/phase | none | No (read-only) |
| B Roadmap change | Breakdown, merge/split/reorder, scheduling a backlog item | `replan/<topic>` | No |
| C Direction / constitution | New direction, stakeholder feedback, stack change | `replan/<topic>` | **No.** Impact analysis first |
| D Phase close | Last feature of a phase committed | `phase/…` → main | No |
| E Process improvement | Repeated friction, a new skill idea | `replan/sdd-process-<topic>` | No |

Load first: `spec/mission.md`, `spec/tech.md`, `spec/roadmap.md`, `spec/sdd-profile.md`, `spec/architecture.md`, `spec/adr/`, `spec/backlog/`, recent `review.md` / `phase-review.md`.

## A — Quick Check
1. Is the next item still right? Should items be merged or split? 2. Did reviews or the backlog surface something that should come first? 3. Is the tree clean and the branch correct? → Recommend continue / B / C / D.

## B — Roadmap Change
Propose diff-style: phases (goal), `Feature X.Y` lines (title, paths), order, dependencies; keep features small. Big breakdown → one phase per independent deliverable. A backlog item → link its file from the roadmap line and mark it `Scheduled`. After approval → edit, commit `docs(roadmap): …`, and ask before merging.

## C — Change Direction / Constitution
**No edits yet.** Write an impact analysis with **verified facts** (with file references), **assumptions**, and **recommendations**:
1. Decisions that remain valid. 2. Conflicts. 3. Reusable completed features. 4. Features/specs to modify, postpone, or drop. 5. Risks (architecture, data, API, security, privacy, performance, cost). 6. Constitution files to change (mission, tech, profile, architecture, a new ADR). 7. New phases/features. 8. Missing decisions → 3–6 questions, each with a recommended answer.
After approval: update the constitution, add an ADR for significant technical decisions, and **propagate** to affected specs and code (small propagation → now; large → schedule via mode B). Validate, present the diff, and commit after approval.

## D — Phase Close
> Phase mode: `sdd-phase` Steps 7–9 already cover validation, deep review, changelog, and merge. Use only items 3 and 6 here.
1. All phase features are `- [x]` with `review.md` or a `phase-review.md` row. 2. Phase-wide deep review plus the full validation suite. 3. **Retrospective**: what went well, spec gaps, process friction → candidates for mode E. 4. `sdd-changelog` phase summary. 5. 🛑 Approval → PR (preferred) or `git merge --no-ff phase/<N>-<slug>`, then push the main branch. 6. Propose the next phase branch.

## E — Process Improvement
Describe the friction with evidence → propose edits to the profile, the conventions, templates, or skills, or a new skill. Decide **scope**: project-only (`spec/sdd-profile.md` or the project's `.agents/skills/`) vs global (`~/.gemini/config/skills/`). Keep skills project-agnostic: project details belong in the profile. After approval → edit on `replan/sdd-process-<topic>` (global skill edits are outside the repo; report them explicitly).

## Rules
Constitution changes never ride on a feature commit (only roadmap ticks may). Intent changes always go into the spec. Prefer small, frequent replans.

