---
name: sdd-research-backlog
description: Captures an idea, research question, tech-debt item, or risk that comes up mid-work in any project, without changing the current scope or the roadmap. Use when the user says "research X", "note this for later", "add to backlog", or "what if we used Y instead", or when sdd-implement or sdd-review finds out-of-scope work. Writes spec/backlog/<date>-<slug>.md and indexes it, so sdd-replan can schedule it later.
---

# sdd-research-backlog — Keep Ideas, Don't Derail the Work

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV).

## Steps
1. **Stay in scope**: don't touch application code, the roadmap, or the current spec.
2. Research (if needed): explore the code, docs, and web with the human; 1–4 questions, each with a recommended answer. Record which recommendations were accepted, changed, or rejected. A changed mind is valuable context.
3. Write `spec/backlog/<YYYY-MM-DD>-<kebab-slug>.md` from `sdd-conventions/templates/backlog-item.md`.
4. Add a row to `spec/backlog/README.md` (create it if missing): `| Date | Item | Type | Status | Raised during |`.
5. Commit with the current work when convenient (`docs(backlog): …`). This is safe on any branch because it changes no intent.
6. Tell the human where it was saved. Scheduling happens via `sdd-replan` mode B (links the file from the roadmap, Status → Scheduled).

## Rules
The backlog isn't the roadmap: nothing here is committed work. One file per topic; update the existing file rather than duplicating it.

