---
name: sdd-changelog
description: Updates CHANGELOG.md in plain, stakeholder-friendly language for any project after a feature is approved (called by sdd-review) or at phase completion (called by sdd-phase / sdd-replan mode D). Use when the user says "update the changelog" or "write release notes". Reads the feature spec, review, and diff, never chat memory.
---

# sdd-changelog — Talk to Stakeholders

Rules: `.agents/skills/sdd-conventions/SKILL.md` (CONV) + `spec/sdd-profile.md` (PROFILE "Changelog", "Language").

## File & Format
PROFILE changelog file (default `CHANGELOG.md` at the repo root), in [Keep a Changelog](https://keepachangelog.com) style, in the artifact language. Create it if missing:
```markdown
# Changelog
All notable changes to <product>. Format: Keep a Changelog. Features reference spec/roadmap.md.

## [Unreleased]
### Phase N — <title>
#### Added
- **Feature N.M — <title>**: <user impact in plain words> ([spec](spec/features/<folder>/plan.md))
#### Changed
#### Fixed
#### Security
```

## Per Feature
1. Read `plan.md` Objective, `requirements.md`, `review.md` (or the `phase-review.md` row), and `git diff --stat`.
2. Add 1–3 bullets under `[Unreleased]` → the phase → the right category.
3. Plain language: **user impact first**, then an optional short technical note. No secrets, personal data, or internal URLs.
4. Link the spec folder. The change goes into the same commit as the feature (feature mode) or the changelog commit (phase mode).

## Phase Summary
2–4 sentences under the phase heading: what it delivered, plus known limitations. On release, move `[Unreleased]` content to `## [x.y.z] - YYYY-MM-DD`.

## Self-Check
Understandable by a non-developer · matches what was built (the review), not what was planned · no duplicates.

