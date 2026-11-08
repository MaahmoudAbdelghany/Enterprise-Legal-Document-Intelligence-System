# Requirements: Feature X.Y — <Feature Title>

> Decisions and behaviours, not implementation micro-details. Every requirement must be testable and map to at least one check in `validation.md`.

## 1. Functional Requirements
| ID | Requirement | Validated by (validation.md check) |
|:---|:---|:---|
| R1 | The system must … | T1.1 |

## 2. Non-Functional Requirements
| ID | Requirement (performance budget, security, privacy, cost, accessibility, i18n…) | Validated by |
|:---|:---|:---|
| NFR1 | … | … |

## 3. Constraints (must NOT do / change)
- C1: Must respect `spec/tech.md` invariant "<name>".
- C2: Must not modify <module/behaviour> outside scope.

## 4. Dependencies & Assumptions
- Depends on: Feature X.Y (done), external services, env vars (names only, never values).
- Assumptions (flag clearly; each must be confirmed or accepted at the gate).
