# Validation Criteria: Feature X.Y — <Feature Title>

## 1. Success Definition
One paragraph: what evidence proves this feature is completely successful.

## 2. Requirement Traceability
| Requirement | Checks |
|:---|:---|
| R1 | T1.1, T2.1 |
| NFR1 | T3.2 |

---

## 3. Verification Tiers
Each tier is either a checklist of real checks **or** `Status: Not Applicable` + a concrete technical justification.

### Tier 1: Unit Verification
- [ ] T1.1 — <input> → <expected output> (`<test file>::<test>`)

### Tier 2: Integration Verification
- [ ] T2.1 — <component A → B → C flow and expected state>

### Tier 3: Error & Edge-Case Verification
- [ ] T3.1 — Abnormal: <invalid input> → <graceful behaviour>
- [ ] T3.2 — Boundary: <limit / timeout> → <expected>

### Tier 4: Regression Checks
- [ ] T4.1 — Full suite passes: `<profile: Full test suite>`

### Tier 5: Build, Lint & Type Checks
- [ ] T5.1 — `<profile: Lint>` clean
- [ ] T5.2 — `<profile: Type check>` (or Not Applicable + justification if not configured)
- [ ] T5.3 — `<profile: Build>` (if applicable)

### Tier 6: Manual / Visual Verification
- [ ] T6.1 — Launch: `<profile: Run app>` → do: <steps> → expect: <what the human sees/hears>

### Tier 7: API / CLI / Application Checks
- [ ] T7.1 — `<real command or request>` → <expected live output>

### Extra Gates (from profile, if applicable)
- [ ] TX.1 — `<profile: Extra gates>`

---

## 4. Automated Execution Commands
```text
<targeted test command>
<full suite command>
<lint command>
```

## 5. Execution Results (filled in execute mode)
| Check | Result (✅ / ❌ / ⏭ N/A) | Evidence (command output summary, counts) |
|:---|:---|:---|

