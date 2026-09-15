# Chapter 2 Release Evidence

Generated: `2026-09-15T21:01:19.020453+00:00`

## Release Decision

**BLOCK**

A high-risk sampled case failed the base evaluation.

---

## Evaluation Scope

| Property | Value |
|---|---|
| Chapter | Jason Arbon - Testing AI - Chapter 2 |
| Agent model | `qwen2.5:3b` |
| Judge model | `qwen2.5:3b` |
| Temperature | `0.5` |
| Minimum acceptable score | `7.0` |
| Sampling strategy | `risk_stratified` |
| Random seed | `42` |

Scores use a 0-10 engineering scale.
A score of 7.0 or greater is considered passing.

---

## Evaluation Population

- Total cases: **2**
- Sampled cases: **2**

### Sampled Cases

- `meeting_transcript`
- `team_call_transcript`

---

## Risk Stratification

Cases are classified into risk strata before sampling.

| Risk | Population | Sampled | Sampling Rate |
|---|---:|---:|---:|
| high | 1 | 1 | 100% |
| medium | 1 | 1 | 100% |

This is risk-based stratified sampling.
It is not a statistically calculated sample-size determination.

---

## Base Evaluation Results

| Case | Risk | Summary | Action Items | Status |
|---|---|---:|---:|---|
| meeting_transcript | high | 9.00 | 5.06 | FAIL |
| team_call_transcript | medium | 9.00 | 0.94 | FAIL |

The base evaluation checks summary quality and action-item quality.
Minor wording variation is acceptable.

---

## Metamorphic Testing

| Case | Transformation | Summary | Action Items | Status |
|---|---|---:|---:|---|
| meeting_transcript | append_irrelevant_filler | 8.92 | 4.05 | FAIL |
| team_call_transcript | append_irrelevant_filler | 8.00 | 2.67 | FAIL |

### Transformation

The metamorphic test adds irrelevant conversational filler to the original transcript.

Expected relationship:

Adding irrelevant content should not materially change important meeting facts or action items.

---

## Failure Traces

No failure traces recorded.

Failure traces capture the case, evaluation stage, error type, message, and traceback where available.

---

## Release Gate

### BLOCK

- A high-risk sampled case fails.
- A metamorphic test fails.
- An agent or evaluation failure is recorded.

### REVIEW

- A medium- or low-risk sampled case fails.
- No blocking condition exists.

### RELEASE

- All sampled base evaluations pass.
- All metamorphic evaluations pass.
- No failure traces exist.

---

## Chapter 3 Boundary

Chapter 3 statistical analysis is not implemented in this commit.

This evidence does not claim statistical confidence intervals, statistical significance, population-level confidence, or statistically sufficient sample size.

---

## Evidence Files

- `release-evidence.json`
- `sampled-cases.json`
- `metamorphic-results.json`
- `failure-traces.json`
- `release-evidence.md`
