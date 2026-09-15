# Transcript Agent — Chapter 1 Release Confidence Report

**Generated:** 2026-09-15T14:08:05.421545+00:00

**Chapter:** Jason Arbon — Testing AI, Chapter 1

**Release Status:** **REVIEW**

---

## 1. Evaluation Configuration

| Property | Value |
|---|---|
| Agent model | `qwen2.5:3b` |
| Judge model | `gpt-4.1-mini` |
| Temperature | `0.5` |
| Runs per transcript | `5` |
| Score scale | `0-10` |
| Minimum acceptable score | `7.0` |
| Maximum allowed standard deviation | `1.5` |
| Minimum pass rate | `80%` |

---

## 2. Overall Results

### Summary Quality

| Metric | Value |
|---|---:|
| Runs | 10 |
| Mean | 8.96 |
| Median | 9.0 |
| Minimum | 8.56 |
| Maximum | 9.01 |
| Standard Deviation | 0.14 |
| Pass Rate | 100% |

### Action Item Quality

| Metric | Value |
|---|---:|
| Runs | 10 |
| Mean | 5.8 |
| Median | 6.55 |
| Minimum | 1.99 |
| Maximum | 9.0 |
| Standard Deviation | 2.76 |
| Pass Rate | 50% |

---

## 3. Run-by-Run Scores

The following tables show the individual GEval scores for every repeated
run of each transcript.

This allows run-to-run nondeterminism to be inspected directly rather than
only looking at aggregate statistics.

### Summary Quality — Run Scores

| Case | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std Dev | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `meeting_transcript` | 9.00 | 9.01 | 9.00 | 8.56 | 9.00 | 8.91 | 0.20 | 100% |
| `team_call_transcript` | 9.00 | 9.00 | 9.01 | 9.00 | 9.00 | 9.00 | 0.00 | 100% |

### Action Item Quality — Run Scores

| Case | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std Dev | Pass Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `meeting_transcript` | 8.32 | 9.00 | 5.23 | 8.27 | 7.88 | 7.74 | 1.46 | 80% |
| `team_call_transcript` | 2.68 | 1.99 | 7.95 | 3.68 | 3.02 | 3.86 | 2.36 | 20% |

---

## 4. Run-to-Run Variability

| Case | Metric | Mean | Min | Max | Std Dev | Pass Rate |
|---|---|---:|---:|---:|---:|---:|
| `meeting_transcript` | Summary | 8.91 | 8.56 | 9.01 | 0.20 | 100% |
| `meeting_transcript` | Action Items | 7.74 | 5.23 | 9.00 | 1.46 | 80% |
| `team_call_transcript` | Summary | 9.00 | 9.00 | 9.01 | 0.00 | 100% |
| `team_call_transcript` | Action Items | 3.86 | 1.99 | 7.95 | 2.36 | 20% |

**Note:** `Std Dev` represents the standard deviation of the 0–10 quality
scores across repeated runs of the same transcript. The current implementation
uses sample standard deviation.

---

## 5. Unstable Cases

The following cases require review because one or more Chapter 1 release-confidence thresholds were breached.

### `meeting_transcript`

- Action item minimum score 5.23 < 7.0

### `team_call_transcript`

- Action item standard deviation 2.36 > 1.5
- Action item pass rate 20% < 80%
- Action item minimum score 1.99 < 7.0



---

## 6. Chapter 1 Interpretation

The evaluation repeats the same transcript 5
times using the production temperature of 0.5.

The purpose is to measure **run-to-run quality variability** in the
non-deterministic transcript agent.

The release-confidence checks currently consider:

1. Minimum quality score
2. Run-to-run standard deviation
3. Pass rate
4. Minimum observed score

These thresholds are engineering starting points and should be calibrated
against historical evaluation results and product risk.

This report does not make statistical-significance claims and does not
calculate confidence intervals. Those techniques will be introduced in the
later statistical evaluation stages.

---

## 7. Evidence

Raw run-level evaluation evidence:

`chapter1-evidence.json`

Machine-readable release report:

`chapter1-report.json`
