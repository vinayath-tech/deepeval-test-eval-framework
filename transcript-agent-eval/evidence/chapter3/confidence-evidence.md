# Chapter 3 Confidence Evidence

Generated at: `2026-09-15T22:43:24.450901+00:00`

---

## Confidence Gate

**Decision: BLOCK**

A high-risk case did not demonstrate, at the 95% confidence level, that it meets the release threshold.

| Severity | Case | Risk | Metric | Measure | Verdict |
|---|---|---|---|---|---|
| BLOCK | meeting_transcript | high | summary | pass_rate | INCONCLUSIVE |
| BLOCK | meeting_transcript | high | action_items | score | INCONCLUSIVE |
| BLOCK | meeting_transcript | high | action_items | pass_rate | INCONCLUSIVE |
| REVIEW | team_call_transcript | medium | summary | pass_rate | INCONCLUSIVE |
| REVIEW | team_call_transcript | medium | action_items | score | BELOW_THRESHOLD |
| REVIEW | team_call_transcript | medium | action_items | pass_rate | BELOW_THRESHOLD |

This gate reads the bounds of each interval, not the point estimate. `INCONCLUSIVE` means the sample cannot separate passing from failing, which is different from failing.

---

## Configuration

| Setting | Value |
|---|---|
| Confidence level | 95% |
| Minimum acceptable score | 7.0 |
| Minimum acceptable pass rate | 80% |
| Proportion method | wilson |
| Mean method | t |
| Difference method | paired_t |
| Runs per case (Chapter 1) | 5 |

---

## Average Score Intervals

Calculated over the repeated runs Chapter 1 recorded for each transcript, using:

```
mean           = sum(scores) / n
standard_error = sample_standard_deviation / sqrt(n)
interval       = mean +/- t_multiplier * standard_error
```

| Case | Risk | Metric | n | Mean | 95% Interval | Multiplier | Verdict |
|---|---|---|---:|---:|---|---:|---|
| meeting_transcript | high | summary | 5 | 8.91 | 8.67 to 9.16 | 2.776 | MEETS_THRESHOLD |
| meeting_transcript | high | action_items | 5 | 7.74 | 5.93 to 9.55 | 2.776 | INCONCLUSIVE |
| team_call_transcript | medium | summary | 5 | 9.00 | 9.00 to 9.01 | 2.776 | MEETS_THRESHOLD |
| team_call_transcript | medium | action_items | 5 | 3.86 | 0.93 to 6.80 | 2.776 | BELOW_THRESHOLD |

The multiplier column shows why 1.96 is not used here. With five runs the sample has four degrees of freedom and the correct multiplier is 2.776, which makes the interval wider and the conclusion more honest.

---

## Pass Rate Intervals

Calculated over the same runs, treating each run as pass or fail against the score threshold.

| Case | Metric | Passes | Observed | Wilson Interval | Wald Interval | Verdict |
|---|---|---|---:|---|---|---|
| meeting_transcript | summary | 5/5 | 100% | 57% to 100% | 100% to 100% | INCONCLUSIVE |
| meeting_transcript | action_items | 4/5 | 80% | 38% to 96% | 45% to 100% | INCONCLUSIVE |
| team_call_transcript | summary | 5/5 | 100% | 57% to 100% | 100% to 100% | INCONCLUSIVE |
| team_call_transcript | action_items | 1/5 | 20% | 4% to 62% | 0% to 55% | BELOW_THRESHOLD |

Both methods are shown deliberately. Where a case passed every run, the Wald interval collapses to 100% to 100%, claiming certainty from a handful of runs. The Wilson interval does not, which is why it drives the verdict.

---

## Pooled Across Cases

| Metric | n | Mean | Score Interval | Pass Rate | Pass Rate Interval |
|---|---:|---:|---|---:|---|
| summary | 10 | 8.96 | 8.86 to 9.06 | 100% | 72% to 100% |
| action_items | 10 | 5.80 | 3.83 to 7.77 | 50% | 24% to 76% |

Pooling raises n and narrows the interval, but it mixes transcripts of different difficulty, so a pooled average can hide a case that is consistently weak. The per-case tables above remain the basis for the gate.

---

## Base vs Metamorphic (Paired)

The same cases run through both the original and the transformed transcript, so the comparison is paired: the difference is taken per case first, and the interval is built around those differences.

| Metric | Pairs | Base Mean | Metamorphic Mean | Difference | Interval | Crosses Zero |
|---|---:|---:|---:|---:|---|---|
| summary | 2 | 9.00 | 8.46 | -0.54 | -6.38 to 5.30 | yes |
| action_items | 2 | 3.00 | 3.36 | 0.36 | -17.05 to 17.77 | yes |

An interval that crosses zero means these data do not clearly separate the two versions. It is not proof the transformation was harmless.

---

## Plain-English Summary

**`meeting_transcript`**

- The average summary quality score was 8.91. The approximate 95% confidence interval is about 8.67 to 9.16 (n=5).
- In this sample, 100% of summary quality score runs passed (5/5). The approximate 95% confidence interval is about 57% to 100% (wilson method).
- The average action item quality score was 7.74. The approximate 95% confidence interval is about 5.93 to 9.55 (n=5).
- In this sample, 80% of action item quality score runs passed (4/5). The approximate 95% confidence interval is about 38% to 96% (wilson method).

**`team_call_transcript`**

- The average summary quality score was 9.0. The approximate 95% confidence interval is about 9.0 to 9.01 (n=5).
- In this sample, 100% of summary quality score runs passed (5/5). The approximate 95% confidence interval is about 57% to 100% (wilson method).
- The average action item quality score was 3.86. The approximate 95% confidence interval is about 0.93 to 6.8 (n=5).
- In this sample, 20% of action item quality score runs passed (1/5). The approximate 95% confidence interval is about 4% to 62% (wilson method).


---

## Sample Size Notes

- `meeting_transcript` / summary: Sample size is 5. The interval is wide. Increasing the number of runs will narrow it.
- `meeting_transcript` / action_items: Sample size is 5. The interval is wide. Increasing the number of runs will narrow it.
- `meeting_transcript` / action_items: about 33 runs would narrow the margin of error to +/-0.5 if the spread stays the same.
- `team_call_transcript` / summary: Sample size is 5. The interval is wide. Increasing the number of runs will narrow it.
- `team_call_transcript` / action_items: Sample size is 5. The interval is wide. Increasing the number of runs will narrow it.
- Chapter 2 sample / summary: All 2 observations were identical, so the interval has zero width. This is not evidence of certainty. Sample size is 2. The interval is very wide and should not be treated as evidence for a high-risk release.
- Chapter 2 sample / action_items: Sample size is 2. The interval is very wide and should not be treated as evidence for a high-risk release.

---

## Method Notes

- Intervals follow the pattern estimate +/- multiplier * standard error.
- The t/normal multipliers and the interval calculations themselves are computed by scipy.stats (ttest_1samp, ttest_ind, binomtest), not by a hand-rolled formula or a lookup table.
- Score intervals use the t multiplier because the samples are small; 1.96 would be too narrow.
- Pass rate intervals use the Wilson method (scipy.stats.binomtest), which behaves correctly near 0% and 100% where the simple Wald formula collapses to zero width.
- Base and metamorphic results are compared as paired differences per case, not as two marginal intervals.
- No claim is made that these intervals are wide enough samples for a high-risk release. Where they are not, the verdict is INCONCLUSIVE rather than PASS.

---

## Scope

These intervals describe the transcripts and runs that were actually evaluated. They do not describe meeting transcripts in general, and they assume the judge itself is unbiased, which this packet does not test.
