# Chapter 4 - Statistical Tests for AI Quality

Generated: 2026-09-17T13:27:17.913675+00:00

Every number here is measured against human-written gold answers in `dataset/gold_action_items.json`. No model judges anything.

| Setting | Value |
|---|---|
| Baseline model | `qwen2.5:3b` |
| Variant model | `qwen2.5:0.5b` |
| Temperature | 0.0 |
| Match threshold | 0.6 |

## F-scores

`beta` decides which mistake the score punishes. For a meeting assistant a hallucinated action item sends someone to do work nobody assigned, so **F0.5 is the gate** and the others are context.

| arm | precision | recall | F0.5 | F1 | F2 | TP | FP | FN | parsed |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:0.5b / base | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 | 0 | 8 | 12 | 2/2 |
| qwen2.5:0.5b / filler | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 | 0 | 8 | 12 | 2/2 |
| qwen2.5:3b / base | 0.4286 | 0.5 | **0.4412** | 0.4615 | 0.4839 | 6 | 8 | 6 | 2/2 |
| qwen2.5:3b / filler | 0.3333 | 0.4167 | **0.3472** | 0.3704 | 0.3968 | 5 | 10 | 7 | 2/2 |

## Pre-registered hypotheses

Declared in `chapter4_stats.HYPOTHESES` before the run.

### `h1_filler` - Does irrelevant filler text change what the agent extracts?

- **H0:** Extraction correctness is the same before and after filler is added.
- **Test:** paired_flip_test, alpha = 0.05

**Verdict: UNDERPOWERED**

p = 1.0, but with only 3 changed items the smallest p-value possible was 0.25, which is already above alpha = 0.05. No result could have been significant, so this is not evidence of no effect.

- items the filler fixed: 1
- items the filler broke: 2

### `h2_risk` - Does a transcript's risk level predict extraction failure?

- **H0:** Correct and incorrect extractions appear in the same proportion at every risk level.
- **Test:** chi_squared, alpha = 0.05

**Verdict: NOT_TESTED**

Not tested: some expected cell counts are below 5, where the chi-squared p-value stops being trustworthy. Collect more data or merge categories.

| | extracted | missed |
|---|---|---|
| high risk | 2 | 2 |
| other risk | 4 | 4 |

### `h3_models` - Do two models fail at the same rate?

- **H0:** Both models extract correctly in the same proportion.
- **Test:** chi_squared, alpha = 0.05

**Verdict: NOT_TESTED**

Not tested: some expected cell counts are below 5, where the chi-squared p-value stops being trustworthy. Collect more data or merge categories.

| | extracted | missed |
|---|---|---|
| qwen2.5:3b | 6 | 6 |
| qwen2.5:0.5b | 0 | 12 |
