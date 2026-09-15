# Transcript Agent — Chapter 1 Release Report

## Overall Results

| Metric | Mean | Min | Max | Std Dev | Pass Rate |
|---|---:|---:|---:|---:|---:|
| summary | 3.4 | 2.0 | 7.0 | 1.78 | 10.0% |
| action_items | 4.6 | 2.0 | 8.0 | 2.41 | 30.0% |

## Variance Table

| Case | Metric | Mean | Min | Max | Std Dev | Pass Rate |
|---|---|---:|---:|---:|---:|---:|
| meeting_transcript | summary | 2.8 | 2.0 | 5.0 | 1.3 | 0.0% |
| meeting_transcript | action_items | 6.6 | 5.0 | 8.0 | 1.14 | 60.0% |
| team_call_transcript | summary | 4.0 | 2.0 | 7.0 | 2.12 | 20.0% |
| team_call_transcript | action_items | 2.6 | 2.0 | 5.0 | 1.34 | 0.0% |

## Unstable Cases

**2 unstable case(s) detected.**

### meeting_transcript
- summary pass rate 0.0% < 80.0%
- action-item pass rate 60.0% < 80.0%
- summary minimum score 2.0 < 7.0
- action-item minimum score 5.0 < 7.0

### team_call_transcript
- summary std_dev 2.12 > 1.5
- summary pass rate 20.0% < 80.0%
- action-item pass rate 0.0% < 80.0%
- summary minimum score 2.0 < 7.0
- action-item minimum score 2.0 < 7.0

## Chapter 1 Interpretation

Each transcript was evaluated multiple times to measure behavioural variance rather than relying on a single execution.

Cases with high score variance, low repeat-run pass rates, or low minimum scores are flagged for investigation.