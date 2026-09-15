import json
import os
import statistics
from datetime import datetime, timezone


EVIDENCE_FILE = "./transcript-agent-eval/evidence/chapter1-evidence.json"

REPORT_JSON_FILE = "./transcript-agent-eval/evidence/chapter1-report.json"

REPORT_MD_FILE = "./transcript-agent-eval/evidence/chapter1-report.md"


# Chapter 1 release-confidence starting thresholds.
#
# These are engineering thresholds and should be tuned based on
# actual product risk and historical evaluation behaviour.
MIN_ACCEPTABLE_SCORE = 7.0
MAX_ALLOWED_STD_DEV = 1.5
MIN_PASS_RATE = 0.80


def calculate_overall_statistics(scores: list[float]) -> dict:

    if not scores:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "std_dev": 0,
            "pass_rate": 0,
        }

    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 2),
        "median": round(statistics.median(scores), 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "std_dev": round(
            statistics.stdev(scores) if len(scores) > 1 else 0,
            2,
        ),
        "pass_rate": round(
            sum(
                score >= MIN_ACCEPTABLE_SCORE
                for score in scores
            ) / len(scores),
            2,
        ),
    }


def detect_unstable_cases(cases: list[dict]) -> list[dict]:

    unstable_cases = []

    for case in cases:

        case_id = case["case_id"]

        summary_stats = case["statistics"]["summary"]
        action_stats = case["statistics"]["action_items"]

        reasons = []

        if summary_stats["std_dev"] > MAX_ALLOWED_STD_DEV:
            reasons.append(
                f"Summary standard deviation "
                f"{summary_stats['std_dev']} > {MAX_ALLOWED_STD_DEV}"
            )

        if action_stats["std_dev"] > MAX_ALLOWED_STD_DEV:
            reasons.append(
                f"Action item standard deviation "
                f"{action_stats['std_dev']} > {MAX_ALLOWED_STD_DEV}"
            )

        if summary_stats["pass_rate"] < MIN_PASS_RATE:
            reasons.append(
                f"Summary pass rate "
                f"{summary_stats['pass_rate']:.0%} < {MIN_PASS_RATE:.0%}"
            )

        if action_stats["pass_rate"] < MIN_PASS_RATE:
            reasons.append(
                f"Action item pass rate "
                f"{action_stats['pass_rate']:.0%} < {MIN_PASS_RATE:.0%}"
            )

        if summary_stats["min"] < MIN_ACCEPTABLE_SCORE:
            reasons.append(
                f"Summary minimum score "
                f"{summary_stats['min']} < {MIN_ACCEPTABLE_SCORE}"
            )

        if action_stats["min"] < MIN_ACCEPTABLE_SCORE:
            reasons.append(
                f"Action item minimum score "
                f"{action_stats['min']} < {MIN_ACCEPTABLE_SCORE}"
            )

        if reasons:

            unstable_cases.append(
                {
                    "case_id": case_id,
                    "reasons": reasons,
                }
            )

    return unstable_cases


def build_run_scores_table(
    cases: list[dict],
    metric: str,
) -> str:

    lines = []

    lines.append(
        "| Case | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Mean | Std Dev | Pass Rate |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for case in cases:

        case_id = case["case_id"]
        runs = case["runs"]
        statistics_data = case["statistics"][metric]

        scores = []

        for run in runs:

            score = run[metric]["score"]

            scores.append(score)

        run_values = [
            f"{score:.2f}"
            for score in scores
        ]

        # Ensure five run columns exist even if configuration changes.
        while len(run_values) < 5:
            run_values.append("-")

        pass_rate = statistics_data["pass_rate"]

        lines.append(
            f"| `{case_id}` "
            f"| {run_values[0]} "
            f"| {run_values[1]} "
            f"| {run_values[2]} "
            f"| {run_values[3]} "
            f"| {run_values[4]} "
            f"| {statistics_data['mean']:.2f} "
            f"| {statistics_data['std_dev']:.2f} "
            f"| {pass_rate:.0%} |"
        )

    return "\n".join(lines)


def build_variability_table(cases: list[dict]) -> str:

    lines = []

    lines.append(
        "| Case | Metric | Mean | Min | Max | Std Dev | Pass Rate |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for case in cases:

        case_id = case["case_id"]

        summary = case["statistics"]["summary"]
        action_items = case["statistics"]["action_items"]

        lines.append(
            f"| `{case_id}` | Summary | "
            f"{summary['mean']:.2f} | "
            f"{summary['min']:.2f} | "
            f"{summary['max']:.2f} | "
            f"{summary['std_dev']:.2f} | "
            f"{summary['pass_rate']:.0%} |"
        )

        lines.append(
            f"| `{case_id}` | Action Items | "
            f"{action_items['mean']:.2f} | "
            f"{action_items['min']:.2f} | "
            f"{action_items['max']:.2f} | "
            f"{action_items['std_dev']:.2f} | "
            f"{action_items['pass_rate']:.0%} |"
        )

    return "\n".join(lines)


def generate_report():

    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        evidence = json.load(file)

    cases = evidence["cases"]

    all_summary_scores = []
    all_action_item_scores = []

    for case in cases:

        for run in case["runs"]:

            all_summary_scores.append(
                run["summary"]["score"]
            )

            all_action_item_scores.append(
                run["action_items"]["score"]
            )

    overall_summary = calculate_overall_statistics(
        all_summary_scores
    )

    overall_action_items = calculate_overall_statistics(
        all_action_item_scores
    )

    unstable_cases = detect_unstable_cases(cases)

    release_status = (
        "REVIEW"
        if unstable_cases
        else "PASS"
    )

    # -------------------------------------------------------------
    # JSON report
    # -------------------------------------------------------------

    report = {
        "report_generated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "chapter": "Jason Arbon Chapter 1",

        "release_confidence": {
            "minimum_score": MIN_ACCEPTABLE_SCORE,
            "maximum_allowed_std_dev": MAX_ALLOWED_STD_DEV,
            "minimum_pass_rate": MIN_PASS_RATE,
        },

        "release_status": release_status,

        "overall": {
            "summary": overall_summary,
            "action_items": overall_action_items,
        },

        "unstable_cases": unstable_cases,
    }

    with open(REPORT_JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # -------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------

    summary_run_table = build_run_scores_table(
        cases,
        "summary",
    )

    action_item_run_table = build_run_scores_table(
        cases,
        "action_items",
    )

    variability_table = build_variability_table(cases)

    markdown = f"""# Transcript Agent — Chapter 1 Release Confidence Report

**Generated:** {report["report_generated_at"]}

**Chapter:** Jason Arbon — Testing AI, Chapter 1

**Release Status:** **{release_status}**

---

## 1. Evaluation Configuration

| Property | Value |
|---|---|
| Agent model | `{evidence["configuration"]["agent_model"]}` |
| Judge model | `{evidence["configuration"]["judge_model"]}` |
| Temperature | `{evidence["configuration"]["temperature"]}` |
| Runs per transcript | `{evidence["configuration"]["num_runs"]}` |
| Score scale | `0-10` |
| Minimum acceptable score | `{MIN_ACCEPTABLE_SCORE}` |
| Maximum allowed standard deviation | `{MAX_ALLOWED_STD_DEV}` |
| Minimum pass rate | `{MIN_PASS_RATE:.0%}` |

---

## 2. Overall Results

### Summary Quality

| Metric | Value |
|---|---:|
| Runs | {overall_summary["count"]} |
| Mean | {overall_summary["mean"]} |
| Median | {overall_summary["median"]} |
| Minimum | {overall_summary["min"]} |
| Maximum | {overall_summary["max"]} |
| Standard Deviation | {overall_summary["std_dev"]} |
| Pass Rate | {overall_summary["pass_rate"]:.0%} |

### Action Item Quality

| Metric | Value |
|---|---:|
| Runs | {overall_action_items["count"]} |
| Mean | {overall_action_items["mean"]} |
| Median | {overall_action_items["median"]} |
| Minimum | {overall_action_items["min"]} |
| Maximum | {overall_action_items["max"]} |
| Standard Deviation | {overall_action_items["std_dev"]} |
| Pass Rate | {overall_action_items["pass_rate"]:.0%} |

---

## 3. Run-by-Run Scores

The following tables show the individual GEval scores for every repeated
run of each transcript.

This allows run-to-run nondeterminism to be inspected directly rather than
only looking at aggregate statistics.

### Summary Quality — Run Scores

{summary_run_table}

### Action Item Quality — Run Scores

{action_item_run_table}

---

## 4. Run-to-Run Variability

{variability_table}

**Note:** `Std Dev` represents the standard deviation of the 0–10 quality
scores across repeated runs of the same transcript. The current implementation
uses sample standard deviation.

---

## 5. Unstable Cases

"""

    if unstable_cases:

        markdown += (
            "The following cases require review because one or more "
            "Chapter 1 release-confidence thresholds were breached.\n\n"
        )

        for case in unstable_cases:

            markdown += f"### `{case['case_id']}`\n\n"

            for reason in case["reasons"]:
                markdown += f"- {reason}\n"

            markdown += "\n"

    else:

        markdown += (
            "No unstable cases were detected using the current "
            "Chapter 1 thresholds.\n"
        )

    markdown += f"""

---

## 6. Chapter 1 Interpretation

The evaluation repeats the same transcript {evidence["configuration"]["num_runs"]}
times using the production temperature of {evidence["configuration"]["temperature"]}.

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
"""

    with open(REPORT_MD_FILE, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"JSON report written to: {REPORT_JSON_FILE}")
    print(f"Markdown report written to: {REPORT_MD_FILE}")
    print(f"Release status: {release_status}")


if __name__ == "__main__":
    generate_report()