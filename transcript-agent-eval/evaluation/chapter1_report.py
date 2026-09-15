import json
import os
import statistics
from datetime import datetime, timezone


EVIDENCE_DIR = "./transcript-agent-eval/evidence"

EVIDENCE_FILE = os.path.join(
    EVIDENCE_DIR,
    "chapter1-evidence.json",
)

REPORT_JSON = os.path.join(
    EVIDENCE_DIR,
    "chapter1-report.json",
)

REPORT_MD = os.path.join(
    EVIDENCE_DIR,
    "chapter1-report.md",
)


# Chapter 1 engineering thresholds
MIN_ACCEPTABLE_SCORE = 7.0
MAX_ALLOWED_STD_DEV = 1.5
MIN_PASS_RATE = 0.80


def load_evidence():

    with open(
        EVIDENCE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_overall_statistics(
    cases,
    metric_name,
):
    scores = []

    for case in cases:

        for run in case["runs"]:

            score = run[metric_name]["score"]

            scores.append(score)

    if not scores:
        return {}

    return {
        "count": len(scores),

        "mean": round(
            statistics.mean(scores),
            2,
        ),

        "median": round(
            statistics.median(scores),
            2,
        ),

        "min": round(
            min(scores),
            2,
        ),

        "max": round(
            max(scores),
            2,
        ),

        "std_dev": round(
            statistics.stdev(scores),
            2,
        ) if len(scores) > 1 else 0.0,

        "pass_rate": round(
            sum(
                score >= MIN_ACCEPTABLE_SCORE
                for score in scores
            ) / len(scores),
            4,
        ),
    }


def detect_unstable_cases(evidence):

    unstable_cases = []

    for case in evidence["cases"]:

        summary_stats = case[
            "summary_statistics"
        ]

        action_stats = case[
            "action_item_statistics"
        ]

        reasons = []

        # High summary variance
        if (
            summary_stats["std_dev"]
            > MAX_ALLOWED_STD_DEV
        ):
            reasons.append(
                f"summary std_dev "
                f"{summary_stats['std_dev']} "
                f"> {MAX_ALLOWED_STD_DEV}"
            )

        # High action-item variance
        if (
            action_stats["std_dev"]
            > MAX_ALLOWED_STD_DEV
        ):
            reasons.append(
                f"action-item std_dev "
                f"{action_stats['std_dev']} "
                f"> {MAX_ALLOWED_STD_DEV}"
            )

        # Low summary pass rate
        if (
            summary_stats["pass_rate"]
            < MIN_PASS_RATE
        ):
            reasons.append(
                f"summary pass rate "
                f"{summary_stats['pass_rate'] * 100:.1f}% "
                f"< {MIN_PASS_RATE * 100:.1f}%"
            )

        # Low action-item pass rate
        if (
            action_stats["pass_rate"]
            < MIN_PASS_RATE
        ):
            reasons.append(
                f"action-item pass rate "
                f"{action_stats['pass_rate'] * 100:.1f}% "
                f"< {MIN_PASS_RATE * 100:.1f}%"
            )

        # Low summary minimum score
        if (
            summary_stats["min"]
            < MIN_ACCEPTABLE_SCORE
        ):
            reasons.append(
                f"summary minimum score "
                f"{summary_stats['min']} "
                f"< {MIN_ACCEPTABLE_SCORE}"
            )

        # Low action-item minimum score
        if (
            action_stats["min"]
            < MIN_ACCEPTABLE_SCORE
        ):
            reasons.append(
                f"action-item minimum score "
                f"{action_stats['min']} "
                f"< {MIN_ACCEPTABLE_SCORE}"
            )

        if reasons:

            unstable_cases.append({
                "case_id": case["case_id"],

                "reasons": reasons,

                "summary_statistics":
                    summary_stats,

                "action_item_statistics":
                    action_stats,
            })

    return unstable_cases


def build_variance_table(evidence):

    table = []

    for case in evidence["cases"]:

        summary = case[
            "summary_statistics"
        ]

        table.append({
            "case_id": case["case_id"],
            "metric": "summary",
            "mean": summary["mean"],
            "min": summary["min"],
            "max": summary["max"],
            "std_dev": summary["std_dev"],
            "pass_rate": summary["pass_rate"],
        })

        actions = case[
            "action_item_statistics"
        ]

        table.append({
            "case_id": case["case_id"],
            "metric": "action_items",
            "mean": actions["mean"],
            "min": actions["min"],
            "max": actions["max"],
            "std_dev": actions["std_dev"],
            "pass_rate": actions["pass_rate"],
        })

    return table


def build_report(evidence):

    cases = evidence["cases"]

    summary_overall = calculate_overall_statistics(
        cases,
        "summary",
    )

    action_overall = calculate_overall_statistics(
        cases,
        "action_items",
    )

    unstable_cases = detect_unstable_cases(
        evidence
    )

    return {

        "report_generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "chapter":
            "Jason Arbon Chapter 1",

        "release_confidence": {

            "minimum_score":
                MIN_ACCEPTABLE_SCORE,

            "maximum_allowed_std_dev":
                MAX_ALLOWED_STD_DEV,

            "minimum_pass_rate":
                MIN_PASS_RATE,
        },

        "overall": {

            "summary":
                summary_overall,

            "action_items":
                action_overall,
        },

        "variance_table":
            build_variance_table(
                evidence
            ),

        "unstable_cases":
            unstable_cases,
    }


def generate_markdown(report):

    overall = report["overall"]

    lines = []

    lines.append(
        "# Transcript Agent — Chapter 1 Release Report"
    )

    lines.append("")

    lines.append(
        "## Overall Results"
    )

    lines.append("")

    lines.append(
        "| Metric | Mean | Min | Max | "
        "Std Dev | Pass Rate |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for metric, values in overall.items():

        lines.append(
            f"| {metric} "
            f"| {values['mean']} "
            f"| {values['min']} "
            f"| {values['max']} "
            f"| {values['std_dev']} "
            f"| {values['pass_rate'] * 100:.1f}% |"
        )

    lines.append("")

    lines.append(
        "## Variance Table"
    )

    lines.append("")

    lines.append(
        "| Case | Metric | Mean | Min | Max | "
        "Std Dev | Pass Rate |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for row in report["variance_table"]:

        lines.append(
            f"| {row['case_id']} "
            f"| {row['metric']} "
            f"| {row['mean']} "
            f"| {row['min']} "
            f"| {row['max']} "
            f"| {row['std_dev']} "
            f"| {row['pass_rate'] * 100:.1f}% |"
        )

    lines.append("")

    lines.append(
        "## Unstable Cases"
    )

    lines.append("")

    unstable = report["unstable_cases"]

    if not unstable:

        lines.append(
            "No unstable cases detected."
        )

    else:

        lines.append(
            f"**{len(unstable)} unstable "
            f"case(s) detected.**"
        )

        lines.append("")

        for case in unstable:

            lines.append(
                f"### {case['case_id']}"
            )

            for reason in case["reasons"]:

                lines.append(
                    f"- {reason}"
                )

            lines.append("")

    lines.append(
        "## Chapter 1 Interpretation"
    )

    lines.append("")

    lines.append(
        "Each transcript was evaluated multiple "
        "times to measure behavioural variance "
        "rather than relying on a single execution."
    )

    lines.append("")

    lines.append(
        "Cases with high score variance, low "
        "repeat-run pass rates, or low minimum "
        "scores are flagged for investigation."
    )

    return "\n".join(lines)


def main():

    os.makedirs(
        EVIDENCE_DIR,
        exist_ok=True,
    )

    evidence = load_evidence()

    report = build_report(
        evidence
    )

    with open(
        REPORT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    markdown = generate_markdown(
        report
    )

    with open(
        REPORT_MD,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(markdown)

    print(markdown)

    print()
    print(
        f"JSON report: {REPORT_JSON}"
    )

    print(
        f"Markdown report: {REPORT_MD}"
    )


if __name__ == "__main__":
    main()