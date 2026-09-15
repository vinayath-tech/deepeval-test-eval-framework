"""
Jason Arbon Testing AI - Chapter 3

Renders confidence-evidence.json as a readable markdown report.

Follows the same shape as chapter2_report.py: read the machine-readable
evidence packet, render tables, make no new calculations.
"""

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = BASE_DIR / "evidence" / "chapter3"
EVIDENCE_FILE = EVIDENCE_DIR / "confidence-evidence.json"
REPORT_FILE = EVIDENCE_DIR / "confidence-evidence.md"


def load_evidence():
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def format_bound(value):
    if value is None:
        return "N/A"

    return f"{float(value):.2f}"


def format_percent(value):
    if value is None:
        return "N/A"

    return f"{float(value):.0%}"


def format_interval(interval):
    if interval.get("lower") is None:
        return "not calculable"

    return (
        f"{format_bound(interval['lower'])} to "
        f"{format_bound(interval['upper'])}"
    )


def format_percent_interval(interval):
    return (
        f"{format_percent(interval['lower'])} to "
        f"{format_percent(interval['upper'])}"
    )


def generate_score_table(evidence):
    rows = [
        "| Case | Risk | Metric | n | Mean | 95% Interval | Multiplier | Verdict |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]

    for case in evidence["repeated_run_analysis"]["cases"]:

        for metric, report in case["metrics"].items():

            interval = report["score_interval"]

            multiplier = interval.get("multiplier")

            rows.append(
                f"| {case['case_id']} | "
                f"{case['risk_level']} | "
                f"{metric} | "
                f"{interval['n']} | "
                f"{format_bound(interval['estimate'])} | "
                f"{format_interval(interval)} | "
                f"{multiplier if multiplier is not None else 'N/A'} | "
                f"{report['score_verdict']['verdict']} |"
            )

    return "\n".join(rows)


def generate_pass_rate_table(evidence):
    rows = [
        "| Case | Metric | Passes | Observed | Wilson Interval | Wald Interval | Verdict |",
        "|---|---|---|---:|---|---|---|",
    ]

    for case in evidence["repeated_run_analysis"]["cases"]:

        for metric, report in case["metrics"].items():

            wilson = report["pass_rate_interval"]
            wald = report["pass_rate_interval_wald"]

            rows.append(
                f"| {case['case_id']} | "
                f"{metric} | "
                f"{wilson['passes']}/{wilson['total']} | "
                f"{format_percent(wilson['estimate'])} | "
                f"{format_percent_interval(wilson)} | "
                f"{format_percent_interval(wald)} | "
                f"{report['pass_rate_verdict']['verdict']} |"
            )

    return "\n".join(rows)


def generate_pooled_table(evidence):
    pooled = evidence["repeated_run_analysis"].get("pooled", {})

    if not pooled:
        return "No pooled results available."

    rows = [
        "| Metric | n | Mean | Score Interval | Pass Rate | Pass Rate Interval |",
        "|---|---:|---:|---|---:|---|",
    ]

    for metric, report in pooled.items():

        score = report["score_interval"]
        wilson = report["pass_rate_interval"]

        rows.append(
            f"| {metric} | "
            f"{score['n']} | "
            f"{format_bound(score['estimate'])} | "
            f"{format_interval(score)} | "
            f"{format_percent(wilson['estimate'])} | "
            f"{format_percent_interval(wilson)} |"
        )

    return "\n".join(rows)


def generate_paired_table(evidence):
    release = evidence.get("release_sample_analysis") or {}

    paired = release.get("paired_metamorphic", {})

    if not paired:
        return "No paired metamorphic comparison available."

    rows = [
        "| Metric | Pairs | Base Mean | Metamorphic Mean | Difference | Interval | Crosses Zero |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for metric, report in paired.items():

        interval = report["difference_interval"]

        rows.append(
            f"| {metric} | "
            f"{interval['pairs']} | "
            f"{format_bound(interval['mean_before'])} | "
            f"{format_bound(interval['mean_after'])} | "
            f"{format_bound(interval['estimate'])} | "
            f"{format_interval(interval)} | "
            f"{'yes' if interval['crosses_zero'] else 'no'} |"
        )

    return "\n".join(rows)


def generate_findings_table(evidence):
    gate = evidence["confidence_gate"]

    findings = (
        [("BLOCK", item) for item in gate["blocking_findings"]]
        + [("REVIEW", item) for item in gate["review_findings"]]
    )

    if not findings:
        return "No blocking or review findings."

    rows = [
        "| Severity | Case | Risk | Metric | Measure | Verdict |",
        "|---|---|---|---|---|---|",
    ]

    for severity, item in findings:
        rows.append(
            f"| {severity} | "
            f"{item['case_id']} | "
            f"{item['risk_level']} | "
            f"{item['metric']} | "
            f"{item['measure']} | "
            f"{item['verdict']} |"
        )

    return "\n".join(rows)


def generate_warnings(evidence):
    warnings = []

    for case in evidence["repeated_run_analysis"]["cases"]:

        for metric, report in case["metrics"].items():

            warning = report["score_interval"].get("sample_size_warning")

            if warning:
                warnings.append(
                    f"- `{case['case_id']}` / {metric}: {warning}"
                )

            needed = report.get("runs_needed_for_target_margin")

            if needed and needed.get("estimated_runs"):
                warnings.append(
                    f"- `{case['case_id']}` / {metric}: about "
                    f"{needed['estimated_runs']} runs would narrow the "
                    f"margin of error to +/-{needed['target_margin']} "
                    f"if the spread stays the same."
                )

    release = evidence.get("release_sample_analysis") or {}

    for metric, report in release.get("across_cases", {}).items():

        warning = report["score_interval"].get("sample_size_warning")

        if warning:
            warnings.append(
                f"- Chapter 2 sample / {metric}: {warning}"
            )

    if not warnings:
        return "No sample size warnings."

    return "\n".join(warnings)


def generate_statements(evidence):
    lines = []

    for case in evidence["repeated_run_analysis"]["cases"]:

        lines.append(f"**`{case['case_id']}`**")
        lines.append("")

        for metric, report in case["metrics"].items():
            lines.append(f"- {report['score_statement']}")
            lines.append(f"- {report['pass_rate_statement']}")

        lines.append("")

    return "\n".join(lines)


def generate_report(evidence):
    config = evidence["configuration"]
    gate = evidence["confidence_gate"]
    repeated = evidence["repeated_run_analysis"]

    lines = []

    lines.append("# Chapter 3 Confidence Evidence")
    lines.append("")

    lines.append(f"Generated at: `{evidence['generated_at']}`")
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------------

    lines.append("## Confidence Gate")
    lines.append("")

    lines.append(f"**Decision: {gate['decision']}**")
    lines.append("")

    lines.append(gate["reason"])
    lines.append("")

    lines.append(generate_findings_table(evidence))
    lines.append("")

    lines.append(
        "This gate reads the bounds of each interval, not the point "
        "estimate. `INCONCLUSIVE` means the sample cannot separate "
        "passing from failing, which is different from failing."
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    lines.append("## Configuration")
    lines.append("")

    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Confidence level | {config['confidence_level']:.0%} |"
    )
    lines.append(
        f"| Minimum acceptable score | "
        f"{config['minimum_acceptable_score']} |"
    )
    lines.append(
        f"| Minimum acceptable pass rate | "
        f"{config['minimum_acceptable_pass_rate']:.0%} |"
    )
    lines.append(
        f"| Proportion method | {config['proportion_method']} |"
    )
    lines.append(
        f"| Mean method | {config['mean_method']} |"
    )
    lines.append(
        f"| Difference method | {config['difference_method']} |"
    )
    lines.append(
        f"| Runs per case (Chapter 1) | {repeated['runs_per_case']} |"
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Case 2 - average scores
    # -----------------------------------------------------------------------

    lines.append("## Average Score Intervals")
    lines.append("")

    lines.append(
        "Calculated over the repeated runs Chapter 1 recorded for each "
        "transcript, using:"
    )
    lines.append("")
    lines.append("```")
    lines.append("mean           = sum(scores) / n")
    lines.append("standard_error = sample_standard_deviation / sqrt(n)")
    lines.append("interval       = mean +/- t_multiplier * standard_error")
    lines.append("```")
    lines.append("")

    lines.append(generate_score_table(evidence))
    lines.append("")

    lines.append(
        "The multiplier column shows why 1.96 is not used here. With "
        "five runs the sample has four degrees of freedom and the "
        "correct multiplier is 2.776, which makes the interval wider "
        "and the conclusion more honest."
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Case 1 - pass/fail
    # -----------------------------------------------------------------------

    lines.append("## Pass Rate Intervals")
    lines.append("")

    lines.append(
        "Calculated over the same runs, treating each run as pass or "
        "fail against the score threshold."
    )
    lines.append("")

    lines.append(generate_pass_rate_table(evidence))
    lines.append("")

    lines.append(
        "Both methods are shown deliberately. Where a case passed every "
        "run, the Wald interval collapses to 100% to 100%, claiming "
        "certainty from a handful of runs. The Wilson interval does "
        "not, which is why it drives the verdict."
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Pooled
    # -----------------------------------------------------------------------

    lines.append("## Pooled Across Cases")
    lines.append("")

    lines.append(generate_pooled_table(evidence))
    lines.append("")

    lines.append(
        "Pooling raises n and narrows the interval, but it mixes "
        "transcripts of different difficulty, so a pooled average can "
        "hide a case that is consistently weak. The per-case tables "
        "above remain the basis for the gate."
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Paired comparison
    # -----------------------------------------------------------------------

    lines.append("## Base vs Metamorphic (Paired)")
    lines.append("")

    lines.append(
        "The same cases run through both the original and the "
        "transformed transcript, so the comparison is paired: the "
        "difference is taken per case first, and the interval is built "
        "around those differences."
    )
    lines.append("")

    lines.append(generate_paired_table(evidence))
    lines.append("")

    lines.append(
        "An interval that crosses zero means these data do not clearly "
        "separate the two versions. It is not proof the transformation "
        "was harmless."
    )
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Plain English
    # -----------------------------------------------------------------------

    lines.append("## Plain-English Summary")
    lines.append("")

    lines.append(generate_statements(evidence))
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    lines.append("## Sample Size Notes")
    lines.append("")

    lines.append(generate_warnings(evidence))
    lines.append("")

    lines.append("---")
    lines.append("")

    # -----------------------------------------------------------------------
    # Method
    # -----------------------------------------------------------------------

    lines.append("## Method Notes")
    lines.append("")

    for note in evidence["method_notes"]:
        lines.append(f"- {note}")

    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append("## Scope")
    lines.append("")

    lines.append(
        "These intervals describe the transcripts and runs that were "
        "actually evaluated. They do not describe meeting transcripts "
        "in general, and they assume the judge itself is unbiased, "
        "which this packet does not test."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    evidence = load_evidence()

    report = generate_report(evidence)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Report written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
