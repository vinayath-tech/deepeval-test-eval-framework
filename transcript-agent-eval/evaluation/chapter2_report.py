import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = BASE_DIR / "evidence" / "chapter2"
EVIDENCE_FILE = EVIDENCE_DIR / "release-evidence.json"
REPORT_FILE = EVIDENCE_DIR / "release-evidence.md"


def load_evidence():
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def format_score(score):
    if score is None:
        return "N/A"

    return f"{float(score):.2f}"


def generate_risk_table(evidence):
    rows = [
        "| Risk | Population | Sampled | Sampling Rate |",
        "|---|---:|---:|---:|",
    ]

    for risk, data in evidence["risk_stratification"].items():
        rows.append(
            f"| {risk} | "
            f"{data['population']} | "
            f"{data['sampled']} | "
            f"{data['sampling_rate']:.0%} |"
        )

    return "\n".join(rows)


def generate_base_results_table(evidence):
    rows = [
        "| Case | Risk | Summary | Action Items | Status |",
        "|---|---|---:|---:|---|",
    ]

    for result in evidence["base_results"]:
        status = "PASS" if result["passed"] else "FAIL"

        rows.append(
            f"| {result['case_id']} | "
            f"{result['risk_level']} | "
            f"{format_score(result['summary']['score'])} | "
            f"{format_score(result['action_items']['score'])} | "
            f"{status} |"
        )

    return "\n".join(rows)


def generate_metamorphic_table(evidence):
    rows = [
        "| Case | Transformation | Summary | Action Items | Status |",
        "|---|---|---:|---:|---|",
    ]

    for result in evidence["metamorphic_results"]:
        status = "PASS" if result["passed"] else "FAIL"

        rows.append(
            f"| {result['case_id']} | "
            f"{result['transformation']} | "
            f"{format_score(result['summary']['score'])} | "
            f"{format_score(result['action_items']['score'])} | "
            f"{status} |"
        )

    return "\n".join(rows)


def generate_failure_table(evidence):
    failures = evidence.get("failure_traces", [])

    if not failures:
        return "No failure traces recorded."

    rows = [
        "| Case | Stage | Error Type | Message |",
        "|---|---|---|---|",
    ]

    for failure in failures:
        message = str(
            failure.get("message", "")
        )

        message = message.replace("\n", " ")
        message = message.replace("|", "\\|")

        rows.append(
            f"| {failure.get('case_id', 'N/A')} | "
            f"{failure.get('stage', 'N/A')} | "
            f"{failure.get('error_type', 'N/A')} | "
            f"{message} |"
        )

    return "\n".join(rows)


def generate_sampled_cases(evidence):
    case_ids = evidence.get(
        "sampled_case_ids",
        [],
    )

    if not case_ids:
        return "No cases sampled."

    return "\n".join(
        f"- `{case_id}`"
        for case_id in case_ids
    )


def generate_report(evidence):
    config = evidence["configuration"]
    population = evidence["population"]
    gate = evidence["release_gate"]

    risk_table = generate_risk_table(evidence)
    base_table = generate_base_results_table(evidence)
    metamorphic_table = generate_metamorphic_table(evidence)
    failure_table = generate_failure_table(evidence)
    sampled_cases = generate_sampled_cases(evidence)

    report_lines = []

    report_lines.append("# Chapter 2 Release Evidence")
    report_lines.append("")

    report_lines.append(
        f"Generated: `{evidence['generated_at']}`"
    )
    report_lines.append("")

    report_lines.append("## Release Decision")
    report_lines.append("")
    report_lines.append(
        f"**{gate['decision']}**"
    )
    report_lines.append("")
    report_lines.append(
        gate["reason"]
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Evaluation Scope")
    report_lines.append("")

    report_lines.append(
        "| Property | Value |"
    )
    report_lines.append(
        "|---|---|"
    )

    report_lines.append(
        "| Chapter | Jason Arbon - Testing AI - Chapter 2 |"
    )
    report_lines.append(
        f"| Agent model | `{config['agent_model']}` |"
    )
    report_lines.append(
        f"| Judge model | `{config['judge_model']}` |"
    )
    report_lines.append(
        f"| Temperature | `{config['temperature']}` |"
    )
    report_lines.append(
        f"| Minimum acceptable score | `{config['minimum_acceptable_score']}` |"
    )
    report_lines.append(
        f"| Sampling strategy | `{config['sampling_strategy']}` |"
    )
    report_lines.append(
        f"| Random seed | `{config['random_seed']}` |"
    )

    report_lines.append("")

    report_lines.append(
        "Scores use a 0-10 engineering scale."
    )
    report_lines.append(
        "A score of 7.0 or greater is considered passing."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Evaluation Population")
    report_lines.append("")

    report_lines.append(
        f"- Total cases: **{population['total_cases']}**"
    )
    report_lines.append(
        f"- Sampled cases: **{population['sampled_cases']}**"
    )
    report_lines.append("")

    report_lines.append("### Sampled Cases")
    report_lines.append("")
    report_lines.append(sampled_cases)
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Risk Stratification")
    report_lines.append("")

    report_lines.append(
        "Cases are classified into risk strata before sampling."
    )
    report_lines.append("")

    report_lines.append(risk_table)
    report_lines.append("")

    report_lines.append(
        "This is risk-based stratified sampling."
    )
    report_lines.append(
        "It is not a statistically calculated sample-size determination."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Base Evaluation Results")
    report_lines.append("")

    report_lines.append(base_table)
    report_lines.append("")

    report_lines.append(
        "The base evaluation checks summary quality and action-item quality."
    )
    report_lines.append(
        "Minor wording variation is acceptable."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Metamorphic Testing")
    report_lines.append("")

    report_lines.append(metamorphic_table)
    report_lines.append("")

    report_lines.append("### Transformation")
    report_lines.append("")

    report_lines.append(
        "The metamorphic test adds irrelevant conversational filler "
        "to the original transcript."
    )
    report_lines.append("")

    report_lines.append(
        "Expected relationship:"
    )
    report_lines.append("")

    report_lines.append(
        "Adding irrelevant content should not materially change "
        "important meeting facts or action items."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Failure Traces")
    report_lines.append("")

    report_lines.append(failure_table)
    report_lines.append("")

    report_lines.append(
        "Failure traces capture the case, evaluation stage, "
        "error type, message, and traceback where available."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Release Gate")
    report_lines.append("")

    report_lines.append("### BLOCK")
    report_lines.append("")
    report_lines.append(
        "- A high-risk sampled case fails."
    )
    report_lines.append(
        "- A metamorphic test fails."
    )
    report_lines.append(
        "- An agent or evaluation failure is recorded."
    )
    report_lines.append("")

    report_lines.append("### REVIEW")
    report_lines.append("")
    report_lines.append(
        "- A medium- or low-risk sampled case fails."
    )
    report_lines.append(
        "- No blocking condition exists."
    )
    report_lines.append("")

    report_lines.append("### RELEASE")
    report_lines.append("")
    report_lines.append(
        "- All sampled base evaluations pass."
    )
    report_lines.append(
        "- All metamorphic evaluations pass."
    )
    report_lines.append(
        "- No failure traces exist."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Chapter 3 Boundary")
    report_lines.append("")

    report_lines.append(
        "Chapter 3 statistical analysis is not implemented in this commit."
    )
    report_lines.append("")

    report_lines.append(
        "This evidence does not claim statistical confidence intervals, "
        "statistical significance, population-level confidence, or "
        "statistically sufficient sample size."
    )
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    report_lines.append("## Evidence Files")
    report_lines.append("")

    report_lines.append(
        "- `release-evidence.json`"
    )
    report_lines.append(
        "- `sampled-cases.json`"
    )
    report_lines.append(
        "- `metamorphic-results.json`"
    )
    report_lines.append(
        "- `failure-traces.json`"
    )
    report_lines.append(
        "- `release-evidence.md`"
    )
    report_lines.append("")

    return "\n".join(report_lines)


def main():
    evidence = load_evidence()

    report = generate_report(evidence)

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(
        f"Chapter 2 report written to: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()