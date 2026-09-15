"""
Jason Arbon Testing AI - Chapter 3

Confidence Interval Runner

Chapter 2 deliberately stopped at a point estimate and recorded that no
statistical claims were being made. This runner adds that layer.

It does not call the agent or the judge again. Chapter 1 already ran the
agent five times per transcript and preserved every individual run score,
and Chapter 2 preserved the risk-stratified base and metamorphic scores.
Those are the samples. Chapter 3 is the analysis over them, so it is
cheap, deterministic, and re-runnable without spending judge tokens.

Produces:

1. Per-case score intervals        (Case 2 - average 0-10 scores)
2. Per-case pass-rate intervals    (Case 1 - pass/fail results)
3. Pooled intervals across cases
4. Paired base vs metamorphic comparison
5. A confidence-aware release gate
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

AGENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGENT_DIR.parent

for _path in (PROJECT_ROOT, AGENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from confidence_intervals import (
    compare_to_threshold,
    describe_difference,
    describe_mean,
    describe_proportion,
    mean_interval,
    paired_difference_interval,
    proportion_interval,
    runs_needed_for_margin,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset"
EVIDENCE_DIR = BASE_DIR / "evidence"

CHAPTER1_EVIDENCE_FILE = EVIDENCE_DIR / "chapter1" / "chapter1-evidence.json"
CHAPTER2_EVIDENCE_FILE = EVIDENCE_DIR / "chapter2" / "release-evidence.json"

CHAPTER3_EVIDENCE_DIR = EVIDENCE_DIR / "chapter3"
CHAPTER3_EVIDENCE_FILE = CHAPTER3_EVIDENCE_DIR / "confidence-evidence.json"

RISK_METADATA_FILE = DATASET_DIR / "risk_metadata.json"

CONFIDENCE_LEVEL = float(
    os.getenv(
        "CONFIDENCE_LEVEL",
        "0.95",
    )
)

# Kept identical to Chapter 1 and Chapter 2 so the three packets are
# comparable. These are policy choices, not statistical results.
MIN_ACCEPTABLE_SCORE = 7.0
MIN_ACCEPTABLE_PASS_RATE = 0.80

# Used only to answer "how many more runs would settle this?".
TARGET_SCORE_MARGIN = 0.5

METRICS = ("summary", "action_items")

METRIC_LABELS = {
    "summary": "summary quality score",
    "action_items": "action item quality score",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(
    path: Path,
) -> Optional[Dict[str, Any]]:

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_risk_levels() -> Dict[str, str]:
    """
    Case id to risk level, reusing the Chapter 2 risk metadata so the
    gate can hold high-risk cases to a stricter evidence standard.
    """

    metadata = load_json(RISK_METADATA_FILE) or {}

    return {
        case_id: entry.get("risk_level", "unknown")
        for case_id, entry in metadata.items()
    }


# ---------------------------------------------------------------------------
# Chapter 1 analysis - repeated runs per case
# ---------------------------------------------------------------------------

def analyse_repeated_runs(
    chapter1_evidence: Dict[str, Any],
    risk_levels: Dict[str, str],
) -> Dict[str, Any]:
    """
    Chapter 1 runs each transcript NUM_RUNS times. That repetition is
    what makes an interval possible: it measures how much the agent and
    judge wobble on the same input.
    """

    cases: List[Dict[str, Any]] = []

    pooled_scores: Dict[str, List[float]] = {
        metric: []
        for metric in METRICS
    }

    pooled_passes: Dict[str, int] = {
        metric: 0
        for metric in METRICS
    }

    pooled_totals: Dict[str, int] = {
        metric: 0
        for metric in METRICS
    }

    for case in chapter1_evidence.get("cases", []):

        case_id = case["case_id"]

        case_report: Dict[str, Any] = {
            "case_id": case_id,
            "risk_level": risk_levels.get(case_id, "unknown"),
            "metrics": {},
        }

        for metric in METRICS:

            scores = [
                run[metric]["score"]
                for run in case["runs"]
                if run[metric]["score"] is not None
            ]

            passes = sum(
                1
                for run in case["runs"]
                if run[metric]["passed"]
            )

            total = len(case["runs"])

            if not scores:
                continue

            pooled_scores[metric].extend(scores)
            pooled_passes[metric] += passes
            pooled_totals[metric] += total

            case_report["metrics"][metric] = build_metric_report(
                scores=scores,
                passes=passes,
                total=total,
                metric=metric,
            )

        cases.append(case_report)

    pooled: Dict[str, Any] = {}

    for metric in METRICS:

        if not pooled_scores[metric]:
            continue

        pooled[metric] = build_metric_report(
            scores=pooled_scores[metric],
            passes=pooled_passes[metric],
            total=pooled_totals[metric],
            metric=metric,
        )

    return {
        "source": str(
            CHAPTER1_EVIDENCE_FILE.relative_to(BASE_DIR)
        ),
        "runs_per_case": chapter1_evidence.get(
            "configuration",
            {},
        ).get("num_runs"),
        "cases": cases,
        "pooled": pooled,
    }


def build_metric_report(
    scores: List[float],
    passes: int,
    total: int,
    metric: str,
) -> Dict[str, Any]:
    """
    Both chapter cases applied to the same set of runs:

    Case 2 on the raw 0-10 scores, and Case 1 on the pass/fail flags
    derived from them. They answer different questions, so both are
    reported rather than picking one.
    """

    label = METRIC_LABELS[metric]

    score_interval = mean_interval(
        scores,
        confidence=CONFIDENCE_LEVEL,
    )

    score_verdict = compare_to_threshold(
        score_interval,
        threshold=MIN_ACCEPTABLE_SCORE,
    )

    # Wald is reported alongside Wilson to make the small-sample
    # difference visible, but Wilson drives the verdict.
    pass_rate_interval = proportion_interval(
        passes=passes,
        total=total,
        confidence=CONFIDENCE_LEVEL,
        method="wilson",
    )

    pass_rate_wald = proportion_interval(
        passes=passes,
        total=total,
        confidence=CONFIDENCE_LEVEL,
        method="wald",
    )

    pass_rate_verdict = compare_to_threshold(
        pass_rate_interval,
        threshold=MIN_ACCEPTABLE_PASS_RATE,
    )

    report = {
        "scores": scores,
        "score_interval": score_interval,
        "score_verdict": score_verdict,
        "score_statement": describe_mean(
            score_interval,
            label=label,
        ),
        "pass_rate_interval": pass_rate_interval,
        "pass_rate_interval_wald": pass_rate_wald,
        "pass_rate_verdict": pass_rate_verdict,
        "pass_rate_statement": describe_proportion(
            pass_rate_interval,
            label=f"{label} runs",
        ),
    }

    if score_verdict["verdict"] == "INCONCLUSIVE":

        report["runs_needed_for_target_margin"] = {
            "target_margin": TARGET_SCORE_MARGIN,
            "estimated_runs": runs_needed_for_margin(
                sample_standard_deviation=score_interval[
                    "sample_standard_deviation"
                ],
                target_margin=TARGET_SCORE_MARGIN,
                confidence=CONFIDENCE_LEVEL,
            ),
        }

    return report


# ---------------------------------------------------------------------------
# Chapter 2 analysis - stratified sample and metamorphic pairs
# ---------------------------------------------------------------------------

def analyse_release_sample(
    chapter2_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Chapter 2 scored each sampled case once, so there is no within-case
    variability to measure. The interval that can be calculated is
    across cases, and with two sampled cases it is extremely wide.

    Reporting that width is the point. It is the honest answer to
    "should we ship on this evidence?".
    """

    base_results = chapter2_evidence.get("base_results", [])
    metamorphic_results = chapter2_evidence.get("metamorphic_results", [])

    across_cases: Dict[str, Any] = {}

    for metric in METRICS:

        scores = [
            result[metric]["score"]
            for result in base_results
            if result[metric]["score"] is not None
        ]

        if not scores:
            continue

        interval = mean_interval(
            scores,
            confidence=CONFIDENCE_LEVEL,
        )

        across_cases[metric] = {
            "scores": scores,
            "score_interval": interval,
            "score_verdict": compare_to_threshold(
                interval,
                threshold=MIN_ACCEPTABLE_SCORE,
            ),
            "score_statement": describe_mean(
                interval,
                label=METRIC_LABELS[metric],
            ),
        }

    # -----------------------------------------------------------------------
    # Base vs metamorphic, paired by case
    # -----------------------------------------------------------------------

    metamorphic_by_case = {
        result["case_id"]: result
        for result in metamorphic_results
    }

    paired: Dict[str, Any] = {}

    for metric in METRICS:

        before: List[float] = []
        after: List[float] = []
        case_ids: List[str] = []

        for result in base_results:

            case_id = result["case_id"]

            transformed = metamorphic_by_case.get(case_id)

            if transformed is None:
                continue

            base_score = result[metric]["score"]
            transformed_score = transformed[metric]["score"]

            if base_score is None or transformed_score is None:
                continue

            before.append(base_score)
            after.append(transformed_score)
            case_ids.append(case_id)

        if not before:
            continue

        interval = paired_difference_interval(
            scores_before=before,
            scores_after=after,
            confidence=CONFIDENCE_LEVEL,
        )

        paired[metric] = {
            "case_ids": case_ids,
            "base_scores": before,
            "metamorphic_scores": after,
            "difference_interval": interval,
            "statement": describe_difference(
                interval,
                label=METRIC_LABELS[metric],
            ),
        }

    return {
        "source": str(
            CHAPTER2_EVIDENCE_FILE.relative_to(BASE_DIR)
        ),
        "sampled_cases": chapter2_evidence.get(
            "population",
            {},
        ).get("sampled_cases"),
        "across_cases": across_cases,
        "paired_metamorphic": paired,
        "note": (
            "Chapter 2 scores each sampled case once. These intervals "
            "describe variation across cases, not run-to-run stability."
        ),
    }


# ---------------------------------------------------------------------------
# Confidence-aware release gate
# ---------------------------------------------------------------------------

def determine_confidence_gate(
    repeated_runs: Dict[str, Any],
    release_sample: Dict[str, Any],
) -> Dict[str, Any]:
    """
    A gate on interval bounds rather than point estimates.

    Rules:

    - A high-risk case must have its entire interval at or above the
      threshold. An interval that straddles the threshold blocks,
      because a wide interval is not evidence of quality.
    - A non-high-risk case that is below or inconclusive goes to review.
    - A metamorphic difference interval lying entirely below zero is a
      measured regression and blocks.
    """

    blocking: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []

    for case in repeated_runs.get("cases", []):

        risk_level = case["risk_level"]

        for metric, report in case["metrics"].items():

            for verdict_key in ("score_verdict", "pass_rate_verdict"):

                verdict = report[verdict_key]["verdict"]

                if verdict == "MEETS_THRESHOLD":
                    continue

                finding = {
                    "case_id": case["case_id"],
                    "risk_level": risk_level,
                    "metric": metric,
                    "measure": verdict_key.replace("_verdict", ""),
                    "verdict": verdict,
                    "reason": report[verdict_key]["reason"],
                }

                if risk_level == "high":
                    blocking.append(finding)
                else:
                    review.append(finding)

    metamorphic_regressions: List[Dict[str, Any]] = []
    metamorphic_inconclusive: List[Dict[str, Any]] = []

    for metric, report in release_sample.get(
        "paired_metamorphic",
        {},
    ).items():

        interval = report["difference_interval"]

        if interval["lower"] is None:

            metamorphic_inconclusive.append(
                {
                    "metric": metric,
                    "reason": (
                        "Too few pairs to calculate a difference "
                        "interval."
                    ),
                }
            )

            continue

        if interval["upper"] < 0:

            metamorphic_regressions.append(
                {
                    "metric": metric,
                    "difference": interval["estimate"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                    "reason": (
                        "The entire difference interval is below zero. "
                        "Irrelevant filler measurably degraded output."
                    ),
                }
            )

        elif interval["crosses_zero"]:

            metamorphic_inconclusive.append(
                {
                    "metric": metric,
                    "difference": interval["estimate"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                    "reason": (
                        "The difference interval crosses zero. This "
                        "sample cannot show whether the transformation "
                        "changed behaviour."
                    ),
                }
            )

    gate = {
        "confidence_level": CONFIDENCE_LEVEL,
        "minimum_acceptable_score": MIN_ACCEPTABLE_SCORE,
        "minimum_acceptable_pass_rate": MIN_ACCEPTABLE_PASS_RATE,
        "blocking_findings": blocking,
        "review_findings": review,
        "metamorphic_regressions": metamorphic_regressions,
        "metamorphic_inconclusive": metamorphic_inconclusive,
    }

    if blocking:

        gate.update(
            {
                "decision": "BLOCK",
                "reason": (
                    "A high-risk case did not demonstrate, at the "
                    f"{CONFIDENCE_LEVEL:.0%} confidence level, that it "
                    "meets the release threshold."
                ),
            }
        )

        return gate

    if metamorphic_regressions:

        gate.update(
            {
                "decision": "BLOCK",
                "reason": (
                    "A metamorphic difference interval lies entirely "
                    "below zero, which is a measured regression rather "
                    "than noise."
                ),
            }
        )

        return gate

    if review or metamorphic_inconclusive:

        gate.update(
            {
                "decision": "REVIEW",
                "reason": (
                    "Some measures are below threshold or too "
                    "uncertain to decide at the "
                    f"{CONFIDENCE_LEVEL:.0%} confidence level."
                ),
            }
        )

        return gate

    gate.update(
        {
            "decision": "RELEASE",
            "reason": (
                "Every measured interval lies at or above its "
                f"threshold at the {CONFIDENCE_LEVEL:.0%} confidence "
                "level."
            ),
        }
    )

    return gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    CHAPTER3_EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Chapter 3 - Confidence Interval Runner")
    print("=" * 80)

    print(f"Confidence level  : {CONFIDENCE_LEVEL:.0%}")
    print(f"Score threshold   : {MIN_ACCEPTABLE_SCORE}")
    print(f"Pass rate threshold: {MIN_ACCEPTABLE_PASS_RATE:.0%}")

    chapter1_evidence = load_json(CHAPTER1_EVIDENCE_FILE)
    chapter2_evidence = load_json(CHAPTER2_EVIDENCE_FILE)

    if chapter1_evidence is None:
        raise FileNotFoundError(
            f"Chapter 1 evidence not found at {CHAPTER1_EVIDENCE_FILE}. "
            f"Run chapter1_runner.py first."
        )

    risk_levels = load_risk_levels()

    repeated_runs = analyse_repeated_runs(
        chapter1_evidence,
        risk_levels,
    )

    # -----------------------------------------------------------------------
    # Repeated run intervals
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("Repeated run intervals (Chapter 1 sample)")
    print("-" * 80)

    for case in repeated_runs["cases"]:

        print(
            f"\n{case['case_id']} [risk={case['risk_level']}]"
        )

        for metric, report in case["metrics"].items():

            print(f"  {metric}:")
            print(f"    {report['score_statement']}")
            print(
                f"    verdict: "
                f"{report['score_verdict']['verdict']}"
            )
            print(f"    {report['pass_rate_statement']}")
            print(
                f"    verdict: "
                f"{report['pass_rate_verdict']['verdict']}"
            )

            needed = report.get("runs_needed_for_target_margin")

            if needed and needed["estimated_runs"]:
                print(
                    f"    ~{needed['estimated_runs']} runs would narrow "
                    f"the margin to +/-{needed['target_margin']}"
                )

    print("\n" + "-" * 80)
    print("Pooled across cases")
    print("-" * 80)

    for metric, report in repeated_runs["pooled"].items():

        print(f"\n  {metric}:")
        print(f"    {report['score_statement']}")
        print(f"    {report['pass_rate_statement']}")

        wald = report["pass_rate_interval_wald"]

        print(
            f"    (wald for comparison: "
            f"{wald['lower']:.0%} to {wald['upper']:.0%})"
        )

    # -----------------------------------------------------------------------
    # Release sample intervals
    # -----------------------------------------------------------------------

    release_sample: Dict[str, Any] = {}

    if chapter2_evidence is not None:

        release_sample = analyse_release_sample(chapter2_evidence)

        print("\n" + "-" * 80)
        print("Release sample intervals (Chapter 2 sample)")
        print("-" * 80)

        for metric, report in release_sample["across_cases"].items():
            print(f"\n  {metric}:")
            print(f"    {report['score_statement']}")

        print("\n  Base vs metamorphic (paired):")

        for metric, report in release_sample[
            "paired_metamorphic"
        ].items():
            print(f"\n  {metric}:")
            print(f"    {report['statement']}")

    else:

        print(
            "\nChapter 2 evidence not found. Skipping release sample "
            "analysis."
        )

    # -----------------------------------------------------------------------
    # Gate
    # -----------------------------------------------------------------------

    gate = determine_confidence_gate(
        repeated_runs=repeated_runs,
        release_sample=release_sample,
    )

    print("\n" + "=" * 80)
    print(f"CONFIDENCE GATE: {gate['decision']}")
    print("=" * 80)
    print(gate["reason"])

    for finding in gate["blocking_findings"]:
        print(
            f"  BLOCK   {finding['case_id']} / {finding['metric']} / "
            f"{finding['measure']}: {finding['verdict']}"
        )

    for finding in gate["review_findings"]:
        print(
            f"  REVIEW  {finding['case_id']} / {finding['metric']} / "
            f"{finding['measure']}: {finding['verdict']}"
        )

    # -----------------------------------------------------------------------
    # Evidence packet
    # -----------------------------------------------------------------------

    evidence = {
        "chapter": "Jason Arbon - Testing AI - Chapter 3",
        "generated_at": utc_timestamp(),

        "configuration": {
            "confidence_level": CONFIDENCE_LEVEL,
            "minimum_acceptable_score": MIN_ACCEPTABLE_SCORE,
            "minimum_acceptable_pass_rate": MIN_ACCEPTABLE_PASS_RATE,
            "target_score_margin": TARGET_SCORE_MARGIN,
            "proportion_method": "wilson",
            "mean_method": "t",
            "difference_method": "paired_t",
        },

        "method_notes": [
            "Intervals follow the pattern estimate +/- multiplier * "
            "standard error.",
            "The t/normal multipliers and the interval calculations "
            "themselves are computed by scipy.stats (ttest_1samp, "
            "ttest_ind, binomtest), not by a hand-rolled formula or a "
            "lookup table.",
            "Score intervals use the t multiplier because the samples "
            "are small; 1.96 would be too narrow.",
            "Pass rate intervals use the Wilson method "
            "(scipy.stats.binomtest), which behaves correctly near 0% "
            "and 100% where the simple Wald formula collapses to zero "
            "width.",
            "Base and metamorphic results are compared as paired "
            "differences per case, not as two marginal intervals.",
            "No claim is made that these intervals are wide enough "
            "samples for a high-risk release. Where they are not, the "
            "verdict is INCONCLUSIVE rather than PASS.",
        ],

        "git": {
            "commit": os.getenv("GITHUB_SHA"),
            "ref": os.getenv("GITHUB_REF"),
        },

        "repeated_run_analysis": repeated_runs,
        "release_sample_analysis": release_sample,
        "confidence_gate": gate,
    }

    with open(
        CHAPTER3_EVIDENCE_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evidence,
            file,
            indent=2,
        )

    print(
        f"\nEvidence written to: {CHAPTER3_EVIDENCE_FILE}"
    )


if __name__ == "__main__":
    main()
