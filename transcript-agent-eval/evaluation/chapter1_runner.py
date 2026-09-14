# transcript-agent-eval/evaluation/chapter1_runner.py

import json, sys
from pathlib import Path
import os
import statistics
import uuid
from datetime import datetime, timezone

AGENT_DIR = Path(__file__).resolve().parents[1]   # transcript-agent-eval/ (conftest.py)
PROJECT_ROOT = AGENT_DIR.parent                   # repo root (config.py)

for _path in (PROJECT_ROOT, AGENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from conftest import MeetingSummarizer
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config import (
    SUMMARIZER_AGENT_MODEL,
    SUMMARIZER_JUDGE_MODEL_LOCAL,
)


RUNS_PER_CASE = 5

# Chapter 1: 0/10 rubric
PASS_SCORE = 7.0

EVIDENCE_DIR = "./transcript-agent-eval/evidence"


def load_transcripts():
    documents_path = "./transcript-agent-eval/dataset"

    transcripts = []

    for filename in sorted(os.listdir(documents_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                transcripts.append({
                    "case_id": filename.replace(".txt", ""),
                    "transcript": file.read().strip()
                })

    return transcripts


def create_summary_metric():

    return GEval(
        name="Summary Quality",
        criteria="""
        Evaluate the meeting summary against the transcript.

        Score according to this rubric:

        10 = Excellent.
             Accurate, concise, captures all critical decisions,
             outcomes and important context. No meaningful omissions
             or hallucinations.

        7 = Good.
            Mostly accurate and focused. Minor omissions or verbosity
            are acceptable but the summary remains useful.

        4 = Poor.
            Several important omissions, inaccuracies or unnecessary
            content. The summary is only partially useful.

        0 = Unacceptable.
            Misleading, substantially incorrect, hallucinated,
            or unusable as a meeting summary.

        Give a score from 0 to 10 based on these anchors.
        """,
        threshold=0.7,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=SUMMARIZER_JUDGE_MODEL_LOCAL,
    )


def create_action_item_metric():

    return GEval(
        name="Action Item Quality",
        criteria="""
        Evaluate the action items against the transcript.

        Score according to this rubric:

        10 = Excellent.
             All important action items are captured accurately,
             with correct ownership/context where available.
             No fabricated actions.

        7 = Good.
            Most important actions are captured accurately.
            Minor omissions or ambiguity are acceptable.

        4 = Poor.
            Several important actions are missing, incorrect,
            or unclear.

        0 = Unacceptable.
            Action items are substantially incorrect, fabricated,
            or unusable.

        Give a score from 0 to 10 based on these anchors.
        """,
        threshold=0.7,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=SUMMARIZER_JUDGE_MODEL_LOCAL,
    )


def evaluate_case(
    case_id,
    transcript,
    summarizer,
    summary_metric,
    action_metric,
):
    """
    Execute one transcript multiple times.

    This is the key Chapter 1 change:
        one transcript -> 5 independent agent runs
    """

    runs = []

    for run_number in range(1, RUNS_PER_CASE + 1):

        print(
            f"Running {case_id}: "
            f"run {run_number}/{RUNS_PER_CASE}"
        )

        summary, action_items = summarizer.process(transcript)

        summary_test_case = LLMTestCase(
            input=transcript,
            actual_output=summary,
        )

        action_test_case = LLMTestCase(
            input=transcript,
            actual_output=action_items,
        )

        # measure() returns the score itself (a float), normalised
        # to 0-1 from GEval's default 0-10 range.
        summary_score = summary_metric.measure(
            summary_test_case
        ) * 10

        action_score = action_metric.measure(
            action_test_case
        ) * 10

        runs.append({
            "run_number": run_number,

            "summary": {
                "score": round(summary_score, 2),
                "passed": summary_score >= PASS_SCORE,
                "output": summary,
            },

            "action_items": {
                "score": round(action_score, 2),
                "passed": action_score >= PASS_SCORE,
                "output": action_items,
            },
        })

    return runs


def calculate_statistics(scores):

    if not scores:
        return {}

    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 2),
        "median": round(statistics.median(scores), 2),
        "min": round(min(scores), 2),
        "max": round(max(scores), 2),
        "std_dev": round(
            statistics.stdev(scores), 2
        ) if len(scores) > 1 else 0.0,
        "pass_rate": round(
            sum(score >= PASS_SCORE for score in scores)
            / len(scores),
            4,
        ),
    }


def build_evidence():

    transcripts = load_transcripts()

    summarizer = MeetingSummarizer(
        model=SUMMARIZER_AGENT_MODEL,
        temperature=0.5,
    )

    summary_metric = create_summary_metric()
    action_metric = create_action_item_metric()

    evidence = {
        "release_run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "method": {
            "chapter": "Jason Arbon Chapter 1",
            "runs_per_case": RUNS_PER_CASE,
            "pass_score": PASS_SCORE,
            "score_range": "0-10",
        },

        "agent": {
            "model": SUMMARIZER_AGENT_MODEL,
            "temperature": 0.5,
        },

        "judge": {
            "model": str(
                SUMMARIZER_JUDGE_MODEL_LOCAL
            ),
        },

        "cases": [],
    }

    for transcript in transcripts:

        runs = evaluate_case(
            transcript["case_id"],
            transcript["transcript"],
            summarizer,
            summary_metric,
            action_metric,
        )

        summary_scores = [
            run["summary"]["score"]
            for run in runs
        ]

        action_scores = [
            run["action_items"]["score"]
            for run in runs
        ]

        case_result = {
            "case_id": transcript["case_id"],

            "summary_statistics":
                calculate_statistics(summary_scores),

            "action_item_statistics":
                calculate_statistics(action_scores),

            "runs": runs,
        }

        evidence["cases"].append(case_result)

    return evidence

def print_variance_table(evidence):

        print()
        print("=" * 90)
        print(
            f"{'Case':<20}"
            f"{'Metric':<20}"
            f"{'Mean':<10}"
            f"{'Min':<10}"
            f"{'Max':<10}"
            f"{'Std Dev':<10}"
            f"{'Pass %':<10}"
        )
        print("=" * 90)

        for case in evidence["cases"]:

            summary = case["summary_statistics"]

            print(
                f"{case['case_id']:<20}"
                f"{'Summary':<20}"
                f"{summary['mean']:<10}"
                f"{summary['min']:<10}"
                f"{summary['max']:<10}"
                f"{summary['std_dev']:<10}"
                f"{summary['pass_rate'] * 100:<10.1f}"
            )

            action = case["action_item_statistics"]

            print(
                f"{case['case_id']:<20}"
                f"{'Action Items':<20}"
                f"{action['mean']:<10}"
                f"{action['min']:<10}"
                f"{action['max']:<10}"
                f"{action['std_dev']:<10}"
                f"{action['pass_rate'] * 100:<10.1f}"
            )

        print("=" * 90)


def main():

    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    evidence = build_evidence()

    print_variance_table(evidence)

    output_file = os.path.join(
        EVIDENCE_DIR,
        "chapter1-evidence.json",
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evidence,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("CHAPTER 1 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Cases evaluated : {len(evidence['cases'])}")
    print(f"Runs per case   : {RUNS_PER_CASE}")
    print(f"Evidence file   : {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()