import json
import os
import statistics
from datetime import datetime, timezone

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]   # transcript-agent-eval/ (conftest.py)
PROJECT_ROOT = AGENT_DIR.parent                   # repo root (config.py)

for _path in (PROJECT_ROOT, AGENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from conftest import MeetingSummarizer
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from config import (
    SUMMARIZER_AGENT_MODEL,
    SUMMARIZER_JUDGE_MODEL_OPENAI,
)


# Chapter 1 configuration
NUM_RUNS = 5
PASS_SCORE = 7.0

EVIDENCE_DIR = "./transcript-agent-eval/evidence"
EVIDENCE_FILE = os.path.join(EVIDENCE_DIR, "chapter1-evidence.json")


class Chapter1Runner:

    def transcript_loader(self) -> list[dict]:
        documents_path = "./transcript-agent-eval/dataset"

        transcripts = []

        for document in sorted(os.listdir(documents_path)):
            if document.endswith(".txt"):
                file_path = os.path.join(documents_path, document)

                with open(file_path, "r", encoding="utf-8") as file:
                    transcript = file.read().strip()

                transcripts.append(
                    {
                        "case_id": os.path.splitext(document)[0],
                        "transcript": transcript,
                    }
                )

        return transcripts

    def summary_metric(self) -> GEval:
        return GEval(
            name="Summary Quality",
            criteria=(
                "Assess whether the summary is accurate, focused and contains "
                "the essential points of the meeting without introducing "
                "unsupported information."
            ),
            threshold=0.5,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=SUMMARIZER_JUDGE_MODEL_OPENAI,
        )

    def action_item_metric(self) -> GEval:
        return GEval(
            name="Action Item Quality",
            criteria=(
                "Assess whether the action items are accurate, complete and "
                "clearly reflect the key tasks mentioned in the meeting. "
                "Do not reward fabricated action items."
            ),
            threshold=0.5,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=SUMMARIZER_JUDGE_MODEL_OPENAI,
        )

    def calculate_statistics(self, scores: list[float]) -> dict:
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
                sum(score >= PASS_SCORE for score in scores) / len(scores),
                2,
            ),
        }

    def run(self):
        os.makedirs(EVIDENCE_DIR, exist_ok=True)

        transcripts = self.transcript_loader()

        print(f"Found {len(transcripts)} transcript(s)")
        print(f"Running {NUM_RUNS} evaluation runs per transcript")

        evidence = {
            "run_generated_at": datetime.now(timezone.utc).isoformat(),
            "chapter": "Jason Arbon Chapter 1",
            "configuration": {
                "num_runs": NUM_RUNS,
                "pass_score": PASS_SCORE,
                "agent_model": SUMMARIZER_AGENT_MODEL,
                "judge_model": SUMMARIZER_JUDGE_MODEL_OPENAI,
                "temperature": 0.5,
                "score_scale": "0-10",
            },
            "cases": [],
        }

        for case in transcripts:

            case_id = case["case_id"]
            transcript = case["transcript"]

            print("\n" + "=" * 80)
            print(f"Case: {case_id}")
            print("=" * 80)

            summary_scores = []
            action_item_scores = []

            case_evidence = {
                "case_id": case_id,
                "runs": [],
            }

            for run_number in range(1, NUM_RUNS + 1):

                print(f"\nRunning {case_id} - Run {run_number}/{NUM_RUNS}")

                summarizer = MeetingSummarizer(
                    model=SUMMARIZER_AGENT_MODEL,
                    temperature=0.5,
                )

                summary, action_items = summarizer.process(transcript)

                # ---------------------------------------------------------
                # Summary evaluation
                # ---------------------------------------------------------

                summary_test_case = LLMTestCase(
                    input=transcript,
                    actual_output=summary,
                )

                summary_metric = self.summary_metric()
                summary_metric.measure(summary_test_case)

                summary_score = round(
                    float(summary_metric.score) * 10,
                    2,
                )

                # ---------------------------------------------------------
                # Action item evaluation
                # ---------------------------------------------------------

                action_item_test_case = LLMTestCase(
                    input=transcript,
                    actual_output=action_items,
                )

                action_metric = self.action_item_metric()
                action_metric.measure(action_item_test_case)

                action_item_score = round(
                    float(action_metric.score) * 10,
                    2,
                )

                summary_pass = summary_score >= PASS_SCORE
                action_item_pass = action_item_score >= PASS_SCORE

                summary_scores.append(summary_score)
                action_item_scores.append(action_item_score)

                # ---------------------------------------------------------
                # Preserve individual run scores
                # ---------------------------------------------------------

                run_evidence = {
                    "run_number": run_number,
                    "summary": {
                        "score": summary_score,
                        "passed": summary_pass,
                    },
                    "action_items": {
                        "score": action_item_score,
                        "passed": action_item_pass,
                    },
                }

                case_evidence["runs"].append(run_evidence)

                print(
                    f"  Summary Quality: {summary_score}/10 "
                    f"({'PASS' if summary_pass else 'FAIL'})"
                )

                print(
                    f"  Action Item Quality: {action_item_score}/10 "
                    f"({'PASS' if action_item_pass else 'FAIL'})"
                )

            # -------------------------------------------------------------
            # Case-level statistics
            # -------------------------------------------------------------

            case_evidence["statistics"] = {
                "summary": self.calculate_statistics(summary_scores),
                "action_items": self.calculate_statistics(action_item_scores),
            }

            evidence["cases"].append(case_evidence)

            print("\nCase statistics:")

            print(
                f"  Summary: "
                f"mean={case_evidence['statistics']['summary']['mean']}, "
                f"std_dev={case_evidence['statistics']['summary']['std_dev']}, "
                f"pass_rate={case_evidence['statistics']['summary']['pass_rate']}"
            )

            print(
                f"  Action Items: "
                f"mean={case_evidence['statistics']['action_items']['mean']}, "
                f"std_dev={case_evidence['statistics']['action_items']['std_dev']}, "
                f"pass_rate={case_evidence['statistics']['action_items']['pass_rate']}"
            )

        # -----------------------------------------------------------------
        # Save evidence
        # -----------------------------------------------------------------

        with open(EVIDENCE_FILE, "w", encoding="utf-8") as file:
            json.dump(evidence, file, indent=2)

        print("\n" + "=" * 80)
        print("Chapter 1 evaluation complete")
        print(f"Evidence written to: {EVIDENCE_FILE}")
        print("=" * 80)


if __name__ == "__main__":
    Chapter1Runner().run()