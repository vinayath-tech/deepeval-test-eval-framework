"""
Jason Arbon Testing AI - Chapter 2

Release Evidence Runner

Implements:

1. Risk metadata
2. Stratified sampling
3. Base evaluation
4. Metamorphic testing
5. Failure traces
6. Release decision
7. Machine-readable release evidence

Chapter 3 statistical confidence intervals are intentionally
not implemented here.
"""

import json
import os
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]   # transcript-agent-eval/ (conftest.py)
PROJECT_ROOT = AGENT_DIR.parent                   # repo root (config.py)

for _path in (PROJECT_ROOT, AGENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from conftest import MeetingSummarizer
from metamorphic_tests import build_metamorphic_case


from config import (
    SUMMARIZER_AGENT_MODEL,
    SUMMARIZER_JUDGE_MODEL_OPENAI,
)



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset"
EVIDENCE_DIR = BASE_DIR / "evidence" / "chapter2"

RISK_METADATA_FILE = DATASET_DIR / "risk_metadata.json"

MODEL = os.getenv(
    "SUMMARIZER_AGENT_MODEL",
    "qwen2.5:3b",
)

TEMPERATURE = float(
    os.getenv(
        "SUMMARIZER_AGENT_TEMPERATURE",
        "0.5",
    )
)

JUDGE_MODEL = os.getenv(
    "DEEPEVAL_MODEL",
    MODEL,
)

MIN_ACCEPTABLE_SCORE = 7.0

RANDOM_SEED = 42

# Chapter 2 is intentionally risk-based.
# It is NOT a Chapter 3 statistically sufficient sample-size calculation.
SAMPLE_RATE_BY_RISK = {
    "high": 1.0,
    "medium": 1.0,
    "low": 1.0,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_evidence_directory() -> None:
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_risk_metadata() -> Dict[str, Any]:

    with open(
        RISK_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_transcripts() -> List[Dict[str, Any]]:

    transcripts = []

    for transcript_file in sorted(DATASET_DIR.glob("*.txt")):

        case_id = transcript_file.stem

        with open(
            transcript_file,
            "r",
            encoding="utf-8",
        ) as file:
            transcript = file.read()

        transcripts.append(
            {
                "case_id": case_id,
                "file": transcript_file.name,
                "input": transcript,
            }
        )

    return transcripts


def add_risk_metadata(
    transcripts: List[Dict[str, Any]],
    risk_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:

    enriched = []

    for case in transcripts:

        case_id = case["case_id"]

        metadata = risk_metadata.get(case_id)

        if metadata is None:
            raise ValueError(
                f"No risk metadata found for transcript: {case_id}"
            )

        enriched.append(
            {
                **case,
                "risk": metadata,
            }
        )

    return enriched


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Divide the population into risk strata and sample from each stratum.

    At least one case is selected from every non-empty risk stratum.

    Sampling is deterministic using RANDOM_SEED so CI evidence can
    be reproduced.
    """

    rng = random.Random(RANDOM_SEED)

    strata: Dict[str, List[Dict[str, Any]]] = {}

    for case in cases:

        risk_level = case["risk"]["risk_level"]

        strata.setdefault(
            risk_level,
            [],
        ).append(case)

    sampled_cases = []
    strata_report = {}

    for risk_level in sorted(strata.keys()):

        population = sorted(
            strata[risk_level],
            key=lambda item: item["case_id"],
        )

        population_size = len(population)

        configured_rate = SAMPLE_RATE_BY_RISK.get(
            risk_level,
            1.0,
        )

        sample_size = max(
            1,
            int(round(population_size * configured_rate)),
        )

        sample_size = min(
            sample_size,
            population_size,
        )

        # Use deterministic ordering before sampling.
        candidates = population.copy()
        rng.shuffle(candidates)

        selected = sorted(
            candidates[:sample_size],
            key=lambda item: item["case_id"],
        )

        sampled_cases.extend(selected)

        strata_report[risk_level] = {
            "population": population_size,
            "sampled": len(selected),
            "sampling_rate": (
                len(selected) / population_size
                if population_size
                else 0
            ),
            "case_ids": [
                case["case_id"]
                for case in selected
            ],
        }

    return {
        "sampled_cases": sorted(
            sampled_cases,
            key=lambda item: item["case_id"],
        ),
        "strata": strata_report,
    }


# ---------------------------------------------------------------------------
# Failure tracing
# ---------------------------------------------------------------------------

def create_failure_trace(
    case_id: str,
    stage: str,
    exception: Exception | None = None,
    run_number: int | None = None,
    message: str | None = None,
) -> Dict[str, Any]:

    trace = {
        "timestamp": utc_timestamp(),
        "case_id": case_id,
        "stage": stage,
        "run_number": run_number,
    }

    if exception is not None:

        trace.update(
            {
                "error_type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
        )

    else:

        trace.update(
            {
                "error_type": "AgentReturnedError",
                "message": message or "Unknown agent error",
                "traceback": None,
            }
        )

    return trace


def is_agent_error(output: Any) -> bool:

    if not isinstance(output, str):
        return False

    return output.startswith(
        (
            "Could not generate summary:",
            "Could not generate action items:",
        )
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def create_summary_metric() -> GEval:

    return GEval(
        name="Summary Quality",
        criteria=(
            "Evaluate whether the generated meeting summary accurately "
            "captures the important facts, decisions, discussion points, "
            "and outcomes from the transcript. Minor wording differences "
            "are acceptable. Penalize missing important information, "
            "incorrect facts, or invented information."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=SUMMARIZER_JUDGE_MODEL_OPENAI
    )


def create_action_item_metric() -> GEval:

    return GEval(
        name="Action Item Quality",
        criteria=(
            "Evaluate whether the generated action items accurately "
            "identify explicit tasks, owners, and relevant details from "
            "the transcript. Minor formatting differences are acceptable. "
            "Penalize missing explicit action items, incorrect ownership, "
            "or invented actions."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=SUMMARIZER_JUDGE_MODEL_OPENAI,
    )


def create_metamorphic_summary_metric() -> GEval:

    return GEval(
        name="Metamorphic Summary Preservation",
        criteria=(
            "The transformed transcript contains additional irrelevant "
            "conversational filler. Evaluate whether the generated "
            "summary preserves the important meeting facts from the "
            "original summary and does not introduce facts caused by "
            "the irrelevant filler. Minor wording differences are "
            "acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=SUMMARIZER_JUDGE_MODEL_OPENAI,
    )


def create_metamorphic_action_metric() -> GEval:

    return GEval(
        name="Metamorphic Action Item Preservation",
        criteria=(
            "The transformed transcript contains additional irrelevant "
            "conversational filler. Evaluate whether the generated "
            "action items preserve the explicit action items from the "
            "original output and do not introduce new actions because "
            "of the irrelevant filler. Minor formatting differences "
            "are acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=SUMMARIZER_JUDGE_MODEL_OPENAI,
    )


def measure_metric(
    metric: GEval,
    test_case: LLMTestCase,
) -> float:

    metric.measure(test_case)

    # DeepEval returns a normalized score from 0 to 1.
    # Convert to Chapter 1's 0-10 engineering score.
    return round(
        float(metric.score) * 10,
        2,
    )


# ---------------------------------------------------------------------------
# Base evaluation
# ---------------------------------------------------------------------------

def evaluate_base_case(
    case: Dict[str, Any],
    agent: MeetingSummarizer,
    failure_traces: List[Dict[str, Any]],
) -> Dict[str, Any]:

    case_id = case["case_id"]

    summary_metric = create_summary_metric()
    action_metric = create_action_item_metric()

    result = {
        "case_id": case_id,
        "risk_level": case["risk"]["risk_level"],
        "summary": {},
        "action_items": {},
    }

    # -----------------------------------------------------------------------
    # Agent execution
    # -----------------------------------------------------------------------

    try:

        summary = agent.get_summary(
            case["input"]
        )

    except Exception as exc:

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="agent_summary",
                exception=exc,
            )
        )

        summary = None

    if is_agent_error(summary):

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="agent_summary",
                message=summary,
            )
        )

    try:

        action_items = agent.get_action_items(
            case["input"]
        )

    except Exception as exc:

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="agent_action_items",
                exception=exc,
            )
        )

        action_items = None

    if is_agent_error(action_items):

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="agent_action_items",
                message=action_items,
            )
        )

    # -----------------------------------------------------------------------
    # Summary evaluation
    # -----------------------------------------------------------------------

    if summary is not None:

        try:

            test_case = LLMTestCase(
                input=case["input"],
                actual_output=summary,
            )

            score = measure_metric(
                summary_metric,
                test_case,
            )

            result["summary"] = {
                "score": score,
                "passed": score >= MIN_ACCEPTABLE_SCORE,
                "output": summary,
            }

        except Exception as exc:

            failure_traces.append(
                create_failure_trace(
                    case_id=case_id,
                    stage="judge_summary",
                    exception=exc,
                )
            )

            result["summary"] = {
                "score": None,
                "passed": False,
                "output": summary,
            }

    else:

        result["summary"] = {
            "score": None,
            "passed": False,
            "output": None,
        }

    # -----------------------------------------------------------------------
    # Action item evaluation
    # -----------------------------------------------------------------------

    if action_items is not None:

        try:

            test_case = LLMTestCase(
                input=case["input"],
                actual_output=action_items,
            )

            score = measure_metric(
                action_metric,
                test_case,
            )

            result["action_items"] = {
                "score": score,
                "passed": score >= MIN_ACCEPTABLE_SCORE,
                "output": action_items,
            }

        except Exception as exc:

            failure_traces.append(
                create_failure_trace(
                    case_id=case_id,
                    stage="judge_action_items",
                    exception=exc,
                )
            )

            result["action_items"] = {
                "score": None,
                "passed": False,
                "output": action_items,
            }

    else:

        result["action_items"] = {
            "score": None,
            "passed": False,
            "output": None,
        }

    result["passed"] = (
        result["summary"]["passed"]
        and result["action_items"]["passed"]
    )

    return result


# ---------------------------------------------------------------------------
# Metamorphic evaluation
# ---------------------------------------------------------------------------

def evaluate_metamorphic_case(
    case: Dict[str, Any],
    original_result: Dict[str, Any],
    agent: MeetingSummarizer,
    failure_traces: List[Dict[str, Any]],
) -> Dict[str, Any]:

    case_id = case["case_id"]

    metamorphic_case = build_metamorphic_case(
        case_id=case_id,
        transcript=case["input"],
    )

    summary_metric = create_metamorphic_summary_metric()
    action_metric = create_metamorphic_action_metric()

    result = {
        "case_id": case_id,
        "transformation": metamorphic_case.transformation_name,
        "original_input": metamorphic_case.original_input,
        "transformed_input": metamorphic_case.transformed_input,
        "summary": {},
        "action_items": {},
    }

    # -----------------------------------------------------------------------
    # Generate transformed output
    # -----------------------------------------------------------------------

    try:

        transformed_summary = agent.get_summary(
            metamorphic_case.transformed_input
        )

    except Exception as exc:

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="metamorphic_agent_summary",
                exception=exc,
            )
        )

        transformed_summary = None

    if is_agent_error(transformed_summary):

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="metamorphic_agent_summary",
                message=transformed_summary,
            )
        )

    try:

        transformed_action_items = agent.get_action_items(
            metamorphic_case.transformed_input
        )

    except Exception as exc:

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="metamorphic_agent_action_items",
                exception=exc,
            )
        )

        transformed_action_items = None

    if is_agent_error(transformed_action_items):

        failure_traces.append(
            create_failure_trace(
                case_id=case_id,
                stage="metamorphic_agent_action_items",
                message=transformed_action_items,
            )
        )

    # -----------------------------------------------------------------------
    # Summary metamorphic evaluation
    # -----------------------------------------------------------------------

    if transformed_summary is not None:

        try:

            test_case = LLMTestCase(
                input=metamorphic_case.transformed_input,
                actual_output=transformed_summary,
                expected_output=original_result["summary"]["output"],
            )

            score = measure_metric(
                summary_metric,
                test_case,
            )

            result["summary"] = {
                "score": score,
                "passed": score >= MIN_ACCEPTABLE_SCORE,
                "output": transformed_summary,
            }

        except Exception as exc:

            failure_traces.append(
                create_failure_trace(
                    case_id=case_id,
                    stage="metamorphic_judge_summary",
                    exception=exc,
                )
            )

            result["summary"] = {
                "score": None,
                "passed": False,
                "output": transformed_summary,
            }

    else:

        result["summary"] = {
            "score": None,
            "passed": False,
            "output": None,
        }

    # -----------------------------------------------------------------------
    # Action item metamorphic evaluation
    # -----------------------------------------------------------------------

    if transformed_action_items is not None:

        try:

            test_case = LLMTestCase(
                input=metamorphic_case.transformed_input,
                actual_output=transformed_action_items,
                expected_output=original_result["action_items"]["output"],
            )

            score = measure_metric(
                action_metric,
                test_case,
            )

            result["action_items"] = {
                "score": score,
                "passed": score >= MIN_ACCEPTABLE_SCORE,
                "output": transformed_action_items,
            }

        except Exception as exc:

            failure_traces.append(
                create_failure_trace(
                    case_id=case_id,
                    stage="metamorphic_judge_action_items",
                    exception=exc,
                )
            )

            result["action_items"] = {
                "score": None,
                "passed": False,
                "output": transformed_action_items,
            }

    else:

        result["action_items"] = {
            "score": None,
            "passed": False,
            "output": None,
        }

    result["passed"] = (
        result["summary"]["passed"]
        and result["action_items"]["passed"]
    )

    return result


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------

def determine_release_decision(
    base_results: List[Dict[str, Any]],
    metamorphic_results: List[Dict[str, Any]],
    failure_traces: List[Dict[str, Any]],
) -> Dict[str, Any]:

    high_risk_failures = []

    other_risk_failures = []

    for result in base_results:

        if result["passed"]:
            continue

        if result["risk_level"] == "high":

            high_risk_failures.append(
                result["case_id"]
            )

        else:

            other_risk_failures.append(
                result["case_id"]
            )

    metamorphic_failures = [
        result["case_id"]
        for result in metamorphic_results
        if not result["passed"]
    ]

    # -----------------------------------------------------------------------
    # BLOCK
    # -----------------------------------------------------------------------

    if high_risk_failures:

        return {
            "decision": "BLOCK",
            "reason": (
                "A high-risk sampled case failed the base evaluation."
            ),
            "high_risk_failures": high_risk_failures,
            "other_risk_failures": other_risk_failures,
            "metamorphic_failures": metamorphic_failures,
            "failure_trace_count": len(failure_traces),
        }

    if metamorphic_failures:

        return {
            "decision": "BLOCK",
            "reason": (
                "A metamorphic test failed, indicating that an "
                "irrelevant input change affected important output."
            ),
            "high_risk_failures": high_risk_failures,
            "other_risk_failures": other_risk_failures,
            "metamorphic_failures": metamorphic_failures,
            "failure_trace_count": len(failure_traces),
        }

    if failure_traces:

        return {
            "decision": "BLOCK",
            "reason": (
                "One or more execution or evaluation failures "
                "were recorded in the failure traces."
            ),
            "high_risk_failures": high_risk_failures,
            "other_risk_failures": other_risk_failures,
            "metamorphic_failures": metamorphic_failures,
            "failure_trace_count": len(failure_traces),
        }

    # -----------------------------------------------------------------------
    # REVIEW
    # -----------------------------------------------------------------------

    if other_risk_failures:

        return {
            "decision": "REVIEW",
            "reason": (
                "A non-high-risk sampled case failed the base "
                "evaluation and requires engineering review."
            ),
            "high_risk_failures": high_risk_failures,
            "other_risk_failures": other_risk_failures,
            "metamorphic_failures": metamorphic_failures,
            "failure_trace_count": len(failure_traces),
        }

    # -----------------------------------------------------------------------
    # RELEASE
    # -----------------------------------------------------------------------

    return {
        "decision": "RELEASE",
        "reason": (
            "All sampled base evaluations and metamorphic evaluations "
            "passed with no recorded execution failures."
        ),
        "high_risk_failures": [],
        "other_risk_failures": [],
        "metamorphic_failures": [],
        "failure_trace_count": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ensure_evidence_directory()

    print("=" * 80)
    print("Chapter 2 - Release Evidence Runner")
    print("=" * 80)

    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Evidence directory: {EVIDENCE_DIR}")
    print(f"Agent model       : {MODEL}")
    print(f"Temperature       : {TEMPERATURE}")
    print(f"Judge model       : {JUDGE_MODEL}")

    # -----------------------------------------------------------------------
    # Load population
    # -----------------------------------------------------------------------

    risk_metadata = load_risk_metadata()

    transcripts = load_transcripts()

    cases = add_risk_metadata(
        transcripts,
        risk_metadata,
    )

    print(f"\nPopulation size: {len(cases)}")

    # -----------------------------------------------------------------------
    # Stratified sampling
    # -----------------------------------------------------------------------

    sampling = stratified_sample(cases)

    sampled_cases = sampling["sampled_cases"]

    print("\nRisk strata:")

    for risk, data in sampling["strata"].items():

        print(
            f"  {risk}: "
            f"{data['sampled']}/{data['population']} sampled"
        )

    print("\nSampled cases:")

    for case in sampled_cases:

        print(
            f"  {case['case_id']} "
            f"({case['risk']['risk_level']})"
        )

    # Save sampled cases separately.

    with open(
        EVIDENCE_DIR / "sampled-cases.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "population_size": len(cases),
                "sampled_size": len(sampled_cases),
                "strata": sampling["strata"],
                "sampled_case_ids": [
                    case["case_id"]
                    for case in sampled_cases
                ],
            },
            file,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Agent
    # -----------------------------------------------------------------------

    agent = MeetingSummarizer(
        model=MODEL,
        temperature=TEMPERATURE,
    )

    failure_traces = []

    base_results = []
    metamorphic_results = []

    # -----------------------------------------------------------------------
    # Evaluate sampled population
    # -----------------------------------------------------------------------

    for case in sampled_cases:

        print(
            f"\nEvaluating {case['case_id']} "
            f"[risk={case['risk']['risk_level']}]"
        )

        base_result = evaluate_base_case(
            case=case,
            agent=agent,
            failure_traces=failure_traces,
        )

        base_results.append(base_result)

        print(
            f"  Summary score      : "
            f"{base_result['summary']['score']}"
        )

        print(
            f"  Action item score  : "
            f"{base_result['action_items']['score']}"
        )

        print(
            f"  Base result        : "
            f"{'PASS' if base_result['passed'] else 'FAIL'}"
        )

        # ---------------------------------------------------------------
        # Metamorphic evaluation
        # ---------------------------------------------------------------

        print("  Running metamorphic test...")

        metamorphic_result = evaluate_metamorphic_case(
            case=case,
            original_result=base_result,
            agent=agent,
            failure_traces=failure_traces,
        )

        metamorphic_results.append(
            metamorphic_result
        )

        print(
            f"  Metamorphic summary: "
            f"{metamorphic_result['summary']['score']}"
        )

        print(
            f"  Metamorphic actions : "
            f"{metamorphic_result['action_items']['score']}"
        )

        print(
            f"  Metamorphic result  : "
            f"{'PASS' if metamorphic_result['passed'] else 'FAIL'}"
        )

    # -----------------------------------------------------------------------
    # Release gate
    # -----------------------------------------------------------------------

    release_decision = determine_release_decision(
        base_results=base_results,
        metamorphic_results=metamorphic_results,
        failure_traces=failure_traces,
    )

    print("\n" + "=" * 80)
    print(
        f"RELEASE DECISION: {release_decision['decision']}"
    )
    print("=" * 80)

    print(
        release_decision["reason"]
    )

    # -----------------------------------------------------------------------
    # Evidence packet
    # -----------------------------------------------------------------------

    evidence = {
        "chapter": "Jason Arbon - Testing AI - Chapter 2",
        "generated_at": utc_timestamp(),

        "configuration": {
            "agent_model": MODEL,
            "judge_model": JUDGE_MODEL,
            "temperature": TEMPERATURE,
            "minimum_acceptable_score": MIN_ACCEPTABLE_SCORE,
            "random_seed": RANDOM_SEED,
            "sampling_strategy": "risk_stratified",
        },

        "git": {
            "commit": os.getenv(
                "GITHUB_SHA"
            ),
            "ref": os.getenv(
                "GITHUB_REF"
            ),
        },

        "population": {
            "total_cases": len(cases),
            "sampled_cases": len(sampled_cases),
        },

        "risk_stratification": sampling["strata"],

        "sampled_case_ids": [
            case["case_id"]
            for case in sampled_cases
        ],

        "base_results": base_results,

        "metamorphic_results": metamorphic_results,

        "failure_traces": failure_traces,

        "release_gate": release_decision,

        "chapter3_status": (
            "No confidence intervals or statistical significance "
            "claims are made by this evidence packet. Chapter 3 "
            "calculates intervals over this packet and over the "
            "Chapter 1 repeated runs; see "
            "evidence/chapter3/confidence-evidence.json."
        ),
    }

    with open(
        EVIDENCE_DIR / "release-evidence.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evidence,
            file,
            indent=2,
        )

    with open(
        EVIDENCE_DIR / "metamorphic-results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metamorphic_results,
            file,
            indent=2,
        )

    with open(
        EVIDENCE_DIR / "failure-traces.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            failure_traces,
            file,
            indent=2,
        )

    print(
        f"\nEvidence written to: {EVIDENCE_DIR}"
    )


if __name__ == "__main__":
    main()