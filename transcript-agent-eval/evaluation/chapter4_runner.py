"""
Chapter 4 runner - apply the four statistics to the transcript agent.

    1. run the agent over every transcript, under two models and two
       conditions (original transcript, and the same transcript with
       irrelevant filler appended)
    2. score each run against the human gold answers with F-scores
    3. run the three pre-registered hypotheses from chapter4_stats.HYPOTHESES
    4. write evidence/chapter4/chapter4-evidence.json and .md

    python evaluation/chapter4_runner.py

Environment overrides:

    CHAPTER4_BASELINE_MODEL   default qwen2.5:3b
    CHAPTER4_VARIANT_MODEL    default qwen2.5:0.5b
    CHAPTER4_TEMPERATURE      default 0.0
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AGENT_DIR.parent

for path in (str(PROJECT_ROOT), str(AGENT_DIR), str(AGENT_DIR / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from chapter4_stats import (  # noqa: E402
    HYPOTHESES,
    chi_squared,
    explain,
    f_scores,
    flatten_gold,
    flatten_prediction,
    get_hypothesis,
    paired_flip_test,
    read_result,
    to_labels,
)
from conftest import MeetingSummarizer  # noqa: E402
from metamorphic_tests import append_irrelevant_filler  # noqa: E402

DATASET_DIR = AGENT_DIR / "dataset"
EVIDENCE_DIR = AGENT_DIR / "evidence" / "chapter4"

BASELINE_MODEL = os.getenv("CHAPTER4_BASELINE_MODEL", "qwen2.5:3b")
VARIANT_MODEL = os.getenv("CHAPTER4_VARIANT_MODEL", "qwen2.5:0.5b")
TEMPERATURE = float(os.getenv("CHAPTER4_TEMPERATURE", "0.0"))

MATCH_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Step 1 - run the agent
# ---------------------------------------------------------------------------

def run_agent(model, transcripts, gold):
    """
    Run one model over every transcript, twice: once on the original and once
    with filler appended. Returns one record per transcript per condition.
    """

    agent = MeetingSummarizer(model=model, temperature=TEMPERATURE)
    records = []

    for case_id, transcript in transcripts.items():

        for condition, text in (
            ("base", transcript),
            ("filler", append_irrelevant_filler(transcript)),
        ):

            raw = agent.get_action_items(text)
            predicted = flatten_prediction(raw)

            gold_items = flatten_gold(gold[case_id])

            if predicted is None:
                # Unreadable JSON is not the same failure as a bad
                # extraction, so it is recorded rather than scored as zero.
                record = {
                    "case_id": case_id,
                    "model": model,
                    "condition": condition,
                    "parsed": False,
                    "y_true": [1] * len(gold_items),
                    "y_pred": [0] * len(gold_items),
                }
            else:
                y_true, y_pred = to_labels(
                    predicted, gold_items, threshold=MATCH_THRESHOLD
                )
                record = {
                    "case_id": case_id,
                    "model": model,
                    "condition": condition,
                    "parsed": True,
                    "predicted_items": predicted,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }

            print(
                f"  {model:14} {case_id:22} {condition:7} "
                f"{'ok' if record['parsed'] else 'UNPARSEABLE'}"
            )
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Step 2 - F-scores
# ---------------------------------------------------------------------------

def score_records(records):
    """F-scores per (model, condition), pooling every transcript."""

    scores = {}

    for model in sorted({r["model"] for r in records}):
        for condition in ("base", "filler"):

            rows = [
                r for r in records
                if r["model"] == model and r["condition"] == condition
            ]

            if not rows:
                continue

            y_true = [v for r in rows for v in r["y_true"]]
            y_pred = [v for r in rows for v in r["y_pred"]]

            scores[f"{model} / {condition}"] = {
                "model": model,
                "condition": condition,
                "parsed_runs": sum(1 for r in rows if r["parsed"]),
                "total_runs": len(rows),
                **f_scores(y_true, y_pred),
            }

    return scores


# ---------------------------------------------------------------------------
# Step 3 - the pre-registered hypotheses
# ---------------------------------------------------------------------------

def test_h1_filler(records, baseline_model):
    """Paired: the same items, before and after filler was added."""

    entry = get_hypothesis("h1_filler")

    better = worse = 0

    for case_id in sorted({r["case_id"] for r in records}):

        base = next(
            (r for r in records
             if r["case_id"] == case_id and r["condition"] == "base"
             and r["model"] == baseline_model),
            None,
        )
        filler = next(
            (r for r in records
             if r["case_id"] == case_id and r["condition"] == "filler"
             and r["model"] == baseline_model),
            None,
        )

        if base is None or filler is None:
            continue

        # Per item: was it found before, and was it found after?
        base_hits = sum(1 for t, p in zip(base["y_true"], base["y_pred"])
                        if t == 1 and p == 1)
        filler_hits = sum(1 for t, p in zip(filler["y_true"], filler["y_pred"])
                          if t == 1 and p == 1)

        if filler_hits > base_hits:
            better += filler_hits - base_hits
        elif base_hits > filler_hits:
            worse += base_hits - filler_hits

    result = paired_flip_test(got_better=better, got_worse=worse)
    result["verdict"] = read_result(result, entry["alpha"])
    result["explanation"] = explain(result, entry["alpha"])

    return result


def test_h2_risk(records, risk_metadata, baseline_model):
    """Contingency table: risk level against extraction correctness."""

    entry = get_hypothesis("h2_risk")

    counts = {"high": [0, 0], "other": [0, 0]}

    for record in records:

        if record["model"] != baseline_model or record["condition"] != "base":
            continue

        level = risk_metadata.get(record["case_id"], {}).get("risk_level", "other")
        row = "high" if level == "high" else "other"

        for true_label, pred_label in zip(record["y_true"], record["y_pred"]):
            if true_label == 1:
                counts[row][0 if pred_label == 1 else 1] += 1

    table = [counts["high"], counts["other"]]

    if min(sum(row) for row in table) == 0:
        return {
            "test": "chi_squared",
            "observed": table,
            "verdict": "NOT_TESTED",
            "explanation": "One risk level had no items to score.",
        }

    result = chi_squared(table)
    result["row_labels"] = ["high risk", "other risk"]
    result["column_labels"] = ["extracted", "missed"]
    result["verdict"] = read_result(result, entry["alpha"])
    result["explanation"] = explain(result, entry["alpha"])

    return result


def test_h3_models(records, baseline_model, variant_model):
    """Contingency table: model against extraction correctness."""

    entry = get_hypothesis("h3_models")

    table = []

    for model in (baseline_model, variant_model):

        found = missed = 0

        for record in records:
            if record["model"] != model or record["condition"] != "base":
                continue
            for true_label, pred_label in zip(record["y_true"], record["y_pred"]):
                if true_label == 1:
                    if pred_label == 1:
                        found += 1
                    else:
                        missed += 1

        table.append([found, missed])

    if min(sum(row) for row in table) == 0:
        return {
            "test": "chi_squared",
            "observed": table,
            "verdict": "NOT_TESTED",
            "explanation": "One model produced no scoreable items.",
        }

    result = chi_squared(table)
    result["row_labels"] = [baseline_model, variant_model]
    result["column_labels"] = ["extracted", "missed"]
    result["verdict"] = read_result(result, entry["alpha"])
    result["explanation"] = explain(result, entry["alpha"])

    return result


# ---------------------------------------------------------------------------
# Step 4 - report
# ---------------------------------------------------------------------------

def render_markdown(evidence):
    """Evidence dict -> a readable markdown packet."""

    lines = [
        "# Chapter 4 - Statistical Tests for AI Quality",
        "",
        f"Generated: {evidence['generated_at']}",
        "",
        "Every number here is measured against human-written gold answers in "
        "`dataset/gold_action_items.json`. No model judges anything.",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Baseline model | `{evidence['baseline_model']}` |",
        f"| Variant model | `{evidence['variant_model']}` |",
        f"| Temperature | {evidence['temperature']} |",
        f"| Match threshold | {evidence['match_threshold']} |",
        "",
        "## F-scores",
        "",
        "`beta` decides which mistake the score punishes. For a meeting "
        "assistant a hallucinated action item sends someone to do work nobody "
        "assigned, so **F0.5 is the gate** and the others are context.",
        "",
        "| arm | precision | recall | F0.5 | F1 | F2 | TP | FP | FN | parsed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for name, s in evidence["f_scores"].items():
        lines.append(
            f"| {name} | {s['precision']} | {s['recall']} | **{s['f0.5']}** | "
            f"{s['f1']} | {s['f2']} | {s['true_positives']} | "
            f"{s['false_positives']} | {s['false_negatives']} | "
            f"{s['parsed_runs']}/{s['total_runs']} |"
        )

    lines += ["", "## Pre-registered hypotheses", "",
              "Declared in `chapter4_stats.HYPOTHESES` before the run.", ""]

    for entry in HYPOTHESES:

        result = evidence["hypotheses"].get(entry["id"], {})

        lines += [
            f"### `{entry['id']}` - {entry['question']}",
            "",
            f"- **H0:** {entry['h0']}",
            f"- **Test:** {entry['test']}, alpha = {entry['alpha']}",
            "",
            f"**Verdict: {result.get('verdict', 'NOT_TESTED')}**",
            "",
            result.get("explanation", ""),
            "",
        ]

        if "observed" in result and result.get("row_labels"):
            lines += [
                "| | " + " | ".join(result["column_labels"]) + " |",
                "|---|" + "---|" * len(result["column_labels"]),
            ]
            for label, row in zip(result["row_labels"], result["observed"]):
                lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
            lines.append("")

        elif result.get("test") == "paired_flip_test":
            lines += [
                f"- items the filler fixed: {result.get('got_better')}",
                f"- items the filler broke: {result.get('got_worse')}",
                "",
            ]

    return "\n".join(lines)


def main():

    transcripts = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(DATASET_DIR.glob("*_transcript.txt"))
    }
    gold = json.loads((DATASET_DIR / "gold_action_items.json").read_text("utf-8"))
    risk = json.loads((DATASET_DIR / "risk_metadata.json").read_text("utf-8"))

    print(f"Running {len(transcripts)} transcripts x 2 conditions x 2 models\n")

    records = (
        run_agent(BASELINE_MODEL, transcripts, gold)
        + run_agent(VARIANT_MODEL, transcripts, gold)
    )

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": BASELINE_MODEL,
        "variant_model": VARIANT_MODEL,
        "temperature": TEMPERATURE,
        "match_threshold": MATCH_THRESHOLD,
        "f_scores": score_records(records),
        "hypotheses": {
            "h1_filler": test_h1_filler(records, BASELINE_MODEL),
            "h2_risk": test_h2_risk(records, risk, BASELINE_MODEL),
            "h3_models": test_h3_models(records, BASELINE_MODEL, VARIANT_MODEL),
        },
        "records": records,
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    (EVIDENCE_DIR / "chapter4-evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    (EVIDENCE_DIR / "chapter4-evidence.md").write_text(
        render_markdown(evidence), encoding="utf-8"
    )

    print(f"\nWrote {EVIDENCE_DIR / 'chapter4-evidence.md'}")

    for hypothesis_id, result in evidence["hypotheses"].items():
        print(f"  {hypothesis_id:12} {result.get('verdict')}")


if __name__ == "__main__":
    main()
