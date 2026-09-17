"""
Chapter 4 - Statistical tests for AI quality.

Four concepts, one section each. Every test is a thin wrapper around scipy or
scikit-learn - the library does the maths, this file only decides what to feed
it and how to read the answer.

    Topic 1  Null hypothesis    a plain list of what we assume before running
    Topic 2  F-scores           sklearn.metrics
    Topic 3  Chi-squared        scipy.stats.chi2_contingency
    Topic 4  P-values           reading the number the tests return

Run `python chapter4_stats.py` to see each topic demonstrated on the book's
own worked examples.
"""

import json
import re
from difflib import SequenceMatcher

from scipy.stats import binomtest, chi2_contingency
from sklearn.metrics import fbeta_score, precision_recall_fscore_support


def _p(value):
    """Round a p-value without flattening a very small one to zero."""

    return float(f"{float(value):.6g}")


# ===========================================================================
# Topic 1 - Null hypothesis
# ===========================================================================
#
# The null hypothesis is the boring assumption: nothing changed. The eval's
# job is to collect enough evidence to reject it.
#
# The only thing that makes this more than a comment is *when* you write it.
# Declared before the run, it is a prediction. Declared after, you have simply
# described whichever slice happened to look interesting - with enough slices,
# something always does.
#
# So it lives here as plain data, written before the run.

HYPOTHESES = [
    {
        "id": "h1_filler",
        "question": "Does irrelevant filler text change what the agent extracts?",
        "h0": "Extraction correctness is the same before and after filler is added.",
        "test": "paired_flip_test",
        "alpha": 0.05,
    },
    {
        "id": "h2_risk",
        "question": "Does a transcript's risk level predict extraction failure?",
        "h0": "Correct and incorrect extractions appear in the same proportion "
              "at every risk level.",
        "test": "chi_squared",
        "alpha": 0.05,
    },
    {
        "id": "h3_models",
        "question": "Do two models fail at the same rate?",
        "h0": "Both models extract correctly in the same proportion.",
        "test": "chi_squared",
        "alpha": 0.05,
    },
]


def get_hypothesis(hypothesis_id):
    """Look up a pre-registered hypothesis, or refuse."""

    for entry in HYPOTHESES:
        if entry["id"] == hypothesis_id:
            return entry

    raise KeyError(
        f"'{hypothesis_id}' was not pre-registered in HYPOTHESES. "
        f"Add it before the run, not after."
    )


# ===========================================================================
# Topic 2 - F-scores (precision, recall, F1)
# ===========================================================================
#
# precision = of everything the agent extracted, how much was real?
# recall    = of everything it should have found, how much did it catch?
# F1        = the harmonic mean, which punishes lopsidedness
#
# sklearn computes all of these. The work on our side is turning two lists of
# free-text action items into the 0/1 label vectors sklearn expects.


def normalise(text):
    """Lowercase, strip punctuation, collapse whitespace."""

    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def similarity(a, b):
    """0.0 to 1.0. Action items are free text, so 'same item' is fuzzy."""

    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def to_labels(predicted, gold, threshold=0.6):
    """
    Turn two lists of action items into (y_true, y_pred) for sklearn.

    Each row is one item:

        matched prediction  -> y_true=1, y_pred=1   true positive
        unmatched prediction-> y_true=0, y_pred=1   false positive
        unmatched gold item -> y_true=1, y_pred=0   false negative

    Greedy matching: each gold item can only be claimed once, so an agent
    that repeats the same item three times scores one hit and two misses.
    """

    y_true, y_pred = [], []
    remaining = list(gold)

    for item in predicted:

        best, best_score = None, threshold

        for candidate in remaining:
            score = similarity(item, candidate)
            if score >= best_score:
                best, best_score = candidate, score

        if best is not None:
            remaining.remove(best)
            y_true.append(1)
            y_pred.append(1)
        else:
            y_true.append(0)
            y_pred.append(1)

    for _ in remaining:
        y_true.append(1)
        y_pred.append(0)

    return y_true, y_pred


def f_scores(y_true, y_pred):
    """
    Precision, recall and three F-scores.

    beta decides which mistake the score cares about, and the book is clear
    that you pick it before the run, not after seeing which one flatters you:

        F0.5  false alarms hurt more   (a hallucinated action item sends
                                        someone to do work nobody assigned)
        F1    both hurt equally
        F2    misses hurt more         (a dropped commitment is forgotten)
    """

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f0.5": round(float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)), 4),
        "f1": round(float(f1), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)), 4),
        "true_positives": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),
        "false_positives": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
        "false_negatives": sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
    }


# ===========================================================================
# Topic 3 - Chi-squared
# ===========================================================================
#
# Chi-squared is for counts, not averages. "Was this extracted correctly?" is
# yes/no, so the data is a contingency table:
#
#                   correct   incorrect
#       high risk       5          1
#       low risk        8          6
#
# The test compares that against the table you would expect if risk level and
# correctness were unrelated, and reports how surprising the gap is.


def chi_squared(table):
    """
    scipy.stats.chi2_contingency on a 2D list of counts.

    One guard worth keeping: chi-squared assumes every *expected* cell count
    is at least 5. Below that the p-value is not approximately right, it is
    wrong, so we say so rather than printing a number we do not believe.
    """

    chi2, p_value, dof, expected = chi2_contingency(table)

    total = sum(sum(row) for row in table)
    smaller_side = min(len(table), len(table[0])) - 1

    return {
        "test": "chi_squared",
        "chi2": round(float(chi2), 4),
        "p_value": _p(p_value),
        "degrees_of_freedom": int(dof),
        "observed": table,
        "expected": [[round(float(v), 2) for v in row] for row in expected],
        # Cramer's V is the effect size: how strong the association is,
        # on a 0-1 scale, independent of sample size.
        "cramers_v": round(float((chi2 / (total * smaller_side)) ** 0.5), 4)
        if total and smaller_side else 0.0,
        "expected_counts_ok": bool(expected.min() >= 5),
    }


def paired_flip_test(got_better, got_worse):
    """
    The paired version, for before/after on the *same* items.

    When the same transcripts run through two variants, the rows are not
    independent - the same item appears twice. Items that behaved the same
    way both times carry no information, so this looks only at the ones that
    flipped, and asks whether the split between them is worse than a coin.
    """

    changed = got_better + got_worse

    if changed == 0:
        return {
            "test": "paired_flip_test",
            "p_value": 1.0,
            "got_better": 0,
            "got_worse": 0,
            "note": "Nothing changed, so there is nothing to test.",
        }

    result = binomtest(got_better, changed, 0.5)

    return {
        "test": "paired_flip_test",
        "p_value": _p(result.pvalue),
        "got_better": got_better,
        "got_worse": got_worse,
        "changed": changed,
        # The smallest p-value this many flips could ever produce. If it is
        # above alpha, "not significant" says nothing about the agent - the
        # sample was too small for any result to count.
        "smallest_possible_p": _p(binomtest(changed, changed, 0.5).pvalue),
    }


# ===========================================================================
# Topic 4 - P-values
# ===========================================================================
#
# A p-value answers one narrow question: if the null hypothesis were true,
# how often would we see a gap at least this big by chance alone?
#
# It does not say the difference is large, that users will notice, or that
# the change is good. p = 0.03 with a tiny effect means "a real but pointless
# difference, measured carefully". So a verdict needs both numbers.


def verdict(p_value, alpha=0.05, effect=None, practical_threshold=0.1):
    """Read a p-value the way the book asks: evidence, not permission."""

    if p_value is None:
        return "NOT_TESTED"

    if p_value >= alpha:
        return "NOT_SIGNIFICANT"

    if effect is not None and effect < practical_threshold:
        return "SIGNIFICANT_BUT_TOO_SMALL_TO_ACT_ON"

    return "SIGNIFICANT"


def read_result(result, alpha=0.05, practical_threshold=0.1):
    """
    The verdict for a whole test result, guards included.

    `verdict` above reads a bare p-value. This reads the dict a test returns,
    so it can catch the two cases where the p-value should not be read at all:
    expected counts too small for chi-squared to mean anything, and a sample
    so small that no result could have cleared alpha.
    """

    if not result.get("expected_counts_ok", True):
        return "NOT_TESTED"

    floor = result.get("smallest_possible_p")

    if floor is not None and floor > alpha:
        # Not the same as NOT_SIGNIFICANT. Nothing could have been.
        return "UNDERPOWERED"

    return verdict(
        result.get("p_value"),
        alpha,
        result.get("cramers_v"),
        practical_threshold,
    )


def explain(result, alpha=0.05):
    """One sentence a non-statistician can read."""

    if not result.get("expected_counts_ok", True):
        return (
            "Not tested: some expected cell counts are below 5, where the "
            "chi-squared p-value stops being trustworthy. Collect more data "
            "or merge categories."
        )

    floor = result.get("smallest_possible_p")
    if floor is not None and floor > alpha:
        return (
            f"p = {result['p_value']}, but with only {result['changed']} "
            f"changed items the smallest p-value possible was {floor}, which "
            f"is already above alpha = {alpha}. No result could have been "
            f"significant, so this is not evidence of no effect."
        )

    if result["p_value"] < alpha:
        return (
            f"p = {result['p_value']} is below alpha = {alpha}. A gap this "
            f"large would be unusual if the null hypothesis were true."
        )

    return (
        f"p = {result['p_value']} is above alpha = {alpha}. This sample does "
        f"not contradict the null hypothesis."
    )


# ===========================================================================
# Loading gold answers
# ===========================================================================


def flatten_gold(gold_entry):
    """Gold JSON -> one flat list of action item strings."""

    items = []

    for tasks in gold_entry.get("individual_actions", {}).values():
        items.extend(tasks)

    items.extend(gold_entry.get("team_actions", []))

    return items


def flatten_prediction(raw):
    """Agent output (a JSON string) -> one flat list of action item strings.

    Returns None when the agent did not produce readable JSON, which is a
    different failure from extracting badly and should be counted separately.
    """

    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    if not isinstance(raw, dict):
        return None

    items = []

    individual = raw.get("individual_actions") or {}
    if isinstance(individual, dict):
        for tasks in individual.values():
            if isinstance(tasks, list):
                items.extend(str(t) for t in tasks)

    team = raw.get("team_actions") or []
    if isinstance(team, list):
        items.extend(str(t) for t in team)

    return items


# ===========================================================================
# Demo - the book's own worked examples
# ===========================================================================

if __name__ == "__main__":

    print("Topic 2 - F-scores (book: 40 TP, 10 FP, 20 FN)")
    y_true = [1] * 40 + [0] * 10 + [1] * 20
    y_pred = [1] * 40 + [1] * 10 + [0] * 20
    for key, value in f_scores(y_true, y_pred).items():
        print(f"    {key:16} {value}")
    print("    book says precision 0.80, recall 0.67, F1 0.73, "
          "F0.5 0.77, F2 0.69\n")

    print("Topic 3 - Chi-squared (book: CartCare old vs new policy)")
    table = [[70, 20, 10], [55, 35, 10]]
    result = chi_squared(table)
    for key in ("chi2", "p_value", "degrees_of_freedom", "cramers_v",
                "expected_counts_ok"):
        print(f"    {key:20} {result[key]}")
    print("    book says chi2 = 5.89, p = about 0.053\n")

    print("Topic 3 - Paired flip test (book: 15 new wins, 5 old wins)")
    paired = paired_flip_test(got_better=15, got_worse=5)
    for key, value in paired.items():
        print(f"    {key:22} {value}")
    print("    book says two-sided p = about 0.041\n")

    print("Topic 4 - Reading it")
    print("    verdict:", verdict(paired["p_value"], alpha=0.05))
    print("   ", explain(paired))
