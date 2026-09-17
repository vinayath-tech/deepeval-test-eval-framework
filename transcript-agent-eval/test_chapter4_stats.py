"""
Tests for chapter4_stats.

The book prints its own worked answers, so most of these assert that our
wrappers reproduce them. If a test here fails, the wrapper is feeding scipy
or sklearn the wrong thing.
"""

import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent

for path in (str(AGENT_DIR.parent), str(AGENT_DIR), str(AGENT_DIR / "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from chapter4_stats import (  # noqa: E402
    chi_squared,
    explain,
    f_scores,
    flatten_gold,
    flatten_prediction,
    get_hypothesis,
    paired_flip_test,
    to_labels,
    verdict,
)


# ---------------------------------------------------------------------------
# Topic 1 - null hypothesis
# ---------------------------------------------------------------------------

class TestHypotheses:

    def test_registered_hypothesis_is_found(self):
        entry = get_hypothesis("h1_filler")
        assert entry["h0"]
        assert entry["alpha"] == 0.05

    def test_unregistered_hypothesis_is_refused(self):
        """A comparison invented after the run should not be runnable."""

        with pytest.raises(KeyError, match="not pre-registered"):
            get_hypothesis("h9_something_that_looked_interesting")


# ---------------------------------------------------------------------------
# Topic 2 - F-scores
# ---------------------------------------------------------------------------

class TestFScores:

    def test_book_bugpilot_example(self):
        """40 true positives, 10 false positives, 20 false negatives."""

        y_true = [1] * 40 + [0] * 10 + [1] * 20
        y_pred = [1] * 40 + [1] * 10 + [0] * 20

        scores = f_scores(y_true, y_pred)

        assert scores["precision"] == pytest.approx(0.80, abs=0.01)
        assert scores["recall"] == pytest.approx(0.67, abs=0.01)
        assert scores["f0.5"] == pytest.approx(0.77, abs=0.01)
        assert scores["f1"] == pytest.approx(0.73, abs=0.01)
        assert scores["f2"] == pytest.approx(0.69, abs=0.01)

    def test_beta_orders_the_scores(self):
        """With precision above recall, F0.5 > F1 > F2 - the book's point."""

        y_true = [1] * 40 + [0] * 10 + [1] * 20
        y_pred = [1] * 40 + [1] * 10 + [0] * 20

        scores = f_scores(y_true, y_pred)

        assert scores["f0.5"] > scores["f1"] > scores["f2"]

    def test_f1_punishes_lopsidedness(self):
        """One correct extraction and nothing else is not 50% good."""

        y_true = [1] * 10
        y_pred = [1] + [0] * 9

        scores = f_scores(y_true, y_pred)

        assert scores["precision"] == 1.0
        assert scores["recall"] == pytest.approx(0.1)
        assert scores["f1"] < 0.2


class TestMatching:

    def test_paraphrase_counts_as_a_hit(self):
        y_true, y_pred = to_labels(
            ["send the report to Sarah"], ["send the report to Sarah"]
        )
        assert y_true == [1] and y_pred == [1]

    def test_unrelated_item_is_a_false_positive(self):
        y_true, y_pred = to_labels(["buy a new coffee machine"], ["ship the release"])

        # one false positive and one false negative
        assert sorted(zip(y_true, y_pred)) == [(0, 1), (1, 0)]

    def test_each_gold_item_can_only_be_claimed_once(self):
        """Repeating the same item three times is one hit and two misses."""

        gold = ["ship the release", "write the doc"]
        predicted = ["ship the release", "ship the release", "ship the release"]

        scores = f_scores(*to_labels(predicted, gold))

        assert scores["true_positives"] == 1
        assert scores["false_positives"] == 2
        assert scores["false_negatives"] == 1


# ---------------------------------------------------------------------------
# Topic 3 - chi-squared
# ---------------------------------------------------------------------------

class TestChiSquared:

    def test_book_cartcare_example(self):
        """Old policy 70/20/10 vs new policy 55/35/10."""

        result = chi_squared([[70, 20, 10], [55, 35, 10]])

        assert result["chi2"] == pytest.approx(5.89, abs=0.01)
        assert result["p_value"] == pytest.approx(0.053, abs=0.001)
        assert result["degrees_of_freedom"] == 2

    def test_identical_rows_are_not_surprising(self):
        result = chi_squared([[50, 50], [50, 50]])

        assert result["p_value"] == pytest.approx(1.0)
        assert result["cramers_v"] == pytest.approx(0.0, abs=0.01)

    def test_small_expected_counts_are_flagged(self):
        """The guard the book asks for: below 5 expected, the p is not real."""

        result = chi_squared([[3, 1], [1, 2]])

        assert result["expected_counts_ok"] is False
        assert "below 5" in explain(result)

    def test_large_sample_is_flagged_as_ok(self):
        result = chi_squared([[70, 20, 10], [55, 35, 10]])

        assert result["expected_counts_ok"] is True


class TestPairedFlipTest:

    def test_book_preference_example(self):
        """New version preferred 15 times, old version 5 times."""

        result = paired_flip_test(got_better=15, got_worse=5)

        assert result["p_value"] == pytest.approx(0.041, abs=0.001)

    def test_nothing_changed(self):
        result = paired_flip_test(got_better=0, got_worse=0)

        assert result["p_value"] == 1.0
        assert "nothing to test" in result["note"]

    def test_small_sample_reports_its_floor(self):
        """With 5 flips the smallest possible p is 0.0625 - above alpha."""

        result = paired_flip_test(got_better=5, got_worse=0)

        assert result["smallest_possible_p"] == pytest.approx(0.0625, abs=0.0001)


# ---------------------------------------------------------------------------
# Topic 4 - p-values
# ---------------------------------------------------------------------------

class TestVerdicts:

    def test_above_alpha_is_not_significant(self):
        assert verdict(0.20, alpha=0.05) == "NOT_SIGNIFICANT"

    def test_below_alpha_is_significant(self):
        assert verdict(0.01, alpha=0.05, effect=0.4) == "SIGNIFICANT"

    def test_significant_but_tiny_effect_is_called_out(self):
        """The book's warning: a big sample makes trivial gaps significant."""

        assert verdict(0.001, alpha=0.05, effect=0.02, practical_threshold=0.1) == (
            "SIGNIFICANT_BUT_TOO_SMALL_TO_ACT_ON"
        )

    def test_underpowered_sample_says_so(self):
        """'Not significant' is misleading when nothing could have been."""

        result = paired_flip_test(got_better=5, got_worse=0)

        assert "No result could have been significant" in explain(result)


# ---------------------------------------------------------------------------
# Loading agent output
# ---------------------------------------------------------------------------

class TestLoading:

    def test_flatten_gold_merges_individual_and_team(self):
        gold = {
            "individual_actions": {"Maya": ["build the metric"], "Ethan": ["sync"]},
            "team_actions": ["regroup next week"],
        }

        assert sorted(flatten_gold(gold)) == [
            "build the metric",
            "regroup next week",
            "sync",
        ]

    def test_flatten_prediction_reads_json_string(self):
        raw = '{"individual_actions": {"Maya": ["build the metric"]}, "team_actions": []}'

        assert flatten_prediction(raw) == ["build the metric"]

    def test_flatten_prediction_strips_code_fences(self):
        raw = '```json\n{"individual_actions": {}, "team_actions": ["regroup"]}\n```'

        assert flatten_prediction(raw) == ["regroup"]

    def test_unparseable_output_is_none_not_empty(self):
        """An unreadable answer is a different failure from an empty one."""

        assert flatten_prediction("Sure! Here are the action items:") is None
        assert flatten_prediction('{"individual_actions": {"Maya": [') is None

    def test_real_gold_file_loads(self):
        import json

        gold = json.loads(
            (AGENT_DIR / "dataset" / "gold_action_items.json").read_text("utf-8")
        )

        items = flatten_gold(gold["team_call_transcript"])

        assert len(items) > 0
        assert all(isinstance(item, str) for item in items)


class TestReadResult:
    """read_result is the verdict that knows about the guards."""

    def test_underpowered_is_not_reported_as_not_significant(self):
        from chapter4_stats import read_result

        result = paired_flip_test(got_better=1, got_worse=2)

        assert result["p_value"] > 0.05
        assert read_result(result, alpha=0.05) == "UNDERPOWERED"

    def test_thin_chi_squared_table_is_not_tested(self):
        from chapter4_stats import read_result

        assert read_result(chi_squared([[3, 1], [1, 2]])) == "NOT_TESTED"

    def test_real_result_is_read_normally(self):
        from chapter4_stats import read_result

        result = paired_flip_test(got_better=15, got_worse=5)

        assert read_result(result, alpha=0.05) == "SIGNIFICANT"
