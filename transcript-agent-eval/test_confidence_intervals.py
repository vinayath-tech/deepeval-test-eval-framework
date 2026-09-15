"""
Tests for the Chapter 3 confidence interval calculations.

These make no LLM or judge calls, so they are fast and deterministic
and can gate CI without spending tokens.

The first two classes reproduce the worked examples printed in the
chapter, so a regression in the math shows up as a failed assertion
against a published number rather than against a value this repo made up.
"""

import math
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent

if str(AGENT_DIR / "evaluation") not in sys.path:
    sys.path.insert(0, str(AGENT_DIR / "evaluation"))

from confidence_intervals import (  # noqa: E402
    compare_to_threshold,
    describe_proportion,
    mean_interval,
    paired_difference_interval,
    proportion_interval,
    runs_needed_for_margin,
    t_multiplier,
    two_sample_difference_interval,
    z_multiplier,
)


class TestChapterWorkedExamples:
    """
    Case 1: 92 passes out of 100 -> about 0.867 to 0.973
    Case 2: n=100, mean 8.10, sd 1.40 -> about 7.83 to 8.37
    """

    def test_pass_fail_example(self):

        interval = proportion_interval(
            passes=92,
            total=100,
            method="wald",
        )

        assert interval["estimate"] == 0.92
        assert round(interval["standard_error"], 3) == 0.027
        assert round(interval["margin_of_error"], 3) == 0.053
        assert round(interval["lower"], 3) == 0.867
        assert round(interval["upper"], 3) == 0.973

    def test_pass_fail_plain_english(self):

        statement = describe_proportion(
            proportion_interval(
                passes=92,
                total=100,
                method="wald",
            )
        )

        assert "92% of outputs passed" in statement
        assert "about 87% to 97%" in statement

    def test_average_score_example(self):

        # Two values either side of the mean, repeated 50 times each.
        # The offset is scaled by sqrt((n-1)/n) so that the SAMPLE
        # standard deviation is 1.40, matching the chapter's input.
        # Using 1.40 directly would give a population sd of 1.40 and a
        # sample sd of 1.41.
        offset = 1.40 * math.sqrt(99 / 100)

        scores = [8.10 - offset, 8.10 + offset] * 50

        interval = mean_interval(scores)

        assert interval["estimate"] == 8.10
        assert interval["sample_standard_deviation"] == pytest.approx(
            1.40,
            abs=0.005,
        )
        assert round(interval["standard_error"], 2) == 0.14

        # The chapter uses 1.96 for n=100. This module uses the nearest
        # tabulated t multiplier below df=99, which is slightly larger
        # and therefore slightly more conservative.
        assert interval["lower"] == pytest.approx(7.83, abs=0.02)
        assert interval["upper"] == pytest.approx(8.37, abs=0.02)


class TestMultipliers:

    def test_z_is_about_1_96_at_95_percent(self):
        assert round(z_multiplier(0.95), 2) == 1.96

    def test_small_samples_get_a_larger_multiplier(self):
        # The chapter's point: with few samples the multiplier grows.
        assert t_multiplier(4, 0.95) == 2.776
        assert t_multiplier(4, 0.95) > z_multiplier(0.95)

    def test_multiplier_converges_on_z_for_large_samples(self):
        assert t_multiplier(5000, 0.95) == pytest.approx(1.96, abs=0.01)

    def test_untabulated_df_stays_conservative(self):
        # df=35 is not tabulated; it must not be narrower than df=40.
        assert t_multiplier(35, 0.95) >= t_multiplier(40, 0.95)

    def test_rejects_unsupported_confidence_level(self):
        with pytest.raises(ValueError):
            z_multiplier(0.975)


class TestProportionInterval:

    def test_wald_collapses_at_100_percent(self):
        # This is the failure mode the Wilson default exists to avoid.
        wald = proportion_interval(10, 10, method="wald")

        assert wald["lower"] == 1.0
        assert wald["upper"] == 1.0

    def test_wilson_does_not_claim_certainty_at_100_percent(self):
        wilson = proportion_interval(10, 10, method="wilson")

        assert wilson["lower"] < 0.8
        assert wilson["upper"] == 1.0

    def test_wilson_stays_within_zero_and_one(self):
        for passes in range(0, 6):
            interval = proportion_interval(passes, 5)

            assert 0.0 <= interval["lower"] <= 1.0
            assert 0.0 <= interval["upper"] <= 1.0

    def test_larger_sample_narrows_the_interval(self):
        small = proportion_interval(8, 10)
        large = proportion_interval(80, 100)

        small_width = small["upper"] - small["lower"]
        large_width = large["upper"] - large["lower"]

        assert large_width < small_width

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            proportion_interval(11, 10)

        with pytest.raises(ValueError):
            proportion_interval(1, 0)


class TestMeanInterval:

    def test_single_observation_has_no_interval(self):
        interval = mean_interval([9.0])

        assert interval["lower"] is None
        assert interval["upper"] is None
        assert "single observation" in interval["sample_size_warning"]

    def test_identical_observations_are_flagged_as_degenerate(self):
        # Zero width is an artefact of the sample, not certainty.
        interval = mean_interval([9.0, 9.0, 9.0])

        assert interval["degenerate"] is True
        assert interval["lower"] == interval["upper"] == 9.0
        assert "not evidence of certainty" in (
            interval["sample_size_warning"]
        )

    def test_more_runs_narrow_the_interval(self):
        scores = [7.0, 8.0, 9.0]

        few = mean_interval(scores)
        many = mean_interval(scores * 10)

        few_width = few["upper"] - few["lower"]
        many_width = many["upper"] - many["lower"]

        assert many_width < few_width
        assert few["estimate"] == many["estimate"]

    def test_rejects_empty_sample(self):
        with pytest.raises(ValueError):
            mean_interval([])


class TestThresholdVerdicts:

    def test_interval_entirely_above_threshold_passes(self):
        verdict = compare_to_threshold(
            mean_interval([8.9, 9.0, 9.1, 9.0, 8.8]),
            threshold=7.0,
        )

        assert verdict["verdict"] == "MEETS_THRESHOLD"

    def test_interval_entirely_below_threshold_fails(self):
        verdict = compare_to_threshold(
            mean_interval([2.0, 2.5, 3.0, 2.2, 2.8]),
            threshold=7.0,
        )

        assert verdict["verdict"] == "BELOW_THRESHOLD"

    def test_straddling_interval_is_inconclusive_not_passing(self):
        # Mean is 7.4, above the bar, but the sample cannot support it.
        verdict = compare_to_threshold(
            mean_interval([5.0, 9.0, 6.0, 9.0, 8.0]),
            threshold=7.0,
        )

        assert verdict["verdict"] == "INCONCLUSIVE"

    def test_single_observation_is_insufficient_data(self):
        verdict = compare_to_threshold(
            mean_interval([9.5]),
            threshold=7.0,
        )

        assert verdict["verdict"] == "INSUFFICIENT_DATA"


class TestDifferenceIntervals:

    def test_paired_difference_detects_a_consistent_drop(self):
        before = [9.0, 8.5, 9.2, 8.8, 9.1]
        after = [6.0, 5.5, 6.2, 5.8, 6.1]

        interval = paired_difference_interval(before, after)

        assert interval["estimate"] == pytest.approx(-3.0, abs=0.01)
        assert interval["crosses_zero"] is False
        assert interval["upper"] < 0

    def test_paired_difference_on_noise_crosses_zero(self):
        before = [9.0, 5.0, 8.0, 4.0, 7.0]
        after = [5.0, 9.0, 4.0, 8.0, 7.0]

        interval = paired_difference_interval(before, after)

        assert interval["crosses_zero"] is True

    def test_paired_requires_equal_lengths(self):
        with pytest.raises(ValueError):
            paired_difference_interval([1.0, 2.0], [1.0])

    def test_two_sample_difference_matches_chapter_formula(self):
        a = [7.0, 8.0, 9.0, 8.0, 7.0]
        b = [9.0, 9.0, 9.0, 9.0, 9.0]

        interval = two_sample_difference_interval(a, b)

        assert interval["estimate"] == pytest.approx(1.2, abs=0.01)
        assert interval["degrees_of_freedom"] >= 1

    def test_two_sample_requires_two_observations_each(self):
        with pytest.raises(ValueError):
            two_sample_difference_interval([8.0], [9.0, 9.0])


class TestRunsNeeded:

    def test_tighter_margin_needs_more_runs(self):
        loose = runs_needed_for_margin(1.5, target_margin=1.0)
        tight = runs_needed_for_margin(1.5, target_margin=0.25)

        assert tight > loose

    def test_no_spread_needs_no_estimate(self):
        assert runs_needed_for_margin(0.0, target_margin=0.5) is None

    def test_rejects_non_positive_margin(self):
        with pytest.raises(ValueError):
            runs_needed_for_margin(1.0, target_margin=0.0)
