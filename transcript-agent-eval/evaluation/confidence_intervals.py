"""
Jason Arbon Testing AI - Chapter 3

Confidence Intervals

Implements the calculations described in the chapter:

1. Case 1 - Pass/Fail results       -> proportion_interval()
2. Case 2 - Average 0-10 scores     -> mean_interval()
3. Comparing two versions           -> two_sample_difference_interval()
4. Paired comparison (preferred)    -> paired_difference_interval()

The general pattern used throughout is:

    confidence interval = estimate +/- multiplier * standard error

The actual distribution math (t and normal quantiles, Wilson score
intervals, and the t-test confidence intervals themselves) comes from
`scipy.stats` rather than a hand-rolled formula or a lookup table, so
this module is a thin, book-shaped wrapper around library calls rather
than an independent implementation of the statistics.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------

SUPPORTED_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


def z_multiplier(
    confidence: float = 0.95,
) -> float:
    """
    Normal multiplier for the requested confidence level.

    Returns approximately 1.96 for a 95% interval. Backed by
    scipy.stats.norm.ppf rather than a hardcoded constant.
    """

    _validate_confidence(confidence)

    tail = (1.0 - confidence) / 2.0

    return float(
        stats.norm.ppf(1.0 - tail)
    )


def t_multiplier(
    degrees_of_freedom: float,
    confidence: float = 0.95,
) -> float:
    """
    Two-sided t multiplier for the given degrees of freedom.

    Backed by scipy.stats.t.ppf, so this is exact for any degrees of
    freedom rather than an approximation looked up from a table. As df
    grows, this converges on z_multiplier() the same way the chapter's
    t-table does.
    """

    _validate_confidence(confidence)

    if degrees_of_freedom < 1:
        raise ValueError(
            "degrees_of_freedom must be at least 1"
        )

    tail = (1.0 - confidence) / 2.0

    return float(
        stats.t.ppf(1.0 - tail, df=degrees_of_freedom)
    )


def _validate_confidence(
    confidence: float,
) -> None:

    if confidence not in SUPPORTED_CONFIDENCE_LEVELS:
        raise ValueError(
            f"Unsupported confidence level: {confidence}. "
            f"Supported levels: {SUPPORTED_CONFIDENCE_LEVELS}"
        )


# ---------------------------------------------------------------------------
# Case 1 - Pass/Fail results
# ---------------------------------------------------------------------------

def proportion_interval(
    passes: int,
    total: int,
    confidence: float = 0.95,
    method: str = "wilson",
) -> Dict[str, Any]:
    """
    Confidence interval for a pass rate.

    method="wald" reproduces the simple formula from the chapter:

        p_hat          = passes / total
        standard_error = sqrt((p_hat * (1 - p_hat)) / total)
        interval       = p_hat +/- 1.96 * standard_error

    scipy has no Wald proportion interval (it is known to misbehave at
    the edges), so this is the one formula in the module still written
    out directly; the multiplier itself still comes from
    scipy.stats.norm.

    method="wilson" is the default, computed by
    scipy.stats.binomtest(...).proportion_ci(method="wilson"), because
    the chapter notes that Wilson or exact binomial intervals are often
    better for small samples or for rates close to 0% or 100%, which is
    the normal situation in AI evals.

    A Wald interval on a 10/10 sample collapses to [1.0, 1.0], which
    claims a certainty the sample does not support. Wilson does not.
    """

    _validate_confidence(confidence)

    if total <= 0:
        raise ValueError(
            "total must be greater than zero"
        )

    if not 0 <= passes <= total:
        raise ValueError(
            "passes must be between 0 and total"
        )

    if method not in ("wald", "wilson"):
        raise ValueError(
            f"Unknown method: {method}. Use 'wald' or 'wilson'."
        )

    p_hat = passes / total

    z = z_multiplier(confidence)

    if method == "wald":

        standard_error = float(
            np.sqrt((p_hat * (1.0 - p_hat)) / total)
        )

        margin = z * standard_error

        lower = p_hat - margin
        upper = p_hat + margin

    else:

        result = stats.binomtest(passes, total)

        ci = result.proportion_ci(
            confidence_level=confidence,
            method="wilson",
        )

        lower, upper = ci.low, ci.high

        # Report the standard error/margin in the same shape as Wald so
        # callers and the report tables don't need to branch on method.
        standard_error = (upper - lower) / (2 * z)
        margin = z * standard_error

    return {
        "kind": "proportion",
        "method": method,
        "confidence": confidence,
        "passes": passes,
        "total": total,
        "estimate": round(p_hat, 4),
        "standard_error": round(standard_error, 4),
        "multiplier": round(z, 3),
        "margin_of_error": round(margin, 4),
        "lower": round(max(0.0, lower), 4),
        "upper": round(min(1.0, upper), 4),
        "sample_size_warning": _sample_size_warning(total),
    }


# ---------------------------------------------------------------------------
# Case 2 - Average 0-10 scores
# ---------------------------------------------------------------------------

def mean_interval(
    scores: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Confidence interval for an average score.

        mean           = sum(scores) / n
        standard_error = sample_standard_deviation / sqrt(n)
        interval       = mean +/- t_multiplier * standard_error

    numpy computes the mean and sample standard deviation (ddof=1,
    matching STDEV.S in a spreadsheet); the interval itself is produced
    by scipy.stats.ttest_1samp(...).confidence_interval(), a one-sample
    t-test's confidence interval for the mean, rather than assembled by
    hand from the multiplier and standard error.

    A single observation has no spread to measure, so no interval is
    reported for n = 1. That is a real limitation of the sample, not a
    bug to work around.
    """

    _validate_confidence(confidence)

    values = np.asarray(scores, dtype=float)

    n = values.size

    if n == 0:
        raise ValueError(
            "scores must not be empty"
        )

    mean = float(np.mean(values))

    if n == 1:

        return {
            "kind": "mean",
            "method": "t",
            "confidence": confidence,
            "n": 1,
            "estimate": round(mean, 2),
            "sample_standard_deviation": None,
            "standard_error": None,
            "multiplier": None,
            "margin_of_error": None,
            "lower": None,
            "upper": None,
            "degenerate": False,
            "sample_size_warning": (
                "A single observation gives no estimate of variability. "
                "No confidence interval can be calculated."
            ),
        }

    sample_standard_deviation = float(np.std(values, ddof=1))

    degenerate = sample_standard_deviation == 0

    with warnings.catch_warnings():

        # A degenerate (zero-variance) sample makes scipy warn about
        # precision loss in its internal moment calculation. The result
        # (a zero-width interval) is correct and is exactly what
        # `degenerate` below is flagging, so the warning is expected
        # noise here rather than a sign something went wrong.
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
        )

        result = stats.ttest_1samp(values, popmean=0.0)
        ci = result.confidence_interval(confidence_level=confidence)

    lower = float(ci.low)
    upper = float(ci.high)

    standard_error = sample_standard_deviation / float(np.sqrt(n))
    multiplier = t_multiplier(
        degrees_of_freedom=n - 1,
        confidence=confidence,
    )
    margin = multiplier * standard_error

    warning = _sample_size_warning(n)

    if degenerate:

        degenerate_warning = (
            f"All {n} observations were identical, so the interval has "
            f"zero width. This is not evidence of certainty."
        )

        warning = (
            f"{degenerate_warning} {warning}"
            if warning
            else degenerate_warning
        )

    return {
        "kind": "mean",
        "method": "t",
        "confidence": confidence,
        "n": int(n),
        "estimate": round(mean, 2),
        "sample_standard_deviation": round(
            sample_standard_deviation,
            2,
        ),
        "standard_error": round(standard_error, 4),
        "multiplier": round(multiplier, 3),
        "margin_of_error": round(margin, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "degenerate": degenerate,
        "sample_size_warning": warning,
    }


# ---------------------------------------------------------------------------
# Comparing two versions
# ---------------------------------------------------------------------------

def two_sample_difference_interval(
    scores_a: List[float],
    scores_b: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Confidence interval around the difference between two independent
    samples (Version B minus Version A).

        difference                = mean_B - mean_A
        standard_error_difference = sqrt((sd_A^2 / n_A) + (sd_B^2 / n_B))
        interval                  = difference +/- t * standard_error_difference

    Computed by scipy.stats.ttest_ind(..., equal_var=False), Welch's
    t-test, which does not assume the two samples have equal variance.
    Its .confidence_interval() gives the interval directly and its
    `.df` gives the Welch-Satterthwaite degrees of freedom, so neither
    is derived by hand here.

    The chapter is explicit that comparing two marginal intervals for
    visual overlap is not a substitute for this calculation.
    """

    _validate_confidence(confidence)

    values_a = np.asarray(scores_a, dtype=float)
    values_b = np.asarray(scores_b, dtype=float)

    n_a = values_a.size
    n_b = values_b.size

    if n_a < 2 or n_b < 2:
        raise ValueError(
            "Both samples need at least two observations to estimate "
            "a difference interval."
        )

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))

    with warnings.catch_warnings():

        # As in mean_interval(), a zero-variance sample (every score in
        # one version identical) makes scipy warn about precision loss
        # in its internal moment calculation. The resulting interval is
        # still correct.
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
        )

        result = stats.ttest_ind(
            values_b,
            values_a,
            equal_var=False,
        )

        ci = result.confidence_interval(confidence_level=confidence)

    lower = float(ci.low)
    upper = float(ci.high)

    difference = mean_b - mean_a
    degrees_of_freedom = float(result.df)

    standard_error = (
        (upper - lower)
        / (2 * t_multiplier(degrees_of_freedom, confidence))
    )

    return {
        "kind": "difference_independent",
        "method": "welch_t",
        "confidence": confidence,
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "estimate": round(difference, 2),
        "standard_error": round(standard_error, 4),
        "degrees_of_freedom": round(degrees_of_freedom, 2),
        "multiplier": round(
            t_multiplier(degrees_of_freedom, confidence),
            3,
        ),
        "margin_of_error": round((upper - lower) / 2, 2),
        "lower": round(lower, 2),
        "upper": round(upper, 2),
        "crosses_zero": lower <= 0 <= upper,
        "sample_size_warning": _sample_size_warning(
            min(n_a, n_b)
        ),
    }


def paired_difference_interval(
    scores_before: List[float],
    scores_after: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Confidence interval around a paired difference (after minus before).

    The chapter prefers this form when the same eval cases run through
    both versions, because differencing case by case removes a lot of
    case-to-case noise.

    Mathematically, a paired t-test is a one-sample t-test on the
    per-pair differences (this is also how scipy.stats.ttest_rel is
    implemented internally), so the differences are handed to
    mean_interval(), which is itself backed by
    scipy.stats.ttest_1samp(...).confidence_interval().
    """

    _validate_confidence(confidence)

    if len(scores_before) != len(scores_after):
        raise ValueError(
            "Paired samples must be the same length."
        )

    if not scores_before:
        raise ValueError(
            "Paired samples must not be empty."
        )

    before = np.asarray(scores_before, dtype=float)
    after = np.asarray(scores_after, dtype=float)

    differences = (after - before).tolist()

    interval = mean_interval(
        differences,
        confidence=confidence,
    )

    interval.update(
        {
            "kind": "difference_paired",
            "pairs": len(differences),
            "mean_before": round(float(np.mean(before)), 2),
            "mean_after": round(float(np.mean(after)), 2),
            "differences": [
                round(value, 2)
                for value in differences
            ],
            "crosses_zero": (
                interval["lower"] is not None
                and interval["lower"] <= 0 <= interval["upper"]
            ),
        }
    )

    return interval


# ---------------------------------------------------------------------------
# Decisions and reporting
# ---------------------------------------------------------------------------

def compare_to_threshold(
    interval: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    """
    Turn an interval into a release-relevant verdict.

    The point of the chapter is that a point estimate above the bar is
    not the same as evidence that the true value is above the bar.

        MEETS_THRESHOLD   entire interval is at or above the threshold
        BELOW_THRESHOLD   entire interval is below the threshold
        INCONCLUSIVE      the interval straddles the threshold

    INCONCLUSIVE is not a failure of the system under test. It means the
    sample is too small or too noisy to separate the two possibilities.
    """

    lower = interval.get("lower")
    upper = interval.get("upper")

    if lower is None or upper is None:

        return {
            "verdict": "INSUFFICIENT_DATA",
            "threshold": threshold,
            "reason": (
                "No interval could be calculated from this sample."
            ),
        }

    if lower >= threshold:

        return {
            "verdict": "MEETS_THRESHOLD",
            "threshold": threshold,
            "reason": (
                f"The entire interval ({lower} to {upper}) is at or "
                f"above the threshold of {threshold}."
            ),
        }

    if upper < threshold:

        return {
            "verdict": "BELOW_THRESHOLD",
            "threshold": threshold,
            "reason": (
                f"The entire interval ({lower} to {upper}) is below "
                f"the threshold of {threshold}."
            ),
        }

    return {
        "verdict": "INCONCLUSIVE",
        "threshold": threshold,
        "reason": (
            f"The interval ({lower} to {upper}) straddles the "
            f"threshold of {threshold}. This sample cannot separate "
            f"passing from failing."
        ),
    }


def runs_needed_for_margin(
    sample_standard_deviation: Optional[float],
    target_margin: float,
    confidence: float = 0.95,
) -> Optional[int]:
    """
    Approximate number of runs needed to get the margin of error down to
    target_margin, assuming the spread stays roughly the same.

        n = (z * sd / target_margin) ^ 2

    This answers the practical question raised by an INCONCLUSIVE
    verdict: how much more sampling would it take to decide?

    It uses z rather than t, so for small samples it is an optimistic
    lower bound on the runs required, not a guarantee.
    """

    _validate_confidence(confidence)

    if target_margin <= 0:
        raise ValueError(
            "target_margin must be greater than zero"
        )

    if not sample_standard_deviation:
        return None

    z = z_multiplier(confidence)

    required = (
        (z * sample_standard_deviation) / target_margin
    ) ** 2

    return max(
        2,
        int(np.ceil(required)),
    )


def describe_proportion(
    interval: Dict[str, Any],
    label: str = "outputs",
) -> str:
    """
    Plain-English sentence in the style the chapter recommends.
    """

    confidence_percent = f"{interval['confidence']:.0%}"

    return (
        f"In this sample, {interval['estimate']:.0%} of {label} passed "
        f"({interval['passes']}/{interval['total']}). The approximate "
        f"{confidence_percent} confidence interval is about "
        f"{interval['lower']:.0%} to {interval['upper']:.0%} "
        f"({interval['method']} method)."
    )


def describe_mean(
    interval: Dict[str, Any],
    label: str = "score",
) -> str:
    """
    Plain-English sentence in the style the chapter recommends.
    """

    confidence_percent = f"{interval['confidence']:.0%}"

    if interval["lower"] is None:

        return (
            f"The observed {label} was {interval['estimate']} from a "
            f"single run. No confidence interval can be calculated "
            f"from one observation."
        )

    return (
        f"The average {label} was {interval['estimate']}. The "
        f"approximate {confidence_percent} confidence interval is "
        f"about {interval['lower']} to {interval['upper']} "
        f"(n={interval['n']})."
    )


def describe_difference(
    interval: Dict[str, Any],
    label: str = "score",
) -> str:
    """
    Plain-English sentence for a difference interval.
    """

    confidence_percent = f"{interval['confidence']:.0%}"

    if interval["lower"] is None:

        return (
            f"The observed change in {label} was "
            f"{interval['estimate']}, from too few pairs to calculate "
            f"an interval."
        )

    if interval["crosses_zero"]:

        conclusion = (
            "The interval crosses zero, so these data do not clearly "
            "separate the two versions under this method."
        )

    elif interval["upper"] < 0:

        conclusion = (
            "The entire interval is below zero, so the evidence of a "
            "decrease on this measure is strong."
        )

    else:

        conclusion = (
            "The entire interval is above zero, so the evidence of an "
            "increase on this measure is strong."
        )

    return (
        f"The observed change in {label} was {interval['estimate']}. "
        f"The approximate {confidence_percent} confidence interval is "
        f"about {interval['lower']} to {interval['upper']}. "
        f"{conclusion}"
    )


def _sample_size_warning(
    n: int,
) -> Optional[str]:

    if n < 5:
        return (
            f"Sample size is {n}. The interval is very wide and should "
            f"not be treated as evidence for a high-risk release."
        )

    if n < 20:
        return (
            f"Sample size is {n}. The interval is wide. Increasing the "
            f"number of runs will narrow it."
        )

    return None
