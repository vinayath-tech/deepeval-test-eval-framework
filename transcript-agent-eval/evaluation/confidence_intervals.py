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

Standard library only. No numpy, scipy or pandas dependency, so this
module can run in CI without extra installs.
"""

import math
import statistics
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------

# Two-sided t multipliers keyed by confidence level, then degrees of freedom.
#
# The chapter notes that 1.96 is only appropriate when the sample is large
# enough. For the small samples typical of AI evals the multiplier is larger,
# so a t table is used instead of assuming 1.96.

T_TABLE: Dict[float, Dict[int, float]] = {
    0.90: {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
        6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
        16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
        21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708,
        26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697,
        40: 1.684, 50: 1.676, 60: 1.671, 80: 1.664, 100: 1.660,
        120: 1.658,
    },
    0.95: {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984,
        120: 1.980,
    },
    0.99: {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
        6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
        11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947,
        16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861, 20: 2.845,
        21: 2.831, 22: 2.819, 23: 2.807, 24: 2.797, 25: 2.787,
        26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756, 30: 2.750,
        40: 2.704, 50: 2.678, 60: 2.660, 80: 2.639, 100: 2.626,
        120: 2.617,
    },
}

SUPPORTED_CONFIDENCE_LEVELS = sorted(T_TABLE.keys())


def z_multiplier(
    confidence: float = 0.95,
) -> float:
    """
    Normal multiplier for the requested confidence level.

    Returns approximately 1.96 for a 95% interval.
    """

    _validate_confidence(confidence)

    tail = (1.0 - confidence) / 2.0

    return statistics.NormalDist().inv_cdf(1.0 - tail)


def t_multiplier(
    degrees_of_freedom: int,
    confidence: float = 0.95,
) -> float:
    """
    Two-sided t multiplier.

    Degrees of freedom that are not tabulated fall back to the largest
    tabulated value below them, which keeps the multiplier conservative
    (slightly wider interval) rather than optimistic.
    """

    _validate_confidence(confidence)

    if degrees_of_freedom < 1:
        raise ValueError(
            "degrees_of_freedom must be at least 1"
        )

    table = T_TABLE[confidence]

    if degrees_of_freedom in table:
        return table[degrees_of_freedom]

    # Beyond the table the t multiplier has effectively converged on z.
    if degrees_of_freedom > max(table):
        return round(
            z_multiplier(confidence),
            3,
        )

    tabulated = [
        df
        for df in table
        if df <= degrees_of_freedom
    ]

    if not tabulated:
        return table[1]

    return table[max(tabulated)]


def _validate_confidence(
    confidence: float,
) -> None:

    if confidence not in T_TABLE:
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

    method="wilson" is the default because the chapter notes that Wilson
    or exact binomial intervals are often better for small samples or for
    rates close to 0% or 100%, which is the normal situation in AI evals.

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

        standard_error = math.sqrt(
            (p_hat * (1.0 - p_hat)) / total
        )

        margin = z * standard_error

        lower = p_hat - margin
        upper = p_hat + margin

    else:

        denominator = 1.0 + (z ** 2) / total

        center = (
            p_hat + (z ** 2) / (2 * total)
        ) / denominator

        standard_error = math.sqrt(
            (p_hat * (1.0 - p_hat)) / total
            + (z ** 2) / (4 * total ** 2)
        )

        margin = (z / denominator) * standard_error

        lower = center - margin
        upper = center + margin

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

    The sample standard deviation uses the n-1 denominator, matching
    STDEV.S in a spreadsheet.

    A single observation has no spread to measure, so no interval is
    reported for n = 1. That is a real limitation of the sample, not a
    bug to work around.
    """

    _validate_confidence(confidence)

    n = len(scores)

    if n == 0:
        raise ValueError(
            "scores must not be empty"
        )

    mean = statistics.mean(scores)

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

    sample_standard_deviation = statistics.stdev(scores)

    standard_error = sample_standard_deviation / math.sqrt(n)

    multiplier = t_multiplier(
        degrees_of_freedom=n - 1,
        confidence=confidence,
    )

    margin = multiplier * standard_error

    # Every observation identical gives a zero-width interval. That is an
    # artefact of a small sample, not proof the true value is exact, and
    # it is the same trap as a Wald interval on a 100% pass rate.
    degenerate = sample_standard_deviation == 0

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
        "n": n,
        "estimate": round(mean, 2),
        "sample_standard_deviation": round(
            sample_standard_deviation,
            2,
        ),
        "standard_error": round(standard_error, 4),
        "multiplier": round(multiplier, 3),
        "margin_of_error": round(margin, 2),
        "lower": round(mean - margin, 2),
        "upper": round(mean + margin, 2),
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

    Degrees of freedom use the Welch-Satterthwaite approximation, which
    does not assume the two samples have equal variance.

    The chapter is explicit that comparing two marginal intervals for
    visual overlap is not a substitute for this calculation.
    """

    _validate_confidence(confidence)

    n_a = len(scores_a)
    n_b = len(scores_b)

    if n_a < 2 or n_b < 2:
        raise ValueError(
            "Both samples need at least two observations to estimate "
            "a difference interval."
        )

    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)

    var_a = statistics.variance(scores_a)
    var_b = statistics.variance(scores_b)

    difference = mean_b - mean_a

    standard_error = math.sqrt(
        (var_a / n_a) + (var_b / n_b)
    )

    degrees_of_freedom = _welch_degrees_of_freedom(
        var_a=var_a,
        n_a=n_a,
        var_b=var_b,
        n_b=n_b,
    )

    multiplier = t_multiplier(
        degrees_of_freedom=degrees_of_freedom,
        confidence=confidence,
    )

    margin = multiplier * standard_error

    lower = difference - margin
    upper = difference + margin

    return {
        "kind": "difference_independent",
        "method": "welch_t",
        "confidence": confidence,
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": round(mean_a, 2),
        "mean_b": round(mean_b, 2),
        "estimate": round(difference, 2),
        "standard_error": round(standard_error, 4),
        "degrees_of_freedom": degrees_of_freedom,
        "multiplier": round(multiplier, 3),
        "margin_of_error": round(margin, 2),
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

    The interval is a mean interval over the per-case differences.
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

    differences = [
        after - before
        for before, after in zip(
            scores_before,
            scores_after,
        )
    ]

    interval = mean_interval(
        differences,
        confidence=confidence,
    )

    interval.update(
        {
            "kind": "difference_paired",
            "pairs": len(differences),
            "mean_before": round(
                statistics.mean(scores_before),
                2,
            ),
            "mean_after": round(
                statistics.mean(scores_after),
                2,
            ),
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


def _welch_degrees_of_freedom(
    var_a: float,
    n_a: int,
    var_b: float,
    n_b: int,
) -> int:

    term_a = var_a / n_a
    term_b = var_b / n_b

    numerator = (term_a + term_b) ** 2

    denominator = (
        (term_a ** 2) / (n_a - 1)
        + (term_b ** 2) / (n_b - 1)
    )

    if denominator == 0:
        return min(n_a, n_b) - 1

    # Round down so the multiplier stays conservative.
    return max(
        1,
        int(math.floor(numerator / denominator)),
    )


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
    sample_standard_deviation: float,
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

    if sample_standard_deviation is None:
        return None

    if sample_standard_deviation == 0:
        return None

    z = z_multiplier(confidence)

    required = (
        (z * sample_standard_deviation) / target_margin
    ) ** 2

    return max(
        2,
        int(math.ceil(required)),
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
