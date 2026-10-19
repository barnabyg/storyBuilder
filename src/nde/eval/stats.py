"""Statistics for evaluation reports, with no dependencies beyond the standard library."""

import math
import random
from collections.abc import Sequence

# Relative tolerance when comparing outcome probabilities, as scipy's binomtest uses.
_RELATIVE_TOLERANCE = 1 + 1e-7


def _binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test: the probability of an outcome no more likely than this one.

    Matches scipy.stats.binomtest(successes, trials, p).pvalue.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    observed = _binomial_pmf(successes, trials, p)
    total = sum(
        probability
        for k in range(trials + 1)
        if (probability := _binomial_pmf(k, trials, p)) <= observed * _RELATIVE_TOLERANCE
    )
    return min(1.0, total)


DEFAULT_RESAMPLES = 10_000


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
) -> tuple[float, float]:
    """A percentile bootstrap confidence interval for the mean of `values`.

    Deterministic for a given seed, so a report regenerates identically.
    """
    if not values:
        raise ValueError("a bootstrap interval needs at least one value")
    rng = random.Random(seed)  # noqa: S311 - resampling for statistics, not cryptography
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    low = means[math.floor(tail * resamples)]
    high = means[math.ceil((1 - tail) * resamples) - 1]
    return float(low), float(high)
