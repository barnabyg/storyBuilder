import pytest

from nde.eval.stats import binomial_two_sided, bootstrap_mean_ci, cohens_kappa


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        # Exact values: 2 * P(X >= max(k, n - k)) for X ~ Bin(n, 0.5), capped at 1.
        (9, 10, 0.021484375),
        (10, 10, 0.001953125),
        (0, 10, 0.001953125),
        (7, 10, 0.34375),
        (5, 10, 1.0),
        (21, 30, 0.04277394525706768),
        (17, 30, 0.584664711728692),
        (1, 1, 1.0),
    ],
)
def test_the_exact_binomial_test_matches_known_values(
    successes: int, trials: int, expected: float
) -> None:
    assert binomial_two_sided(successes, trials) == pytest.approx(expected, rel=1e-12)


def test_the_binomial_test_handles_an_uneven_null() -> None:
    # P(X >= 3) + P(X = 0) for X ~ Bin(10, 0.2): the outcomes no more likely than 3.
    assert binomial_two_sided(3, 10, 0.2) == pytest.approx(0.429574656, rel=1e-12)


def test_the_binomial_test_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="successes"):
        binomial_two_sided(11, 10)
    with pytest.raises(ValueError, match="trials"):
        binomial_two_sided(0, 0)


def test_a_bootstrap_interval_of_a_constant_is_that_constant() -> None:
    assert bootstrap_mean_ci([3, 3, 3, 3], seed=1) == (3.0, 3.0)


def test_a_bootstrap_interval_of_two_values_spans_both() -> None:
    # Resampled means of [0, 1] are 0, 0.5 or 1 with probability 1/4, 1/2, 1/4,
    # so the 2.5th and 97.5th percentiles are 0 and 1.
    assert bootstrap_mean_ci([0, 1], seed=7, resamples=4000) == (0.0, 1.0)


def test_a_bootstrap_interval_is_reproducible_with_a_fixed_seed_and_contains_the_mean() -> None:
    values = [1, 2, 3, 4, 5] * 4

    low, high = bootstrap_mean_ci(values, seed=42)

    assert bootstrap_mean_ci(values, seed=42) == (low, high)
    assert 2.0 < low < 3.0 < high < 4.0


def test_a_bootstrap_interval_needs_values() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_mean_ci([], seed=0)


def test_cohens_kappa_matches_a_worked_example() -> None:
    # 2x2 table: both "x" 20, rater 1 "x" / rater 2 "y" 5, "y"/"x" 10, both "y" 15.
    # Observed agreement 35/50 = 0.7; chance (25*30 + 25*20) / 50^2 = 0.5; kappa 0.4.
    first = ["x"] * 25 + ["y"] * 25
    second = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15

    assert cohens_kappa(first, second) == pytest.approx(0.4)


def test_cohens_kappa_handles_a_category_only_one_rater_uses() -> None:
    # Observed 2/4; chance: rater 1 is 2 x, 2 y; rater 2 is 2 x, 1 y, 1 z,
    # so chance agreement is (2*2 + 2*1) / 16 = 0.375 and kappa is 0.125 / 0.625 = 0.2.
    assert cohens_kappa(["x", "x", "y", "y"], ["x", "z", "y", "x"]) == pytest.approx(0.2)


def test_cohens_kappa_is_one_for_perfect_agreement_and_undefined_without_variation() -> None:
    assert cohens_kappa(["x", "y", "y"], ["x", "y", "y"]) == pytest.approx(1.0)
    assert cohens_kappa(["x", "x"], ["x", "x"]) is None


def test_cohens_kappa_needs_paired_ratings() -> None:
    with pytest.raises(ValueError, match="same number"):
        cohens_kappa(["x"], ["x", "y"])
