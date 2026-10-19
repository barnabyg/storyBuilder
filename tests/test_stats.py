import pytest

from nde.eval.stats import binomial_two_sided


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
