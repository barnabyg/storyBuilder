"""Building blocks shared by the batch report and the calibration report."""

from collections.abc import Sequence
from pathlib import Path

from nde.eval.batch import read_warnings
from nde.eval.stats import bootstrap_mean_ci

# Fixed, so a report regenerates identically.
BOOTSTRAP_SEED = 0


def mean_ci(values: Sequence[int]) -> str:
    """A table cell: the mean and its 95% bootstrap interval, or "-" without values."""
    if not values:
        return "-"
    low, high = bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED)
    return f"{sum(values) / len(values):.2f} [{low:.2f}, {high:.2f}]"


def warnings_section(batch: Path) -> list[str]:
    """The reader's warnings, such as the 48-hour rule, or nothing when there are none."""
    warnings = read_warnings(batch)
    if not warnings:
        return []
    return ["## Warnings", "", *(f"- {w.at}: {w.warning}" for w in warnings), ""]
