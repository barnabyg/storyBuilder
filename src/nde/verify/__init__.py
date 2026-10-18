"""`nde-verify`: the canonical, non-mutating verification command."""

import sys

from nde.verify.gates import canonical_profile
from nde.verify.runner import run_gates


def main() -> int:
    report = run_gates(canonical_profile(), out=sys.stdout)
    return 0 if report.passed else 1
