"""`nde-verify`: the canonical, non-mutating verification command.

Interactive local runs also start the live dashboard (see docs/verification.md)
and print its address as TEST_DASHBOARD_URL. The dashboard only observes: gate
order, output and exit status are the same with it or without it.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from typing import TextIO

from nde.verify.dashboard import DEFAULT_PORT, Dashboard, DashboardState
from nde.verify.gates import canonical_profile, quick_profile
from nde.verify.pytest_reporter import EVENTS_URL_ENV
from nde.verify.runner import Gate, run_gates

# Seconds to keep serving after the run so an open page can show the final result.
DEFAULT_LINGER_S = 3.0


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="nde-verify", description=__doc__)
    parser.add_argument(
        "--no-dashboard", action="store_true", help="do not start the live dashboard"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run every gate except the tests, before pushing; this is not full verification",
    )
    return parser.parse_args(argv)


def dashboard_port(argv: Sequence[str], environ: Mapping[str, str]) -> int | None:
    """The port to try for the dashboard, or None when it is switched off."""
    args = _parse(argv)
    if args.no_dashboard or environ.get("NDE_DASHBOARD") == "0" or environ.get("CI"):
        return None
    return int(environ.get("NDE_DASHBOARD_PORT", DEFAULT_PORT))


def verify(
    gates: Sequence[Gate],
    *,
    out: TextIO,
    err: TextIO,
    dashboard_port: int | None,
    linger_s: float = DEFAULT_LINGER_S,
) -> int:
    """Run the gates; returns the exit status. The dashboard cannot change it."""
    dashboard: Dashboard | None = None
    if dashboard_port is not None:
        try:
            dashboard = Dashboard(DashboardState(), preferred_port=dashboard_port)
            out.write(f"TEST_DASHBOARD_URL={dashboard.start()}\n")
            out.flush()
        except OSError as error:
            dashboard = None
            err.write(f"nde-verify: dashboard unavailable ({error}); continuing without it\n")
    try:
        report = run_gates(
            gates,
            out=out,
            observer=dashboard.state.observer() if dashboard else None,
            extra_env={EVENTS_URL_ENV: dashboard.events_url} if dashboard else None,
        )
        if report.observer_failure is not None:
            err.write(
                f"nde-verify: dashboard stopped updating ({report.observer_failure!r}); "
                "the result above is unaffected\n"
            )
        if dashboard:
            dashboard.linger(linger_s)
    finally:
        if dashboard:
            dashboard.stop()
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    port = dashboard_port(arguments, os.environ)
    quick = _parse(arguments).quick
    gates = quick_profile() if quick else canonical_profile()
    status = verify(gates, out=sys.stdout, err=sys.stderr, dashboard_port=port)
    if quick:
        print("Quick check only: the tests did not run, so this is not full verification.")
    return status
