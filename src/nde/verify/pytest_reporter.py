"""A pytest plugin that reports per-test progress to the `nde-verify` dashboard.

The tests gate always loads it (`-p nde.verify.pytest_reporter`), so the gate's
command line is the same with and without the dashboard. It does nothing unless
NDE_DASHBOARD_EVENTS_URL names a dashboard on 127.0.0.1, and it switches itself
off at the first failed delivery, so a missing or broken dashboard never
affects the test run.
"""

import json
import os
import urllib.request
from typing import Any

import pytest

EVENTS_URL_ENV = "NDE_DASHBOARD_EVENTS_URL"
_LOCAL_PREFIX = "http://127.0.0.1:"
_TIMEOUT_S = 1.0


class Reporter:
    def __init__(self, url: str | None) -> None:
        self.url = url
        self.enabled = bool(url and url.startswith(_LOCAL_PREFIX))

    def send(self, event: dict[str, Any]) -> None:
        if not self.enabled or self.url is None:
            return
        request = urllib.request.Request(  # noqa: S310 - only http://127.0.0.1 URLs get here
            self.url,
            data=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S):  # noqa: S310 - as above
                pass
        except (OSError, ValueError):
            self.enabled = False


_reporter = Reporter(os.environ.get(EVENTS_URL_ENV))


def pytest_collection_finish(session: pytest.Session) -> None:
    _reporter.send({"type": "tests_collected", "count": len(session.items)})


def pytest_runtest_logstart(nodeid: str) -> None:
    _reporter.send({"type": "test_started", "nodeid": nodeid})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # One result per test: the call phase, or setup when it failed or skipped the test.
    if report.when == "call" or (report.when == "setup" and not report.passed):
        _reporter.send(
            {"type": "test_finished", "nodeid": report.nodeid, "outcome": report.outcome}
        )
