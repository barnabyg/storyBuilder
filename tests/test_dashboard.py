import io
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence

import pytest

from nde.verify import dashboard_port, verify
from nde.verify.dashboard import Dashboard, DashboardState
from nde.verify.gates import GATE_ORDER, canonical_profile
from nde.verify.pytest_reporter import EVENTS_URL_ENV, Reporter
from nde.verify.runner import Command, Gate, GateResult, RunReport, run_gates


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _gate(name: str, code: str) -> Gate:
    return Gate(name, (Command((sys.executable, "-c", code)),))


def test_the_state_follows_a_run_through_gates_tests_and_result() -> None:
    clock = Clock()
    state = DashboardState(clock=clock)

    state.apply({"type": "run_started", "gates": ["lint", "tests"]})
    state.apply({"type": "gate_started", "gate": "lint"})
    state.apply({"type": "output", "line": "All checks passed!"})
    state.apply({"type": "gate_finished", "gate": "lint", "status": "passed", "seconds": 0.5})
    state.apply({"type": "gate_started", "gate": "tests"})
    state.apply({"type": "tests_collected", "count": 3})
    state.apply({"type": "test_started", "nodeid": "t.py::a"})
    state.apply({"type": "test_finished", "nodeid": "t.py::a", "outcome": "passed"})
    state.apply({"type": "test_started", "nodeid": "t.py::b"})
    state.apply({"type": "test_finished", "nodeid": "t.py::b", "outcome": "failed"})
    clock.now = 112.5
    snapshot = state.snapshot()

    assert snapshot["status"] == "running"
    assert snapshot["active_gate"] == "tests"
    assert snapshot["elapsed_s"] == 12.5
    assert [(g["name"], g["status"]) for g in snapshot["gates"]] == [
        ("lint", "passed"),
        ("tests", "running"),
    ]
    tests = snapshot["tests"]
    assert (tests["collected"], tests["done"], tests["passed"], tests["failed"]) == (3, 2, 1, 1)
    assert tests["current"] is None
    assert tests["failures"] == ["t.py::b"]
    assert snapshot["output"] == ["All checks passed!"]

    state.apply({"type": "gate_finished", "gate": "tests", "status": "failed", "seconds": 2.0})
    state.apply({"type": "run_finished", "passed": False})
    final = state.snapshot()

    assert (final["status"], final["active_gate"]) == ("failed", None)
    assert final["failures"] == [{"gate": "tests", "status": "failed"}]


def test_recent_output_is_bounded() -> None:
    state = DashboardState(max_output_lines=3)
    for index in range(5):
        state.apply({"type": "output", "line": f"line {index}"})

    assert state.snapshot()["output"] == ["line 2", "line 3", "line 4"]


def test_unknown_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown event"):
        DashboardState().apply({"type": "explode"})


def test_the_dashboard_lists_the_canonical_gates_in_order() -> None:
    state = DashboardState()
    observer = state.observer()

    observer.run_started(canonical_profile())

    assert tuple(g["name"] for g in state.snapshot()["gates"]) == GATE_ORDER


@pytest.fixture
def dashboard() -> Iterator[Dashboard]:
    server = Dashboard(DashboardState(), preferred_port=0)
    server.start()
    yield server
    server.stop()


def _get(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - local test server
        return response.status, response.headers["Content-Type"], response.read().decode()


def _post(url: str, body: bytes) -> int:
    request = urllib.request.Request(url, data=body, method="POST")  # noqa: S310 - local test server
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - local test server
            return int(response.status)
    except urllib.error.HTTPError as error:
        return error.code


def test_the_server_serves_the_page_and_the_state(dashboard: Dashboard) -> None:
    status, content_type, page = _get(dashboard.url)
    assert (status, content_type.split(";")[0]) == (200, "text/html")
    assert "nde-verify" in page

    status, content_type, body = _get(dashboard.url + "api/state")
    assert (status, content_type) == (200, "application/json")
    assert json.loads(body)["status"] == "waiting"


def test_the_server_accepts_events_and_rejects_bad_ones(dashboard: Dashboard) -> None:
    event = json.dumps({"type": "tests_collected", "count": 4}).encode()

    assert _post(dashboard.events_url, event) == 204
    assert _post(dashboard.events_url, b"not json") == 400
    assert _post(dashboard.events_url, b'{"type": "explode"}') == 400
    assert json.loads(_get(dashboard.url + "api/state")[2])["tests"]["collected"] == 4


def test_the_server_listens_on_localhost_only(dashboard: Dashboard) -> None:
    assert dashboard.url.startswith("http://127.0.0.1:")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_concurrent_dashboards_fall_back_to_different_ports() -> None:
    preferred = _free_port()
    first = Dashboard(DashboardState(), preferred_port=preferred)
    second = Dashboard(DashboardState(), preferred_port=preferred)
    try:
        first.start()
        second.start()

        assert first.port == preferred
        assert second.port not in {0, preferred}
        assert _get(first.url + "api/state")[0] == _get(second.url + "api/state")[0] == 200
    finally:
        first.stop()
        second.stop()


def test_a_failing_observer_changes_neither_output_nor_result() -> None:
    gates = [_gate("one", "print('hello')"), _gate("two", "raise SystemExit(1)")]

    class Broken:
        """Fails on its first output line, after the run and first gate have started."""

        def run_started(self, gates: Sequence[Gate]) -> None:
            pass

        def gate_started(self, gate: Gate) -> None:
            pass

        def output(self, line: str) -> None:
            raise RuntimeError(line)

        def gate_finished(self, result: GateResult) -> None:
            raise AssertionError(f"called after a failure: {result}")

        def run_finished(self, report: RunReport) -> None:
            raise AssertionError(f"called after a failure: {report}")

    plain, observed = io.StringIO(), io.StringIO()
    expected = run_gates(gates, out=plain)
    report = run_gates(gates, out=observed, observer=Broken())

    assert _without_timings(observed.getvalue()) == _without_timings(plain.getvalue())
    assert [r.status for r in report.results] == [r.status for r in expected.results]
    assert isinstance(report.observer_failure, RuntimeError)


def test_the_reporter_disables_itself_when_the_dashboard_is_unreachable() -> None:
    reporter = Reporter(f"http://127.0.0.1:{_free_port()}/api/events")

    reporter.send({"type": "test_started", "nodeid": "x"})
    reporter.send({"type": "test_started", "nodeid": "y"})

    assert not reporter.enabled


def _without_timings(text: str) -> str:
    return re.sub(r"\(\d+\.\d+s\)", "(…s)", text)


def _profile() -> list[Gate]:
    return [
        _gate("format", "print('formatted')"),
        _gate(
            "tests",
            "import os; print('reporter' if os.environ.get('" + EVENTS_URL_ENV + "') else '')",
        ),
        _gate("build", "raise SystemExit(3)"),
    ]


def test_gate_order_output_and_exit_status_match_with_and_without_the_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under `nde-verify` itself, the outer dashboard's variable is already set.
    monkeypatch.delenv(EVENTS_URL_ENV, raising=False)
    plain_out, plain_err = io.StringIO(), io.StringIO()
    live_out, live_err = io.StringIO(), io.StringIO()

    plain = verify(_profile(), out=plain_out, err=plain_err, dashboard_port=None)
    live = verify(_profile(), out=live_out, err=live_err, dashboard_port=0, linger_s=0)

    assert plain == live == 1
    first, *rest = live_out.getvalue().splitlines(keepends=True)
    assert first.startswith("TEST_DASHBOARD_URL=http://127.0.0.1:")
    # Only the reporter variable differs: the tests gate sees it with the dashboard on.
    live_gates = _without_timings("".join(rest))
    assert "reporter\n" in live_gates
    assert live_gates.replace("reporter\n", "\n") == _without_timings(plain_out.getvalue())


def test_verification_continues_without_the_dashboard_when_it_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(_self: Dashboard) -> str:
        raise OSError("no sockets today")

    monkeypatch.setattr(Dashboard, "start", refuse)
    out, err = io.StringIO(), io.StringIO()

    code = verify([_gate("only", "print('ok')")], out=out, err=err, dashboard_port=0)

    assert code == 0
    assert "TEST_DASHBOARD_URL" not in out.getvalue()
    assert "dashboard unavailable" in err.getvalue()


@pytest.mark.parametrize(
    ("argv", "environ", "expected"),
    [
        ([], {}, 8765),
        (["--no-dashboard"], {}, None),
        ([], {"NDE_DASHBOARD": "0"}, None),
        ([], {"CI": "true"}, None),
        ([], {"NDE_DASHBOARD_PORT": "9001"}, 9001),
    ],
    ids=["default", "flag", "env-opt-out", "ci", "port"],
)
def test_the_dashboard_is_on_locally_and_off_in_ci_or_when_opted_out(
    argv: list[str], environ: dict[str, str], expected: int | None
) -> None:
    assert dashboard_port(argv, environ) == expected


def test_a_viewer_receives_the_final_state_before_the_dashboard_closes(
    dashboard: Dashboard,
) -> None:
    _get(dashboard.url + "api/state")
    dashboard.state.apply({"type": "run_finished", "passed": True})
    seen: list[str] = []
    viewer = threading.Thread(target=lambda: seen.append(_get(dashboard.url + "api/state")[2]))

    viewer.start()
    dashboard.linger(timeout_s=5)
    viewer.join()

    assert json.loads(seen[0])["status"] == "passed"


def test_lingering_is_skipped_when_nobody_is_watching(dashboard: Dashboard) -> None:
    dashboard.state.apply({"type": "run_finished", "passed": True})
    started = time.monotonic()

    dashboard.linger(timeout_s=60)

    assert time.monotonic() - started < 1


def test_the_pytest_gate_loads_the_reporter_plugin() -> None:
    [tests_gate] = [gate for gate in canonical_profile() if gate.name == "tests"]

    assert "nde.verify.pytest_reporter" in tests_gate.commands[0].args
