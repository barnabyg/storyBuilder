import io
import sys

from nde.verify.gates import GATE_ORDER, canonical_profile
from nde.verify.runner import Command, Gate, run_gates


def _python(code: str) -> Command:
    return Command((sys.executable, "-c", code))


def _gate(name: str, *codes: str) -> Gate:
    return Gate(name=name, commands=tuple(_python(code) for code in codes))


def test_canonical_profile_runs_the_seven_gates_in_standard_order() -> None:
    names = [gate.name for gate in canonical_profile()]

    assert names == [
        "format",
        "lint",
        "types",
        "static-analysis",
        "tests",
        "dependencies-and-secrets",
        "build",
    ]
    assert tuple(names) == GATE_ORDER


def test_clean_gates_pass() -> None:
    out = io.StringIO()

    report = run_gates([_gate("one", "print('ok')"), _gate("two", "pass")], out=out)

    assert report.passed
    assert [result.status for result in report.results] == ["passed", "passed"]


def test_nonzero_exit_fails_the_gate_and_the_run() -> None:
    out = io.StringIO()

    report = run_gates([_gate("broken", "raise SystemExit(3)")], out=out)

    assert not report.passed
    assert report.results[0].status == "failed"
    assert report.results[0].exit_code == 3


def test_warning_output_fails_a_gate_even_with_zero_exit() -> None:
    out = io.StringIO()

    report = run_gates(
        [
            _gate("uv-style", "print('warning: something is off')"),
            _gate("python-style", "print('x.py:1: DeprecationWarning: old api')"),
            _gate("tool-style", "print('  src/a.py:1:2 - warning: unused import')"),
        ],
        out=out,
    )

    assert [result.status for result in report.results] == ["warning", "warning", "warning"]
    assert not report.passed


def test_words_mentioning_warnings_are_not_warnings() -> None:
    out = io.StringIO()

    report = run_gates([_gate("summary", "print('0 errors, 0 warnings, 0 informations')")], out=out)

    assert report.passed


def test_later_gates_still_run_after_a_failure_and_output_is_streamed() -> None:
    out = io.StringIO()

    report = run_gates(
        [_gate("first", "raise SystemExit(1)"), _gate("second", "print('second ran')")],
        out=out,
    )

    assert [result.status for result in report.results] == ["failed", "passed"]
    assert "second ran" in out.getvalue()


def test_a_gate_stops_at_its_first_failing_command() -> None:
    out = io.StringIO()

    report = run_gates(
        [_gate("multi", "raise SystemExit(1)", "print('should not run')")],
        out=out,
    )

    assert report.results[0].status == "failed"
    assert "should not run" not in out.getvalue()


def test_a_missing_executable_fails_the_gate_instead_of_crashing() -> None:
    out = io.StringIO()

    report = run_gates(
        [Gate(name="missing", commands=(Command(("definitely-not-a-real-tool-xyz",)),))],
        out=out,
    )

    assert report.results[0].status == "failed"
