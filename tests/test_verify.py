import io
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from nde.verify import main
from nde.verify.gates import GATE_ORDER, canonical_profile, quick_profile
from nde.verify.runner import Command, Gate, run_gates
from nde.verify.steps import missing_from_wheel
from tests.conftest import REPO


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


def test_quick_profile_is_every_gate_except_the_tests_in_the_same_order() -> None:
    assert [gate.name for gate in quick_profile()] == [
        name for name in GATE_ORDER if name != "tests"
    ]


def test_the_secret_scan_runs_before_anything_that_needs_uv() -> None:
    [gate] = [g for g in canonical_profile() if g.name == "dependencies-and-secrets"]

    assert gate.commands[0].args[-1] == "secrets"


@pytest.mark.parametrize(
    ("arguments", "tests_run", "note"),
    [(["--no-dashboard"], True, False), (["--no-dashboard", "--quick"], False, True)],
)
def test_main_chooses_the_gates_and_says_when_tests_were_skipped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    tests_run: bool,
    note: bool,
) -> None:
    chosen: list[str] = []

    def fake_verify(gates: Sequence[Gate], **_: object) -> int:
        chosen.extend(gate.name for gate in gates)
        return 7

    monkeypatch.setattr("nde.verify.verify", fake_verify)

    # The gates' result stays the exit status: a quick pass or failure is not altered.
    assert main(arguments) == 7

    assert ("tests" in chosen) is tests_run
    assert ("not full verification" in capsys.readouterr().out) is note


def test_ci_runs_the_full_verification() -> None:
    workflow = (REPO / ".github" / "workflows" / "verify.yml").read_text(encoding="utf-8")

    assert "nde-verify" in workflow
    assert "--quick" not in workflow


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


def test_build_check_reports_source_modules_missing_from_the_wheel(tmp_path: Path) -> None:
    package = tmp_path / "src" / "pkg"
    (package / "sub").mkdir(parents=True)
    for name in ("__init__.py", "a.py", "sub/__init__.py", "sub/b.py"):
        (package / name).write_text("", encoding="utf-8")
    wheel = tmp_path / "pkg-0.1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("pkg/__init__.py", "pkg/a.py", "pkg-0.1.dist-info/METADATA"):
            archive.writestr(name, "")

    assert missing_from_wheel(wheel, package) == ["pkg/sub/__init__.py", "pkg/sub/b.py"]
