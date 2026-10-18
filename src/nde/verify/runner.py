"""Run verification gates in order and decide pass or fail.

A gate passes only when every command in it exits with status 0 and prints no
warning. Commands run in order and a gate stops at its first failing command.
Later gates still run, so one verification shows every problem at once.
"""

import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TextIO

GateStatus = Literal["passed", "failed", "warning"]

# Lines that mean a tool emitted a warning:
# - "warning: ..." at the start of a line or after whitespace (uv, pyright, many CLIs)
# - Python's "SomethingWarning: ..." from the warnings module
# - pytest's warnings summary header
_WARNING_PATTERNS = (
    re.compile(r"(?im)(?:^|\s)warning:"),
    re.compile(r"(?m)\b[A-Z]\w*Warning: "),
    re.compile(r"(?im)^=+ warnings summary =+$"),
)


@dataclass(frozen=True)
class Command:
    args: tuple[str, ...]

    def display(self) -> str:
        return " ".join(self.args)


@dataclass(frozen=True)
class Gate:
    name: str
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    exit_code: int
    seconds: float
    output: str


@dataclass
class RunReport:
    results: list[GateResult] = field(default_factory=list[GateResult])

    @property
    def passed(self) -> bool:
        return all(result.status == "passed" for result in self.results)


def contains_warning(text: str) -> bool:
    return any(pattern.search(text) for pattern in _WARNING_PATTERNS)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # Child Python tools write UTF-8 regardless of the Windows console code page.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_command(command: Command, out: TextIO) -> tuple[int, str]:
    out.write(f"$ {command.display()}\n")
    out.flush()
    try:
        # S603: the commands come from the repository's own gate profile, not user input.
        process = subprocess.Popen(  # noqa: S603
            command.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_child_env(),
        )
    except OSError as error:
        message = f"could not start {command.args[0]}: {error}\n"
        out.write(message)
        return 127, message

    lines: list[str] = []
    with process:
        stdout = process.stdout
        if stdout is None:  # pragma: no cover - stdout=PIPE always provides a stream
            raise RuntimeError("subprocess stdout was not captured")
        for line in stdout:
            lines.append(line)
            out.write(line)
            out.flush()
        exit_code = process.wait()
    return exit_code, "".join(lines)


def run_gate(gate: Gate, out: TextIO) -> GateResult:
    started = time.monotonic()
    outputs: list[str] = []
    exit_code = 0
    for command in gate.commands:
        exit_code, output = _run_command(command, out)
        outputs.append(output)
        if exit_code != 0:
            break
    output = "".join(outputs)
    status: GateStatus
    if exit_code != 0:
        status = "failed"
    elif contains_warning(output):
        status = "warning"
    else:
        status = "passed"
    return GateResult(
        gate=gate.name,
        status=status,
        exit_code=exit_code,
        seconds=time.monotonic() - started,
        output=output,
    )


def run_gates(gates: Sequence[Gate], out: TextIO) -> RunReport:
    report = RunReport()
    for index, gate in enumerate(gates, start=1):
        out.write(f"\n=== [{index}/{len(gates)}] {gate.name} ===\n")
        out.flush()
        result = run_gate(gate, out)
        report.results.append(result)
        out.write(f"--- {gate.name}: {result.status} ({result.seconds:.1f}s)\n")
    _write_summary(report, out)
    return report


def _write_summary(report: RunReport, out: TextIO) -> None:
    out.write("\n=== summary ===\n")
    for result in report.results:
        out.write(f"{result.status.upper():8} {result.gate} ({result.seconds:.1f}s)\n")
    out.write("verification PASSED\n" if report.passed else "verification FAILED\n")
    out.flush()
