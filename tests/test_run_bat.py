"""Opt-in: run.bat in a real console window, stopped the way a person stops it."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import REPO, write_config
from tests.continuity_support import continuity_config
from tests.fake_openai import FakeOpenAI

# Joins the console of the process named, and sends it Ctrl+C without being stopped by it.
_PRESS_CTRL_C = """
import ctypes, sys
console = ctypes.windll.kernel32
console.FreeConsole()
attached = console.AttachConsole(int(sys.argv[1]))
ignoring = console.SetConsoleCtrlHandler(None, True)
sys.exit(0 if attached and ignoring and console.GenerateConsoleCtrlEvent(0, 0) else 1)
"""
_LIMIT_S = 30
# subprocess names this only on Windows, and the type check also runs on Linux.
_CREATE_NO_WINDOW = 0x08000000


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _powershell(command: str) -> str:
    found = subprocess.run(  # noqa: S603 - fixed program, with commands from this test
        ["powershell", "-NoProfile", "-Command", command],  # noqa: S607 - always on the path
        capture_output=True,
        text=True,
        check=True,
    )
    return found.stdout.strip()


def _listener_of(port: int) -> int:
    return int(_powershell(f"(Get-NetTCPConnection -LocalPort {port} -State Listen).OwningProcess"))


def _window_of(port: int) -> str:
    """The process ID of the window run.bat opened for this port, if it is still open."""
    return _powershell(
        "(Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | "
        f"Where-Object CommandLine -Match 'gui --no-browser --port {port} ').ProcessId"
    )


def _is_free(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("NDE_CONSOLE") != "1",
    reason="set NDE_CONSOLE=1 on Windows to open a real console window",
)
def test_ctrl_c_stops_story_builder_started_by_run_bat(
    fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    port = _free_port()
    config = write_config(tmp_path, continuity_config(fake_llm.base_url))
    subprocess.run(  # noqa: S603 - this repository's own batch file, with fixed arguments
        [  # noqa: S607 - cmd is always on the Windows path
            "cmd",
            "/c",
            str(REPO / "run.bat"),
            "--no-browser",
            "--port",
            str(port),
            "--config",
            str(config),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--ledger",
            str(tmp_path / "ledger.sqlite"),
            "--ideas-dir",
            str(tmp_path / "ideas"),
        ],
        check=True,
        timeout=_LIMIT_S,
    )

    deadline = time.monotonic() + _LIMIT_S
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}", timeout=5)
            break
        except httpx.TransportError:
            time.sleep(0.05)
    else:
        raise AssertionError("Story Builder did not start")

    listener = _listener_of(port)
    try:
        subprocess.run(  # noqa: S603 - this interpreter, with a fixed program
            [sys.executable, "-c", _PRESS_CTRL_C, str(listener)],
            check=True,
            timeout=_LIMIT_S,
            creationflags=_CREATE_NO_WINDOW,
        )
        deadline = time.monotonic() + _LIMIT_S
        while time.monotonic() < deadline and not _is_free(port):
            time.sleep(0.05)
        assert _is_free(port), "Ctrl+C left Story Builder running"
        while time.monotonic() < deadline and _window_of(port):
            time.sleep(0.2)
        assert not _window_of(port), "Ctrl+C left the window open"
    finally:
        for leftover in (str(listener), _window_of(port)):
            if leftover:
                subprocess.run(  # noqa: S603 - fixed program, with a process ID from this test
                    ["taskkill", "/PID", leftover, "/F"],  # noqa: S607 - always on the path
                    capture_output=True,
                    check=False,
                )
