"""Stopping `nde gui` with Ctrl+C, in a real process: the exit must be calm and prompt."""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from tests.conftest import write_config
from tests.continuity_support import continuity_config, fake_continuity
from tests.fake_openai import FakeOpenAI, Json, Reply
from tests.test_gui import IDEA

# Runs the command, and sends it the signal Ctrl+C sends when the trigger file appears.
_PROGRAM = """
import pathlib, signal, sys, threading, time
from nde.cli import main

def press_ctrl_c(trigger):
    while not trigger.exists():
        time.sleep(0.02)
    signal.raise_signal(signal.SIGINT)

threading.Thread(target=press_ctrl_c, args=(pathlib.Path(sys.argv[1]),), daemon=True).start()
sys.exit(main(sys.argv[2:]))
"""
_EXIT_LIMIT_S = 30


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class _Gui:
    def __init__(self, tmp_path: Path, fake: FakeOpenAI) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.trigger = tmp_path / "press-ctrl-c"
        config = write_config(tmp_path, continuity_config(fake.base_url))
        self.process = subprocess.Popen(  # noqa: S603 - this interpreter, with fixed arguments
            [
                sys.executable,
                "-c",
                _PROGRAM,
                str(self.trigger),
                "gui",
                "--no-browser",
                "--port",
                str(self.port),
                "--config",
                str(config),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--ledger",
                str(tmp_path / "ledger.sqlite"),
                "--ideas-dir",
                str(tmp_path / "ideas"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            # pragma: allowlist nextline secret - a fake key for the fake server
            env=os.environ | {"FAKE_API_KEY": "test-key"},
        )

    def wait_until_up(self) -> None:
        deadline = time.monotonic() + _EXIT_LIMIT_S
        while time.monotonic() < deadline:
            try:
                httpx.get(self.url, timeout=5)
            except httpx.TransportError:
                time.sleep(0.05)
            else:
                return
        raise AssertionError("the GUI did not start")

    def press_ctrl_c(self) -> tuple[int, str, str]:
        started = time.monotonic()
        self.trigger.touch()
        out, err = self.process.communicate(timeout=_EXIT_LIMIT_S)
        assert time.monotonic() - started < _EXIT_LIMIT_S
        return self.process.returncode, out, err


@pytest.fixture
def gui(tmp_path: Path, fake_llm: FakeOpenAI) -> _Gui:
    fake_llm.default = fake_continuity()
    return _Gui(tmp_path, fake_llm)


def _assert_calm(code: int, err: str) -> None:
    assert code == 0, err
    assert "Traceback" not in err
    assert "KeyboardInterrupt" not in err
    assert "Error" not in err


def test_ctrl_c_stops_with_one_short_message_and_frees_the_port(gui: _Gui) -> None:
    gui.wait_until_up()

    code, out, err = gui.press_ctrl_c()

    _assert_calm(code, err)
    assert out.splitlines()[-1] == "Story Builder has stopped."
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", gui.port))


def test_ctrl_c_while_writing_says_the_story_was_not_finished(
    gui: _Gui, fake_llm: FakeOpenAI
) -> None:
    test_over = threading.Event()

    def waits_for_the_test(_: Json) -> Reply:
        test_over.wait(timeout=_EXIT_LIMIT_S)
        return Reply()

    fake_llm.default = waits_for_the_test
    gui.wait_until_up()
    assert httpx.post(f"{gui.url}/stories", data=IDEA, timeout=30).status_code == 303

    try:
        code, out, err = gui.press_ctrl_c()
    finally:
        test_over.set()

    _assert_calm(code, err)
    assert out.splitlines()[-1] == (
        "Story Builder has stopped. The story being written was not finished."
    )
