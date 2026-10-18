"""A live, localhost-only dashboard for `nde-verify`.

It only observes. `DashboardState` folds events into a snapshot. Events come
from the gate runner (through `DashboardState.observer()`) and from the pytest
reporter plugin in the tests gate (POSTed to `/api/events`). `Dashboard` serves
the page at `/`, the snapshot at `/api/state` and accepts events at
`/api/events`, on 127.0.0.1 only.
"""

import json
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, override

from nde.verify.runner import Gate, GateObserver, GateResult, RunReport

DEFAULT_PORT = 8765
HOST = "127.0.0.1"
_MAX_OUTPUT_LINES = 400
_MAX_EVENT_BYTES = 64 * 1024
_HTTP_OK = 200
_HTTP_NO_CONTENT = 204
_HTTP_BAD_REQUEST = 400
_HTTP_NOT_FOUND = 404

type Event = Mapping[str, Any]


class DashboardState:
    """The dashboard's view of one verification run. Thread-safe."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_output_lines: int = _MAX_OUTPUT_LINES,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._started: float | None = None
        self._finished: float | None = None
        self._started_at: str | None = None
        self._status = "waiting"
        self._gates: list[dict[str, Any]] = []
        self._active: str | None = None
        self._tests: dict[str, Any] = _empty_tests()
        self._output: deque[str] = deque(maxlen=max_output_lines)
        self._version = 0

    def observer(self) -> GateObserver:
        return _StateObserver(self)

    def apply(self, event: Event) -> None:
        handlers: dict[str, Callable[[Event], None]] = {
            "run_started": self._run_started,
            "gate_started": self._gate_started,
            "gate_finished": self._gate_finished,
            "output": self._output_line,
            "tests_collected": self._tests_collected,
            "test_started": self._test_started,
            "test_finished": self._test_finished,
            "run_finished": self._run_finished,
        }
        handler = handlers.get(str(event.get("type")))
        if handler is None:
            raise ValueError(f"unknown event type {event.get('type')!r}")
        with self._lock:
            handler(event)
            self._version += 1

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            end = self._finished if self._finished is not None else self._clock()
            elapsed = 0.0 if self._started is None else end - self._started
            return {
                "status": self._status,
                "started_at": self._started_at,
                "elapsed_s": round(elapsed, 3),
                "active_gate": self._active,
                "gates": [dict(gate) for gate in self._gates],
                "tests": {**self._tests, "failures": list(self._tests["failures"])},
                "output": list(self._output),
                "failures": [
                    {"gate": gate["name"], "status": gate["status"]}
                    for gate in self._gates
                    if gate["status"] in {"failed", "warning"}
                ],
                "version": self._version,
            }

    # Event handlers. Each runs with the lock held.

    def _run_started(self, event: Event) -> None:
        self._started = self._clock()
        self._started_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._status = "running"
        self._gates = [
            {"name": str(name), "status": "pending", "seconds": None} for name in event["gates"]
        ]

    def _gate(self, name: str) -> dict[str, Any]:
        for gate in self._gates:
            if gate["name"] == name:
                return gate
        gate: dict[str, Any] = {"name": name, "status": "pending", "seconds": None}
        self._gates.append(gate)
        return gate

    def _gate_started(self, event: Event) -> None:
        self._active = str(event["gate"])
        self._gate(self._active)["status"] = "running"
        if self._active == "tests":
            self._tests = _empty_tests()

    def _gate_finished(self, event: Event) -> None:
        gate = self._gate(str(event["gate"]))
        gate["status"] = str(event["status"])
        gate["seconds"] = float(event["seconds"])
        self._active = None

    def _output_line(self, event: Event) -> None:
        self._output.append(str(event["line"]))

    def _tests_collected(self, event: Event) -> None:
        self._tests["collected"] = int(event["count"])

    def _test_started(self, event: Event) -> None:
        self._tests["current"] = str(event["nodeid"])

    def _test_finished(self, event: Event) -> None:
        outcome = str(event["outcome"])
        self._tests["done"] += 1
        if outcome in {"passed", "failed", "skipped"}:
            self._tests[outcome] += 1
        if outcome == "failed":
            self._tests["failures"].append(str(event["nodeid"]))
        self._tests["current"] = None

    def _run_finished(self, event: Event) -> None:
        self._finished = self._clock()
        self._status = "passed" if event["passed"] else "failed"
        self._active = None


def _empty_tests() -> dict[str, Any]:
    return {
        "collected": None,
        "done": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "current": None,
        "failures": [],
    }


class _StateObserver:
    """Turns the gate runner's callbacks into dashboard events."""

    def __init__(self, state: DashboardState) -> None:
        self._state = state

    def run_started(self, gates: Sequence[Gate]) -> None:
        self._state.apply({"type": "run_started", "gates": [gate.name for gate in gates]})

    def gate_started(self, gate: Gate) -> None:
        self._state.apply({"type": "gate_started", "gate": gate.name})

    def output(self, line: str) -> None:
        self._state.apply({"type": "output", "line": line})

    def gate_finished(self, result: GateResult) -> None:
        self._state.apply(
            {
                "type": "gate_finished",
                "gate": result.gate,
                "status": result.status,
                "seconds": result.seconds,
            }
        )

    def run_finished(self, report: RunReport) -> None:
        self._state.apply({"type": "run_finished", "passed": report.passed})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is already in use,
    # which would hide concurrent runs from each other; bind exclusively instead.
    allow_reuse_address = sys.platform != "win32"

    @override
    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _page() -> bytes:
    return resources.files("nde.verify").joinpath("dashboard.html").read_bytes()


class Dashboard:
    """Serves a DashboardState on 127.0.0.1.

    It binds `preferred_port` when free, and otherwise any free port, so
    concurrent runs never collide; `url` is the address actually used.
    """

    def __init__(self, state: DashboardState, *, preferred_port: int = DEFAULT_PORT) -> None:
        self.state = state
        self._preferred_port = preferred_port
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._viewed_version = -1
        self._viewed = threading.Condition()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("dashboard is not running")
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/"

    @property
    def events_url(self) -> str:
        return f"{self.url}api/events"

    def start(self) -> str:
        handler = self._handler_class()
        try:
            self._server = _Server((HOST, self._preferred_port), handler)
        except OSError:
            self._server = _Server((HOST, 0), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def linger(self, timeout_s: float) -> None:
        """Wait until a viewer has fetched the current state, if anyone is watching."""
        with self._viewed:
            if self._viewed_version < 0:
                return
            target = self.state.version
            self._viewed.wait_for(lambda: self._viewed_version >= target, timeout=timeout_s)

    def _mark_viewed(self, version: int) -> None:
        with self._viewed:
            self._viewed_version = max(self._viewed_version, version)
            self._viewed.notify_all()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            disable_nagle_algorithm = True

            def _send(self, status: int, content_type: str | None, body: bytes) -> None:
                self.send_response(status)
                if content_type:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path in {"/", "/index.html"}:
                    self._send(_HTTP_OK, "text/html; charset=utf-8", _page())
                elif self.path == "/api/state":
                    snapshot = dashboard.state.snapshot()
                    self._send(_HTTP_OK, "application/json", json.dumps(snapshot).encode())
                    dashboard._mark_viewed(int(snapshot["version"]))
                else:
                    self._send(_HTTP_NOT_FOUND, "text/plain", b"not found")

            def do_POST(self) -> None:
                if self.path != "/api/events":
                    self._send(_HTTP_NOT_FOUND, "text/plain", b"not found")
                    return
                length = min(int(self.headers.get("Content-Length", "0")), _MAX_EVENT_BYTES)
                try:
                    event: object = json.loads(self.rfile.read(length))
                    if not isinstance(event, dict):
                        raise ValueError("an event is a JSON object")
                    dashboard.state.apply(event)  # pyright: ignore[reportUnknownArgumentType]
                except (ValueError, KeyError, TypeError) as error:
                    self._send(_HTTP_BAD_REQUEST, "text/plain", str(error).encode())
                    return
                self._send(_HTTP_NO_CONTENT, None, b"")

            @override
            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler
