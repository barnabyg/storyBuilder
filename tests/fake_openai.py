"""A fake OpenAI-compatible chat-completions server on 127.0.0.1 for tests.

Each request takes the next scripted reply; with no script left, it answers
with a default completion. Every request body is recorded for assertions.
"""

import json
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, override

type Json = dict[str, Any]


@dataclass(frozen=True)
class Reply:
    status: int = 200
    body: Json | str = field(default_factory=dict[str, Any])
    headers: tuple[tuple[str, str], ...] = ()
    drop_connection: bool = False
    # Seconds to wait before answering, to provoke client timeouts.
    delay_s: float = 0.0


def completion(
    text: str,
    *,
    model: str = "fake-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> Reply:
    return Reply(
        body={
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def error(status: int, message: str = "error", code: str | None = None) -> Reply:
    body: Json = {"error": {"message": message, "type": "fake_error", "code": code}}
    headers = (("Retry-After", "0"),) if status == 429 else ()
    return Reply(status=status, body=body, headers=headers)


def dropped() -> Reply:
    return Reply(drop_connection=True)


@dataclass(frozen=True)
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: Json


class _Server(ThreadingHTTPServer):
    # Join handler threads on close, so a delayed reply never outlives its test.
    daemon_threads = False

    @override
    def handle_error(self, request: Any, client_address: Any) -> None:
        # Timeout and dropped-connection tests disconnect on purpose; writing a
        # late reply then fails, and the default handler prints a traceback.
        if isinstance(sys.exception(), ConnectionError):
            return
        super().handle_error(request, client_address)


class FakeOpenAI:
    def __init__(self) -> None:
        self.script: deque[Reply] = deque()
        self.requests: list[RecordedRequest] = []
        self.default: Callable[[Json], Reply] = self._echo
        self._lock = threading.Lock()
        self._server = _Server(("127.0.0.1", 0), self._handler_class())
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def queue(self, *replies: Reply) -> None:
        self.script.extend(replies)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def _echo(self, body: Json) -> Reply:
        messages: list[Json] = body.get("messages", [])
        last = str(messages[-1]["content"]) if messages else ""
        return completion(f"echo: {last}", model=str(body.get("model", "fake-model")))

    def next_reply(self, request: RecordedRequest) -> Reply:
        with self._lock:
            self.requests.append(request)
            if self.script:
                return self.script.popleft()
        return self.default(request.body)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            # Headers and body go out as separate writes; without this, Nagle plus
            # delayed ACKs stall every response on loopback by tens of milliseconds.
            disable_nagle_algorithm = True

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                request = RecordedRequest(
                    path=self.path,
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=json.loads(raw) if raw else {},
                )
                reply = fake.next_reply(request)
                if reply.delay_s:
                    time.sleep(reply.delay_s)
                if reply.drop_connection:
                    self.close_connection = True
                    self.connection.close()
                    return
                payload = (
                    reply.body if isinstance(reply.body, str) else json.dumps(reply.body)
                ).encode("utf-8")
                self.send_response(reply.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for name, value in reply.headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            @override
            def log_message(self, format: str, *args: object) -> None:
                return

        return Handler
