import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.llm.config import ModelBinding, ProviderConfig, RetryConfig
from nde.llm.errors import (
    AuthenticationError,
    ContentRefusedError,
    InvalidRequestError,
    ProviderError,
    RateLimitedError,
    TransientError,
)
from nde.llm.gateway import Gateway, Prompt
from tests.conftest import base_config, write_config
from tests.fake_openai import FakeOpenAI, Reply, completion, dropped, error

BINDING = ModelBinding(provider="fake", model="m")
PROMPT = Prompt("t", 1, ({"role": "user", "content": "hi"},))

type GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def make_gateway(fake_llm: FakeOpenAI, sleeps: list[float]) -> Iterator[GatewayFactory]:
    made: list[Gateway] = []

    def make(
        *, max_retries: int = 3, timeout_s: float = 5.0, base_url: str | None = None
    ) -> Gateway:
        provider = ProviderConfig(
            base_url=base_url or fake_llm.base_url,
            timeout_s=timeout_s,
            retry=RetryConfig(max_retries=max_retries, base_delay_s=1.0, max_delay_s=8.0),
        )
        gateway = Gateway({"fake": provider}, environ={}, sleep=sleeps.append)
        made.append(gateway)
        return gateway

    yield make
    for gateway in made:
        gateway.close()


def test_transient_failures_are_retried_until_success(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory, sleeps: list[float]
) -> None:
    fake_llm.queue(error(503), error(429), dropped(), completion("finally"))

    result = make_gateway().complete(BINDING, PROMPT, seed=1)

    assert result.text == "finally"
    assert result.retries == 3
    assert fake_llm.call_count == 4
    assert len(sleeps) == 3


def test_backoff_grows_exponentially_with_jitter_and_a_ceiling(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory, sleeps: list[float]
) -> None:
    fake_llm.default = lambda _body: error(500)

    with pytest.raises(TransientError):
        make_gateway(max_retries=5).complete(BINDING, PROMPT, seed=1)

    # Full jitter: retry n waits a random time up to min(8, 1 * 2**n) seconds.
    assert len(sleeps) == 5
    for retry, delay in enumerate(sleeps):
        assert 0 <= delay <= min(8.0, 2.0**retry)


def test_a_rate_limit_waits_at_least_as_long_as_retry_after(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory, sleeps: list[float]
) -> None:
    fake_llm.queue(
        Reply(
            status=429, body={"error": {"message": "slow down"}}, headers=(("Retry-After", "3"),)
        ),
        completion("ok"),
    )

    make_gateway().complete(BINDING, PROMPT, seed=1)

    assert sleeps[0] >= 3.0


def test_retries_are_bounded(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory, sleeps: list[float]
) -> None:
    fake_llm.default = lambda _body: error(429)

    with pytest.raises(RateLimitedError):
        make_gateway(max_retries=2).complete(BINDING, PROMPT, seed=1)

    assert fake_llm.call_count == 3
    assert len(sleeps) == 2


def test_timeouts_are_transient_and_retried(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(Reply(delay_s=0.5, body={}), completion("after timeout"))

    result = make_gateway(timeout_s=0.1).complete(BINDING, PROMPT, seed=1)

    assert result.text == "after timeout"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (error(401), AuthenticationError),
        (error(403), AuthenticationError),
        (error(400, "bad parameter"), InvalidRequestError),
        (error(404, "model not found"), InvalidRequestError),
        (error(400, "flagged", code="content_filter"), ContentRefusedError),
        (completion("", finish_reason="content_filter"), ContentRefusedError),
    ],
    ids=["401", "403", "400", "404", "400-content-filter", "finish-content-filter"],
)
def test_permanent_errors_are_mapped_and_not_retried(
    fake_llm: FakeOpenAI,
    make_gateway: GatewayFactory,
    sleeps: list[float],
    reply: Reply,
    expected: type[ProviderError],
) -> None:
    fake_llm.queue(reply)

    with pytest.raises(expected):
        make_gateway().complete(BINDING, PROMPT, seed=1)

    assert fake_llm.call_count == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (error(429), RateLimitedError),
        (error(500), TransientError),
        (error(502), TransientError),
        (error(503), TransientError),
        (error(504), TransientError),
        (error(408), TransientError),
        (dropped(), TransientError),
        (Reply(body="not json"), TransientError),
    ],
    ids=["429", "500", "502", "503", "504", "408", "dropped", "malformed"],
)
def test_retryable_errors_surface_as_their_common_type_when_retries_run_out(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory, reply: Reply, expected: type[ProviderError]
) -> None:
    fake_llm.default = lambda _body: reply

    with pytest.raises(expected):
        make_gateway(max_retries=1).complete(BINDING, PROMPT, seed=1)

    assert fake_llm.call_count == 2


def _closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def test_a_refused_connection_is_transient(make_gateway: GatewayFactory) -> None:
    with pytest.raises(TransientError):
        make_gateway(max_retries=0, base_url=_closed_port_url()).complete(BINDING, PROMPT, seed=1)


@pytest.fixture
def failing_config(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = base_config(fake_llm.base_url)
    config["providers"]["fake"]["retry"] = {"max_retries": 1, "base_delay_s": 0}
    return config


def _run(tmp_path: Path, config: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
    runs = tmp_path / "runs"
    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])
    [run_dir] = list(runs.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return code, manifest, (run_dir / "report.md").read_text(encoding="utf-8")


def test_a_permanent_failure_is_recorded_instead_of_crashing(
    fake_llm: FakeOpenAI,
    failing_config: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_llm.queue(error(401, "bad key"))

    code, manifest, report = _run(tmp_path, failing_config)

    assert code == 1
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "authentication"
    assert manifest["error"]["call"] == "line"
    assert "failed: authentication" in report
    assert "Traceback" not in capsys.readouterr().err


def test_an_unreachable_server_is_recorded_as_a_transient_failure(
    failing_config: dict[str, Any], tmp_path: Path
) -> None:
    failing_config["providers"]["fake"]["base_url"] = _closed_port_url()
    # Windows takes about two seconds to report a refused connection; do not retry it.
    failing_config["providers"]["fake"]["retry"]["max_retries"] = 0

    code, manifest, _ = _run(tmp_path, failing_config)

    assert code == 1
    assert (manifest["status"], manifest["error"]["type"]) == ("failed", "transient")


def test_a_failed_run_keeps_the_outputs_it_completed(
    fake_llm: FakeOpenAI, failing_config: dict[str, Any], tmp_path: Path
) -> None:
    failing_config["calls"][0]["repeat"] = 3
    fake_llm.queue(completion("one"), error(400, "bad"))

    code, manifest, report = _run(tmp_path, failing_config)

    assert code == 1
    assert manifest["error"]["type"] == "invalid_request"
    assert "> one" in report
