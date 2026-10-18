import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from nde.cli import main
from nde.llm.cache import ResponseCache
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.gateway import Gateway, Prompt
from tests.conftest import base_config, write_config
from tests.fake_openai import FakeOpenAI, completion

type GatewayFactory = Callable[[], Gateway]


@pytest.fixture
def make_gateway(fake_llm: FakeOpenAI, tmp_path: Path) -> Iterator[GatewayFactory]:
    """Gateways on one shared cache file, closed when the test ends."""
    made: list[Gateway] = []

    def make() -> Gateway:
        provider = ProviderConfig(base_url=fake_llm.base_url)
        gateway = Gateway(
            {"fake": provider, "other": provider},
            environ={},
            cache=ResponseCache(tmp_path / "cache.sqlite"),
        )
        made.append(gateway)
        return gateway

    yield make
    for gateway in made:
        gateway.close()


BINDING = ModelBinding(provider="fake", model="m", params={"temperature": 0.5})
PROMPT = Prompt("t", 1, ({"role": "user", "content": "hi"},))


def test_a_repeated_call_is_served_from_the_cache(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    gateway = make_gateway()

    first = gateway.complete(BINDING, PROMPT, seed=1)
    second = gateway.complete(BINDING, PROMPT, seed=1)

    assert fake_llm.call_count == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.text == first.text
    assert (gateway.cache_stats.hits, gateway.cache_stats.misses) == (1, 1)


@pytest.mark.parametrize(
    ("binding", "prompt", "seed"),
    [
        (BINDING.model_copy(update={"provider": "other"}), PROMPT, 1),
        (BINDING.model_copy(update={"model": "m2"}), PROMPT, 1),
        (BINDING.model_copy(update={"params": {"temperature": 0.6}}), PROMPT, 1),
        (BINDING, Prompt("t", 2, PROMPT.messages), 1),
        (BINDING, Prompt("t", 1, ({"role": "user", "content": "hello"},)), 1),
        (BINDING, PROMPT, 2),
    ],
    ids=["provider", "model", "params", "template-version", "rendered-prompt", "seed"],
)
def test_changing_any_key_component_misses_the_cache(
    fake_llm: FakeOpenAI,
    make_gateway: GatewayFactory,
    binding: ModelBinding,
    prompt: Prompt,
    seed: int,
) -> None:
    gateway = make_gateway()
    gateway.complete(BINDING, PROMPT, seed=1)

    result = gateway.complete(binding, prompt, seed=seed)

    assert not result.cached
    assert fake_llm.call_count == 2


def test_concurrent_gateways_sharing_a_cache_do_not_corrupt_it(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    prompts = [Prompt("t", 1, ({"role": "user", "content": f"p{i}"},)) for i in range(15)]
    errors: list[BaseException] = []

    def worker() -> None:
        gateway = make_gateway()
        try:
            for prompt in prompts:
                gateway.complete(BINDING, prompt, seed=1)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    calls_before = fake_llm.call_count

    reader = make_gateway()
    texts = [reader.complete(BINDING, prompt, seed=1).text for prompt in prompts]

    assert fake_llm.call_count == calls_before
    assert texts == [f"echo: p{i}" for i in range(15)]


def _run(config: Path, runs: Path, cache: Path, *flags: str) -> Path:
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    assert main(["run", str(config), "--runs-dir", str(runs), "--cache", str(cache), *flags]) == 0
    [run_dir] = set(runs.iterdir()) - before
    return run_dir


def _manifest_cache(run_dir: Path) -> dict[str, object]:
    cache: dict[str, object] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))[
        "cache"
    ]
    return cache


@pytest.fixture
def config_path(fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = base_config(fake_llm.base_url)
    config["calls"][0]["repeat"] = 2
    return write_config(tmp_path, config)


def test_rerunning_a_config_makes_no_calls_and_reproduces_outputs(
    fake_llm: FakeOpenAI, config_path: Path, tmp_path: Path
) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache.sqlite"
    first = _run(config_path, runs, cache)
    calls_after_first = fake_llm.call_count

    second = _run(config_path, runs, cache)

    assert fake_llm.call_count == calls_after_first == 2
    assert (second / "outputs.jsonl").read_bytes() == (first / "outputs.jsonl").read_bytes()
    assert _manifest_cache(first) | {"path": None} == {
        "mode": "use",
        "hits": 0,
        "misses": 2,
        "path": None,
    }
    assert _manifest_cache(second)["hits"] == 2
    assert _manifest_cache(second)["misses"] == 0


def test_no_cache_flag_bypasses_the_cache_entirely(
    fake_llm: FakeOpenAI, config_path: Path, tmp_path: Path
) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache.sqlite"
    _run(config_path, runs, cache, "--no-cache")
    bypassed = _run(config_path, runs, cache, "--no-cache")
    assert fake_llm.call_count == 4
    assert _manifest_cache(bypassed)["mode"] == "bypass"

    _run(config_path, runs, cache)

    assert fake_llm.call_count == 6, "bypassed runs must not have written the cache"


def test_refresh_cache_flag_replaces_stored_entries(
    fake_llm: FakeOpenAI, config_path: Path, tmp_path: Path
) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache.sqlite"
    _run(config_path, runs, cache)
    fake_llm.default = lambda _body: completion("fresh")

    refreshed = _run(config_path, runs, cache, "--refresh-cache")
    reused = _run(config_path, runs, cache)

    assert fake_llm.call_count == 4
    assert _manifest_cache(refreshed)["misses"] == 2
    assert "fresh" in (reused / "outputs.jsonl").read_text(encoding="utf-8")


def test_cache_defaults_to_a_gitignored_path_in_the_working_directory(
    config_path: Path, tmp_path: Path
) -> None:
    assert main(["run", str(config_path), "--runs-dir", str(tmp_path / "runs")]) == 0

    assert (tmp_path / ".nde" / "cache.sqlite").is_file()


def test_pointing_a_provider_at_another_server_misses_the_cache(tmp_path: Path) -> None:
    servers = [FakeOpenAI(), FakeOpenAI()]
    gateways: list[Gateway] = []
    try:
        for server in servers:
            server.start()
            gateway = Gateway(
                {"fake": ProviderConfig(base_url=server.base_url)},
                environ={},
                cache=ResponseCache(tmp_path / "cache.sqlite"),
            )
            gateways.append(gateway)
            gateway.complete(BINDING, PROMPT, seed=1)

        assert [server.call_count for server in servers] == [1, 1]
    finally:
        for gateway in gateways:
            gateway.close()
        for server in servers:
            server.stop()
