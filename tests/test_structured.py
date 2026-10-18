import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from nde.llm.cache import ResponseCache
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.gateway import Gateway, Prompt
from nde.llm.structured import StructuredOutputError
from nde.schema.smoke import StoryPremise
from tests.conftest import base_config, write_config
from tests.fake_openai import FakeOpenAI, completion

VALID = {
    "title": "The Last Lamp",
    "protagonist": {"name": "Ada", "goal": "keep the light burning"},
    "beats": ["storm rises", "lamp fails", "Ada climbs"],
}
# Two beats is one too few, and the protagonist has no goal.
INVALID = {"title": "The Last Lamp", "protagonist": {"name": "Ada"}, "beats": ["a", "b"]}

BINDING = ModelBinding(provider="fake", model="m")
PROMPT = Prompt("premise", 1, ({"role": "user", "content": "Write a premise."},))

type GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def make_gateway(fake_llm: FakeOpenAI, tmp_path: Path) -> Iterator[GatewayFactory]:
    made: list[Gateway] = []

    def make(*, json_schema: bool = False, cached: bool = False) -> Gateway:
        provider = ProviderConfig(base_url=fake_llm.base_url, json_schema=json_schema)
        gateway = Gateway(
            {"fake": provider},
            environ={},
            cache=ResponseCache(tmp_path / "cache.sqlite") if cached else None,
        )
        made.append(gateway)
        return gateway

    yield make
    for gateway in made:
        gateway.close()


def _sent_messages(fake: FakeOpenAI, index: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = fake.requests[index].body["messages"]
    return messages


def test_a_valid_response_is_returned_as_a_model_instance(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(completion(json.dumps(VALID)))

    result = make_gateway().structured(BINDING, PROMPT, StoryPremise, seed=1)

    assert result.value == StoryPremise.model_validate(VALID)
    assert (len(result.attempts), result.repaired) == (1, False)


def test_fenced_json_is_accepted(fake_llm: FakeOpenAI, make_gateway: GatewayFactory) -> None:
    fake_llm.queue(completion(f"```json\n{json.dumps(VALID)}\n```"))

    result = make_gateway().structured(BINDING, PROMPT, StoryPremise, seed=1)

    assert result.value.title == "The Last Lamp"


def test_validation_errors_are_sent_back_in_a_repair_request(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(completion("{not json"), completion(json.dumps(VALID)))

    result = make_gateway().structured(BINDING, PROMPT, StoryPremise, seed=1)

    assert result.value.protagonist.name == "Ada"
    assert (len(result.attempts), result.repaired) == (2, True)
    repair = _sent_messages(fake_llm, 1)
    assert repair[-2] == {"role": "assistant", "content": "{not json"}
    assert repair[-1]["role"] == "user"
    assert "Invalid JSON" in repair[-1]["content"]


def test_repair_requests_name_each_failing_field(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(completion(json.dumps(INVALID)), completion(json.dumps(VALID)))

    make_gateway().structured(BINDING, PROMPT, StoryPremise, seed=1)

    feedback = _sent_messages(fake_llm, 1)[-1]["content"]
    assert "protagonist.goal: Field required" in feedback
    assert "beats: List should have at least 3 items" in feedback


def test_exhausted_repairs_raise_a_typed_error(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.default = lambda _body: completion(json.dumps(INVALID))

    with pytest.raises(StructuredOutputError) as failure:
        make_gateway().structured(BINDING, PROMPT, StoryPremise, seed=1, max_repairs=2)

    assert fake_llm.call_count == 3
    assert len(failure.value.attempts) == 3
    assert any("protagonist.goal" in error for error in failure.value.errors)


def test_json_schema_mode_is_used_only_when_the_provider_supports_it(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.default = lambda _body: completion(json.dumps(VALID))

    make_gateway(json_schema=True).structured(BINDING, PROMPT, StoryPremise, seed=1)
    make_gateway(json_schema=False).structured(BINDING, PROMPT, StoryPremise, seed=1)

    with_mode, without_mode = (request.body for request in fake_llm.requests)
    assert with_mode["response_format"]["type"] == "json_schema"
    assert with_mode["response_format"]["json_schema"]["schema"]["title"] == "StoryPremise"
    assert with_mode["messages"] == list(PROMPT.messages)
    assert "response_format" not in without_mode
    assert '"protagonist"' in without_mode["messages"][-1]["content"]


def test_a_model_can_override_its_providers_json_schema_capability(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(completion(json.dumps(VALID)))
    binding = BINDING.model_copy(update={"json_schema": False})

    make_gateway(json_schema=True).structured(binding, PROMPT, StoryPremise, seed=1)

    assert "response_format" not in fake_llm.requests[0].body


def test_repair_attempts_are_cached_so_reruns_are_free(
    fake_llm: FakeOpenAI, make_gateway: GatewayFactory
) -> None:
    fake_llm.queue(completion("{not json"), completion(json.dumps(VALID)))
    make_gateway(cached=True).structured(BINDING, PROMPT, StoryPremise, seed=1)

    rerun = make_gateway(cached=True).structured(BINDING, PROMPT, StoryPremise, seed=1)

    assert fake_llm.call_count == 2
    assert rerun.repaired
    assert all(attempt.cached for attempt in rerun.attempts)


@pytest.fixture
def structured_config(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = base_config(fake_llm.base_url)
    config["calls"] = [
        {
            "id": "premise",
            "role": "prose",
            "prompt": {"template": "premise", "version": 1},
            "variables": {"setting": "a lighthouse"},
            "response_model": "story_premise",
            "repeat": 2,
        }
    ]
    config["structured"] = {"max_repairs": 1}
    return config


def _run(tmp_path: Path, config: dict[str, Any]) -> Path:
    runs = tmp_path / "runs"
    assert main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)]) == 0
    [run_dir] = list(runs.iterdir())
    return run_dir


def test_a_run_records_parsed_objects_and_validity_metrics_per_model(
    fake_llm: FakeOpenAI, structured_config: dict[str, Any], tmp_path: Path
) -> None:
    # Repetition 1: invalid, then repaired. Repetition 2: invalid twice, exhausted.
    fake_llm.queue(
        completion(json.dumps(INVALID)),
        completion(json.dumps(VALID)),
        completion("{not json"),
        completion("{still not json"),
    )

    run_dir = _run(tmp_path, structured_config)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    metrics = manifest["structured"]["fake/fake-model"]
    assert metrics == {
        "calls": 2,
        "valid_first_attempt": 0,
        "valid_after_repair": 1,
        "repaired_calls": 2,
        "failed_calls": 1,
        "first_attempt_validity": 0.0,
        "validity_after_repair": 0.5,
        "repair_rate": 1.0,
    }
    outputs = [
        json.loads(line)
        for line in (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert outputs[0]["parsed"] == VALID
    assert outputs[1]["parsed"] is None
    assert "Invalid JSON" in outputs[1]["error"]
    calls = (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["attempt"] for line in calls] == [0, 1, 0, 1]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Structured output" in report
    assert '"keep the light burning"' in report
    assert "| fake/fake-model | 2 | 0.0% | 50.0% | 100.0% | 1 |" in report


def test_an_unknown_response_model_is_a_config_error(
    fake_llm: FakeOpenAI,
    structured_config: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structured_config["calls"][0]["response_model"] = "nope"

    code = main(["run", str(write_config(tmp_path, structured_config)), "--runs-dir", "runs"])

    assert code == 2
    assert "unknown response model 'nope'" in capsys.readouterr().err
    assert fake_llm.call_count == 0
