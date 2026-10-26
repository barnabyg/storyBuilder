import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO, base_config, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import jsonl, manifest


def skeleton() -> dict[str, Any]:
    return {
        "id": "generated-plan",
        "tier": "L",
        "title": "The Last Lamp",
        "characters": [{"id": "keeper", "name": "Mara"}],
        "scenes": [
            {
                "id": f"scene-{i}",
                "title": f"Scene {i}",
                "summary": f"Function {i}",
                "character_ids": ["keeper"],
                "beats": [{"id": f"beat-{i}", "summary": f"Event {i}"}],
            }
            for i in range(5)
        ],
        "arcs": [
            {
                "id": "arc-keeper",
                "character_id": "keeper",
                "scene_ids": ["scene-0", "scene-4"],
                "reading": {"value": "Accepts responsibility"},
            }
        ],
    }


def fake_plan(body: Json) -> Reply:
    text = str(body["messages"][0]["content"])
    if "whole-story skeleton planner" in text:
        return completion(json.dumps(skeleton()))
    request = str(body["messages"][-1]["content"])
    scene_id = request.split("CURRENT_SCENE_ID=", 1)[1].splitlines()[0]
    plan = json.loads(request.split("Accepted plan:\n", 1)[1].split("\nCURRENT_SCENE_ID=", 1)[0])
    plan["scenes"][int(scene_id.rsplit("-", 1)[1])]["summary"] += " realised intention"
    return completion(json.dumps(plan))


def config(base_url: str) -> dict[str, Any]:
    value = base_config(base_url)
    value.pop("calls")
    value["roles"] = {"planner": {"provider": "fake", "model": "fake-model"}}
    value["plans"] = {
        "briefs_dir": str(REPO / "benchmark"),
        "briefs": {"ids": ["brief-01"]},
    }
    return value


def generate(tmp_path: Path, value: dict[str, Any], expected: int = 0) -> Path:
    runs = tmp_path / "runs"
    before: set[Path] = set(runs.iterdir()) if runs.exists() else set()
    assert (
        main(
            [
                "plan",
                "generate",
                str(write_config(tmp_path, value)),
                "--runs-dir",
                str(runs),
                "--cache",
                str(tmp_path / "cache.sqlite"),
                "--ledger",
                str(tmp_path / "ledger.sqlite"),
            ]
        )
        == expected
    )
    [result] = set(runs.iterdir()) - before
    return result


def test_generate_writes_validated_multistage_plan_and_registry(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_plan
    run = generate(tmp_path, config(fake_llm.base_url))
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    plan = json.loads((folder / "plan.json").read_text())
    assert plan["id"] == "generated-plan"
    assert [s["id"] for s in plan["scenes"]] == [f"scene-{i}" for i in range(5)]
    assert plan["sources"] == []
    assert (folder / "cards.md").exists()
    assert (folder / "cards.html").exists()
    assert not [
        d
        for d in json.loads((folder / "validation.json").read_text())["diagnostics"]
        if d["severity"] == "error"
    ]
    assert fake_llm.call_count == 6
    assert len(jsonl(run / "calls.jsonl")) == 6
    assert manifest(run)["status"] == "succeeded"
    assert manifest(run)["stories"] is None


def test_schema_and_semantic_repairs_share_a_bounded_loop_and_keep_originals(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_plan
    broken = skeleton()
    broken["scenes"][0]["character_ids"] = ["missing-person"]
    fake_llm.queue(completion("not JSON"), completion(json.dumps(broken)))
    run = generate(tmp_path, config(fake_llm.base_url))
    assert "Invalid JSON" in str(fake_llm.requests[1].body["messages"][-1]["content"])
    assert "character_reference" in str(fake_llm.requests[2].body["messages"][-1]["content"])
    [record] = jsonl(run / "plans.jsonl")
    folder = run / record["path"]
    assert (folder / "planning/skeleton/1.txt").read_text() == "not JSON"
    assert len(jsonl(run / "calls.jsonl")) == 8
    assert manifest(run)["structured"]["fake/fake-model"]["repaired_calls"] == 1


def test_explicit_locked_constraints_enter_semantic_repair(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_plan
    broken = skeleton()
    broken["states"] = [
        {
            "id": "change-canon",
            "scene_id": "scene-0",
            "kind": "canon",
            "key": "lamp",
            "before": {"value": "lit"},
            "after": {"value": "dark"},
        }
    ]
    fake_llm.queue(completion(json.dumps(broken)))
    value = config(fake_llm.base_url)
    value["plans"]["params"] = {
        "constraints": [
            {
                "id": "locked-lamp",
                "kind": "locked",
                "key": "lamp",
                "expected": "lit",
            }
        ]
    }
    result = generate(tmp_path, value)
    assert "locked" in str(fake_llm.requests[1].body["messages"][-1]["content"])
    assert manifest(result)["status"] == "succeeded"


@pytest.mark.parametrize("failure", ["schema", "semantic", "evidence", "identity"])
def test_exhausted_repairs_remain_failed_artifacts(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = config(fake_llm.base_url)
    value["structured"] = {"max_repairs": 1}
    broken = skeleton()
    if failure == "semantic":
        broken["scenes"][0]["character_ids"] = ["nobody"]
    elif failure == "evidence":
        broken["sources"] = [
            {
                "id": "invented",
                "title": "fiction",
                "text": "x",
                "sha256": "0" * 64,
                "provenance": "fabricated",
            }
        ]
    elif failure == "identity":
        fake_llm.queue(completion(json.dumps(skeleton())))
        broken["id"] = "replacement"
    fake_llm.default = lambda _body: completion(
        "bad JSON" if failure == "schema" else json.dumps(broken)
    )
    run = generate(tmp_path, value, expected=1)
    [record] = jsonl(run / "plans.jsonl")
    assert record["status"] == "failed"
    folder = run / record["path"]
    assert not (folder / "plan.json").exists()
    assert json.loads((folder / "planning-provenance.json").read_text())["status"] == "failed"
    assert fake_llm.call_count == (3 if failure == "identity" else 2)


def test_provider_failure_and_budget_abort_preserve_run_and_attempt_provenance(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.queue(error(401, "test authentication error"))
    failed = generate(tmp_path, config(fake_llm.base_url), expected=1)
    assert manifest(failed)["error"]["type"] == "authentication"
    [record] = jsonl(failed / "plans.jsonl")
    assert record["status"] == "failed"
    assert (failed / record["path"] / "planning-provenance.json").exists()
    value = config(fake_llm.base_url)
    value["budget"] = {"per_run_usd": 0}
    aborted = generate(tmp_path, value, expected=1)
    assert manifest(aborted)["status"] == "aborted"
    assert fake_llm.call_count == 1


def test_cache_replay_reproduces_plans_at_zero_new_spend_and_changed_input_misses(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_plan
    value = config(fake_llm.base_url)
    first = generate(tmp_path, value)
    second = generate(tmp_path, value)
    [a] = jsonl(first / "plans.jsonl")
    [b] = jsonl(second / "plans.jsonl")
    assert (first / a["path"] / "plan.json").read_bytes() == (
        second / b["path"] / "plan.json"
    ).read_bytes()
    assert fake_llm.call_count == 6
    assert manifest(second)["cost"]["usd"] == 0
    assert manifest(second)["cost"]["input_tokens"] == manifest(first)["cost"]["input_tokens"]
    value["seed"] += 1
    generate(tmp_path, value)
    assert fake_llm.call_count == 12


def test_held_out_plan_generation_is_rejected_before_calls(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = config(fake_llm.base_url)
    value["plans"]["briefs"] = {"ids": ["brief-11"]}
    assert main(["plan", "generate", str(write_config(tmp_path, value))]) == 2
    assert fake_llm.call_count == 0
