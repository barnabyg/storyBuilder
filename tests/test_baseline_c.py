import json
from pathlib import Path

import pytest

from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import (
    jsonl,
    last_user_message,
    manifest,
    reply_with_story,
    run,
    stories,
    story_config,
)
from tests.test_plan_generation import config, fake_plan, generate, skeleton

BASELINE_C = {"name": "C", "pipeline": "baseline_c"}


def fake_c(body: Json) -> Reply:
    if "Now write scene" in last_user_message(body):
        return reply_with_story(400)
    return fake_plan(body)


def test_c_uses_shared_prose_contract_and_saves_scene_inputs(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_c
    result = run(
        tmp_path, story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]})
    )
    [record] = stories(result)
    assert record["status"] == "ok"
    assert record["words"] == 2000
    folder = result / record["path"]
    ledger = json.loads((folder / "ledger.json").read_text())
    first = json.loads((folder / "attempts/1/scene-01.inputs.json").read_text())
    assert first["scene_id"] == "scene-0"
    assert first["plan_sha256"] == ledger["plan_sha256"]
    [*_, first_prose] = fake_llm.requests[:7]
    assert "Planned state only" in last_user_message(first_prose.body)
    calls = jsonl(result / "calls.jsonl")
    assert [c["step"] for c in calls][-5:] == [f"scene-{i:02d}" for i in range(1, 6)]
    assert {c["condition"] for c in calls} == {"C"}
    assert {c["template"] for c in calls if c["step"].startswith("scene-")} == {"scene_prose"}


def test_saved_plan_import_retains_planning_budget_without_new_planner_spend(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_c
    source = generate(tmp_path, config(fake_llm.base_url))
    [plan_record] = jsonl(source / "plans.jsonl")
    folder = source / plan_record["path"]
    condition = BASELINE_C | {
        "params": {
            "saved_plan": str(folder / "plan.json"),
            "provenance": str(folder / "planning-provenance.json"),
        }
    }
    result = run(
        tmp_path, story_config(fake_llm.base_url, [condition], briefs={"ids": ["brief-01"]})
    )
    assert fake_llm.call_count == 11
    [story] = stories(result)
    assert story["budget"]["calls"] == 11
    calls = jsonl(result / "calls.jsonl")
    assert all(c["cost_usd"] == 0 and c["imported_from"] for c in calls[:6])
    assert all(c["input_tokens"] == 10 and c["output_tokens"] == 5 for c in calls[:6])


def test_provider_failure_keeps_partial_scene_outputs_and_failed_story_record(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    count = 0

    def failing_scene(body: Json) -> Reply:
        nonlocal count
        if "Now write scene" in last_user_message(body):
            count += 1
            return error(401, "synthetic failure") if count == 2 else reply_with_story(400)
        return fake_plan(body)

    fake_llm.default = failing_scene
    result = run(
        tmp_path,
        story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]}),
        expect=1,
    )
    [record] = stories(result)
    assert record["status"] == "failed"
    assert "synthetic failure" in record["error"]
    assert (result / record["path"] / "attempts/1/scene-01.md").exists()
    assert not (result / record["path"] / "story.md").exists()
    assert record["budget"]["calls"] == 7


def rich_c(body: Json) -> Reply:
    if "whole-story skeleton planner" in str(body["messages"][0]["content"]):
        plan = skeleton()
        plan["states"] = [
            {
                "id": "trust-change",
                "scene_id": "scene-1",
                "character_id": "keeper",
                "kind": "belief",
                "key": "trust",
                "before": {"value": "suspicious"},
                "after": {"value": "trusting"},
            }
        ]
        plan["knowledge_facts"] = [{"id": "secret", "description": "the hidden key location"}]
        plan["reveal_links"] = [
            {
                "id": "reveal-secret",
                "fact_id": "secret",
                "scene_id": "scene-2",
                "holder_ids": ["keeper"],
            }
        ]
        plan["tension"] = [
            {
                "id": "tension-marker",
                "scene_ids": ["scene-1"],
                "reading": {"value": "rising dread"},
            }
        ]
        return completion(json.dumps(plan))
    return fake_c(body)


def test_each_scene_receives_correctly_timed_state_and_current_intentions(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = rich_c
    result = run(
        tmp_path, story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]})
    )
    [record] = stories(result)
    folder = result / record["path"]
    inputs = [
        json.loads((folder / f"attempts/1/scene-{i:02d}.inputs.json").read_text())
        for i in range(1, 6)
    ]
    assert "trusting" not in inputs[0]["planned_state_before"]
    assert "suspicious" in inputs[1]["planned_state_before"]
    assert inputs[1]["scene_record"]["tension"][0]["reading"]["value"] == "rising dread"
    assert "trusting" in inputs[2]["planned_state_before"]
    assert "hidden key" not in inputs[1]["planned_state_before"]
    assert "hidden key" not in inputs[2]["planned_state_before"]
    assert "hidden key" in inputs[3]["planned_state_before"]


def test_length_regeneration_resets_scene_traversal_and_uses_distinct_attempt_seeds(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    prose_calls = 0

    def short_first_draft(body: Json) -> Reply:
        nonlocal prose_calls
        if "Now write scene" in last_user_message(body):
            prose_calls += 1
            return reply_with_story(200 if prose_calls <= 5 else 400)
        return rich_c(body)

    fake_llm.default = short_first_draft
    result = run(
        tmp_path, story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]})
    )
    [record] = stories(result)
    assert [a["words"] for a in record["attempts"]] == [1000, 2000]
    assert record["attempts"][0]["seed"] != record["attempts"][1]["seed"]
    folder = result / record["path"]
    assert (folder / "attempts/1/scene-01.inputs.json").read_bytes() == (
        folder / "attempts/2/scene-01.inputs.json"
    ).read_bytes()
    requests = [r.body for r in fake_llm.requests if "Now write scene" in last_user_message(r.body)]
    assert "This is the first scene." in last_user_message(requests[0])
    assert "This is the first scene." in last_user_message(requests[5])
    assert requests[0]["seed"] != requests[5]["seed"]


@pytest.mark.parametrize("tier", ["S", "M", "L"])
def test_tier_projection_precedes_prose_notes_and_ledger(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tier: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = rich_c
    condition = BASELINE_C | {"params": {"tier": tier}}
    result = run(
        tmp_path, story_config(fake_llm.base_url, [condition], briefs={"ids": ["brief-01"]})
    )
    [record] = stories(result)
    folder = result / record["path"]
    selected = json.loads((folder / "plan.json").read_text())
    assert selected["tier"] == tier
    assert json.loads((folder / "source-plan.json").read_text())["tier"] == "L"
    requests = "\n".join(
        last_user_message(r.body)
        for r in fake_llm.requests
        if "Now write scene" in last_user_message(r.body)
    )
    assert ("rising dread" in requests) == (tier == "L")
    assert ("suspicious" in requests) == (tier != "S")
    assert record["budget"]["calls"] == 11


@pytest.mark.parametrize("change", ["plan", "schema", "prompt", "calls", "seed"])
def test_stale_or_incomplete_saved_plans_fail_before_prose_calls(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_c
    source = generate(tmp_path, config(fake_llm.base_url))
    [record] = jsonl(source / "plans.jsonl")
    folder = source / record["path"]
    source_plan = folder / "plan.json"
    provenance_file = folder / "planning-provenance.json"
    provenance = json.loads(provenance_file.read_text())
    if change == "plan":
        source_plan.write_text(source_plan.read_text().replace("The Last Lamp", "Altered title"))
    elif change == "schema":
        provenance["schema_sha256"] = "0" * 64
    elif change == "prompt":
        provenance["prompts"]["plan_skeleton@1"] = "0" * 64
    elif change == "calls":
        provenance["calls"].pop()
    else:
        provenance["seed"] += 1
    provenance_file.write_text(json.dumps(provenance))
    condition = BASELINE_C | {
        "params": {"saved_plan": str(source_plan), "provenance": str(provenance_file)}
    }
    result = run(
        tmp_path, story_config(fake_llm.base_url, [condition], briefs={"ids": ["brief-01"]})
    )
    [story] = stories(result)
    assert story["status"] == "failed"
    assert "saved-plan" in story["error"]
    assert fake_llm.call_count == 6


def test_c_cached_replay_preserves_story_plan_and_ledger_without_new_spend(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = rich_c
    value = story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]})
    first = run(tmp_path, value)
    second = run(tmp_path, value)
    [a] = stories(first)
    [b] = stories(second)
    assert fake_llm.call_count == 11
    assert manifest(second)["cost"]["usd"] == 0
    for name in ("story.md", "plan.json", "ledger.json", "attempts/1/scene-01.inputs.json"):
        assert (first / a["path"] / name).read_bytes() == (second / b["path"] / name).read_bytes()


def test_c_budget_abort_retains_failed_plan_and_story_provenance(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = story_config(fake_llm.base_url, [BASELINE_C], briefs={"ids": ["brief-01"]})
    value["budget"] = {"per_run_usd": 0}
    result = run(tmp_path, value, expect=1)
    assert manifest(result)["status"] == "aborted"
    [story] = stories(result)
    assert story["status"] == "failed"
    assert (result / story["path"] / "planning-provenance.json").exists()
    assert fake_llm.call_count == 0
