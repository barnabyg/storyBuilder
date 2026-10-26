import json
from pathlib import Path

import pytest

from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import (
    jsonl,
    last_user_message,
    manifest,
    reply_with_story,
    run,
    stories,
    story_config,
)
from tests.test_plan_generation import fake_plan, skeleton

BASELINE_D = {"name": "D", "pipeline": "baseline_d"}


def d_plan() -> dict[str, object]:
    plan = skeleton()
    plan["reader_affect"] = [
        {
            "id": "hope",
            "scene_ids": ["scene-0", "scene-2", "scene-4"],
            "reading": {"value": "Renewed hope"},
            "trajectory": {
                "shape": "valley",
                "points": [
                    {"scene_id": "scene-0", "value": 3},
                    {"scene_id": "scene-2", "value": 1},
                    {"scene_id": "scene-4", "value": 4},
                ],
            },
        }
    ]
    return plan


def fake_d(body: Json) -> Reply:
    system = str(body["messages"][0]["content"])
    if "global intentions planner" in system:
        plan = d_plan()
        return completion(
            json.dumps(
                {
                    "id": plan["id"],
                    "title": plan["title"],
                    "characters": plan["characters"],
                    "scene_ids": [f"scene-{i}" for i in range(5)],
                    "arcs": plan["arcs"],
                    "reader_affect": plan["reader_affect"],
                }
            )
        )
    if "coarse-conditioned skeleton planner" in system:
        plan = d_plan()
        plan["constraints"] = json.loads(
            str(body["messages"][1]["content"])
            .split("Authoritative constraints:\n", 1)[1]
            .split("\nAccepted global intentions:", 1)[0]
        )
        return completion(json.dumps(plan))
    if "Now write scene" in last_user_message(body):
        return reply_with_story(400)
    return fake_plan(body)


def test_d_conditions_each_stage_and_reuses_prose_with_free_cached_replay(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_d
    config = story_config(fake_llm.base_url, [BASELINE_D], briefs={"ids": ["brief-01"]})
    first = run(tmp_path, config)
    [story] = stories(first)
    folder = first / story["path"]
    assert story["status"] == "ok"
    assert story["words"] == 2000
    intentions = json.loads((folder / "intentions.json").read_text())
    saved = json.loads((folder / "plan.json").read_text())
    assert saved["reader_affect"] == intentions["reader_affect"]
    assert saved["arcs"] == intentions["arcs"]
    assert (folder / "cards.html").exists()
    assert (folder / "ledger.json").exists()
    assert "Renewed hope" in last_user_message(fake_llm.requests[1].body)
    assert "target_words" in last_user_message(fake_llm.requests[0].body)
    assert len(jsonl(first / "calls.jsonl")) == 12
    provenance = json.loads((folder / "planning-provenance.json").read_text())
    assert provenance["pipeline"] == "baseline_d"
    assert len(provenance["calls"]) == 7
    second = run(tmp_path, config)
    [replayed] = stories(second)
    assert fake_llm.call_count == 12
    assert manifest(second)["cost"]["usd"] == 0
    assert (folder / "plan.json").read_bytes() == (
        second / replayed["path"] / "plan.json"
    ).read_bytes()


@pytest.mark.parametrize("attack", ["constraints", "global", "schema"])
def test_d_rejects_changed_higher_level_records_and_preserves_partial_artifacts(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    attack: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")

    def adversary(body: Json) -> Reply:
        reply = fake_d(body)
        if "coarse-conditioned skeleton planner" in str(body["messages"][0]["content"]):
            if attack == "schema":
                return completion("invalid JSON")
            assert isinstance(reply.body, dict)
            raw = json.loads(reply.body["choices"][0]["message"]["content"])
            if attack == "constraints":
                raw["constraints"] = []
            else:
                raw["reader_affect"][0]["reading"]["value"] = "Unrelated direction"
            return completion(json.dumps(raw))
        return reply

    fake_llm.default = adversary
    value = story_config(fake_llm.base_url, [BASELINE_D], briefs={"ids": ["brief-01"]})
    value["structured"] = {"max_repairs": 1}
    result = run(tmp_path, value)
    [story] = stories(result)
    folder = result / story["path"]
    assert story["status"] == "failed"
    assert not (folder / "story.md").exists()
    assert not (folder / "plan.json").exists()
    assert (folder / "intentions.json").exists()
    assert (folder / "planning/skeleton/2.txt").exists()
    assert fake_llm.call_count == 3


def test_d_budget_abort_keeps_stage_provenance(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_d
    value = story_config(fake_llm.base_url, [BASELINE_D], briefs={"ids": ["brief-01"]})
    value["budget"] = {"per_run_usd": 0}
    result = run(tmp_path, value, expect=1)
    [story] = stories(result)
    assert manifest(result)["status"] == "aborted"
    assert (result / story["path"] / "planning-provenance.json").exists()
    assert fake_llm.call_count == 0
