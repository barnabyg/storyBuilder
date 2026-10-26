import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import write_config
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


@pytest.mark.parametrize("family", [None, "generator"])
def test_critic_family_rule_is_checked_before_any_generation(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:

    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    value = story_config(
        fake_llm.base_url,
        [BASELINE_D | {"params": {"critic_iterations": 1}}],
        briefs={"ids": ["brief-01"]},
    )
    for binding in value["roles"].values():
        binding["family"] = "generator"
    value["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "family": family}
    assert main(["run", str(write_config(tmp_path, value))]) == 2
    assert "critic family" in capsys.readouterr().err
    assert fake_llm.call_count == 0


def critic_config(base_url: str, iterations: int = 1) -> dict[str, Any]:
    value = story_config(
        base_url,
        [
            BASELINE_D
            | {"params": {"critic_iterations": iterations, "revision_locks": ["scenes/scene-0"]}}
        ],
        briefs={"ids": ["brief-01"]},
    )
    for binding in value["roles"].values():
        binding["family"] = "generator"
    value["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "family": "critic"}
    value["structured"] = {"max_repairs": 1}
    return value


def critic_provider(body: Json, behavior: str = "accepted") -> Reply:
    request = str(body["messages"][1]["content"])
    system = str(body["messages"][0]["content"])
    if "structural revision critic" in system:
        checksum = request.split("Plan SHA-256: ", 1)[1].splitlines()[0]
        if behavior == "no_issues":
            return completion('{"issues": []}')
        issue = {
            "plan_sha256": "stale" if behavior == "stale" else checksum,
            "mask": {
                "units": [
                    "scenes/missing/summary"
                    if behavior == "unknown"
                    else "scenes/scene-0/summary"
                    if behavior == "locked"
                    else "scenes/scene-2/summary"
                ]
            },
            "proposed_change": "Add a concrete reversable decision",
            "rationale": "Scene lacks a decision",
        }
        if behavior == "span":
            issue["mask"] = {"scene_ids": ["scene-1", "scene-2"]}
        return completion(
            json.dumps({"issues": [issue, issue] if behavior == "contradictory" else [issue]})
        )
    if "masked structural revision planner" in system:
        if behavior == "provider":
            return error(401, "Synthetic refinement failure")
        plan = json.loads(
            request.split("Accepted plan:\n", 1)[1].split("\n\nExplicit revision authority", 1)[0]
        )
        if behavior != "noop":
            plan["scenes"][2]["summary"] = "Mara decides to relight the lamp despite the cost"
        if behavior == "attack":
            plan["scenes"][0]["summary"] = "Locked scene replaced"
        return completion(json.dumps(plan))
    return fake_d(body)


def test_critic_accepts_only_targeted_change_and_cache_keeps_all_experimental_calls(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = critic_provider
    config = critic_config(fake_llm.base_url)
    first = run(tmp_path, config)
    [story] = stories(first)
    folder = first / story["path"]
    saved = json.loads((folder / "plan.json").read_text())
    assert saved["scenes"][2]["summary"] == "Mara decides to relight the lamp despite the cost"
    assert saved["scenes"][0]["summary"] == "Function 0 realised intention"
    assert (
        json.loads((folder / "refinement.json").read_text())["iterations"][0]["status"]
        == "accepted"
    )
    calls = jsonl(first / "calls.jsonl")
    assert [c["step"] for c in calls][7:9] == ["critic-1", "revision-critic-1"]
    assert story["budget"]["calls"] == 14
    assert len(json.loads((folder / "planning-provenance.json").read_text())["calls"]) == 9
    second = run(tmp_path, config)
    assert fake_llm.call_count == 14
    assert manifest(second)["cost"]["usd"] == 0


@pytest.mark.parametrize(
    "case",
    [
        ("stale", "no_actionable_issues", 13),
        ("unknown", "no_actionable_issues", 13),
        ("locked", "no_actionable_issues", 13),
        ("contradictory", "no_actionable_issues", 13),
        ("no_issues", "no_actionable_issues", 13),
        ("noop", "noop", 14),
        ("attack", "repeated_failure", 18),
    ],
)
def test_critic_rejections_noops_and_failures_are_bounded_and_keep_last_plan(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: tuple[str, str, int],
) -> None:
    behavior, reason, calls = case
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda body: critic_provider(body, behavior)
    result = run(tmp_path, critic_config(fake_llm.base_url, iterations=3))
    [story] = stories(result)
    folder = result / story["path"]
    history = json.loads((folder / "refinement.json").read_text())
    assert history["stopping_reason"] == reason
    assert len(history["iterations"]) <= 2
    assert fake_llm.call_count == calls
    plan = json.loads((folder / "plan.json").read_text())
    assert plan["scenes"][0]["summary"] == "Function 0 realised intention"
    if behavior != "noop":
        assert plan["scenes"][2]["summary"] == "Function 2 realised intention"


def test_interrupted_refinement_keeps_last_plan_and_no_successful_story(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda body: critic_provider(body, "provider")
    result = run(tmp_path, critic_config(fake_llm.base_url), expect=1)
    [story] = stories(result)
    folder = result / story["path"]
    assert story["status"] == "failed"
    assert (folder / "plan.json").exists()
    assert not (folder / "story.md").exists()
    assert json.loads((folder / "refinement.json").read_text())["stopping_reason"] == "interrupted"


def test_critic_can_authorize_a_contiguous_span_without_changing_surroundings(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda body: critic_provider(body, "span")
    result = run(tmp_path, critic_config(fake_llm.base_url))
    [story] = stories(result)
    folder = result / story["path"]
    history = json.loads((folder / "refinement.json").read_text())
    assert history["iterations"][0]["status"] == "accepted"
    plan = json.loads((folder / "plan.json").read_text())
    assert plan["scenes"][3]["summary"] == "Function 3 realised intention"
    assert plan["scenes"][2]["summary"] == "Mara decides to relight the lamp despite the cost"


def test_budget_stop_during_refinement_retains_selected_plan_and_costs(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")

    def costly_critic(body: Json) -> Reply:
        reply = critic_provider(body)
        if "structural revision critic" in str(body["messages"][0]["content"]):
            assert isinstance(reply.body, dict)
            return completion(
                reply.body["choices"][0]["message"]["content"], completion_tokens=6000
            )
        return reply

    fake_llm.default = costly_critic
    value = critic_config(fake_llm.base_url)
    value["budget"] = {"per_run_usd": 0.0018}
    result = run(tmp_path, value, expect=1)
    [story] = stories(result)
    folder = result / story["path"]
    assert manifest(result)["status"] == "aborted"
    assert story["budget"]["calls"] == 8
    assert (folder / "plan.json").exists()
    assert not (folder / "story.md").exists()


def test_saved_d_revision_import_preserves_every_planning_and_refinement_call_once(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = critic_provider
    value = critic_config(fake_llm.base_url)
    source = run(tmp_path, value)
    [original] = stories(source)
    folder = source / original["path"]
    value["stories"]["conditions"][0]["params"].update(
        {
            "saved_plan": str(folder / "plan.json"),
            "provenance": str(folder / "planning-provenance.json"),
        }
    )
    imported = run(tmp_path, value)
    [story] = stories(imported)
    assert story["status"] == "ok"
    assert story["budget"] == original["budget"]
    calls = jsonl(imported / "calls.jsonl")
    assert len(calls) == 14
    assert all(c["imported_from"] and c["cost_usd"] == 0 for c in calls[:9])
    assert fake_llm.call_count == 14
    assert manifest(imported)["cost"]["usd"] == 0


@pytest.mark.parametrize(
    "tamper", ["calls", "seed", "model", "refinement", "nested", "usage", "plan"]
)
def test_saved_d_import_rejects_missing_or_stale_provenance_before_prose(
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = critic_provider
    value = critic_config(fake_llm.base_url)
    source = run(tmp_path, value)
    [original] = stories(source)
    folder = source / original["path"]
    provenance = folder / "planning-provenance.json"
    raw = json.loads(provenance.read_text())
    if tamper == "calls":
        raw["calls"].pop()
    elif tamper == "seed":
        raw["calls"][-1]["seed"] += 1
    elif tamper == "model":
        raw["critic_model"]["family"] = "changed"
    elif tamper == "refinement":
        raw["refinement"]["iterations"][0]["revision_steps"] = []
    elif tamper == "nested":
        raw["refinement"]["iterations"][0] = "malformed record"
    elif tamper == "usage":
        raw["calls"][-1]["token_cost_usd"] = 0
    else:
        (folder / "plan.json").write_text((folder / "plan.json").read_text() + " ")
    provenance.write_text(json.dumps(raw))
    value["stories"]["conditions"][0]["params"].update(
        {"saved_plan": str(folder / "plan.json"), "provenance": str(provenance)}
    )
    result = run(tmp_path, value)
    [story] = stories(result)
    assert story["status"] == "failed"
    assert "saved-D-plan" in story["error"]
    assert fake_llm.call_count == 14
