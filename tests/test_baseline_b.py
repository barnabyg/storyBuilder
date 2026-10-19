import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.gateway import Gateway
from nde.prose.scenes import SCENE_PROSE, SceneSpec, write_scenes
from nde.runs.config import StructuredConfig
from nde.runs.prompts import load_template
from nde.runs.state import RunState
from nde.schema.brief import Brief
from nde.stories.pipeline import StoryContext
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import (
    BASELINE_A,
    jsonl,
    last_user_message,
    reply_with_story,
    run,
    stories,
    story_config,
    story_files,
    story_text,
)

BASELINE_B = {"name": "B", "pipeline": "baseline_b"}


def outline(scenes: int = 5) -> dict[str, Any]:
    return {
        "title": "The Last Lamp",
        "scenes": [
            {"summary": f"Summary {i}.", "beats": [f"beat {i}a", f"beat {i}b"]}
            for i in range(1, scenes + 1)
        ],
    }


def _is_planner(body: Json) -> bool:
    return "story planner" in str(body["messages"][0]["content"])


def fake_b(body: Json) -> Reply:
    """Outlines for the planner; 400 words for each scene, 2,000 for a whole story."""
    if _is_planner(body):
        return completion(json.dumps(outline()))
    if "Now write scene" in last_user_message(body):
        return reply_with_story(400)
    return reply_with_story(2000)


@pytest.fixture
def b_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_b
    return fake_llm


def _scene_requests(fake: FakeOpenAI) -> list[str]:
    return [
        last_user_message(r.body)
        for r in fake.requests
        if "Now write scene" in last_user_message(r.body)
    ]


def test_a_and_b_run_on_the_same_briefs_in_one_run(b_llm: FakeOpenAI, tmp_path: Path) -> None:
    run_dir = run(tmp_path, story_config(b_llm.base_url, [BASELINE_A, BASELINE_B]))

    records = stories(run_dir)
    assert [(r["brief"], r["condition"]) for r in records] == [
        ("brief-01", "A"),
        ("brief-01", "B"),
        ("brief-02", "A"),
        ("brief-02", "B"),
    ]
    assert all(r["words"] == 2000 and r["in_range"] for r in records)
    # A: one call per brief. B: one outline and five scenes per brief.
    assert b_llm.call_count == 2 * (1 + 1 + 5)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| A | 2 | 0 | 2 |" in report
    assert "| B | 2 | 0 | 12 |" in report


def test_b_keeps_its_outline_and_every_scene_beside_the_story(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, story_config(b_llm.base_url, [BASELINE_B]))

    record = stories(run_dir)[0]
    folder = run_dir / record["path"]
    assert json.loads((folder / "outline.json").read_text(encoding="utf-8")) == outline()
    assert [f for f in record["files"] if f.startswith("attempts/1/scene")] == [
        f"attempts/1/scene-{i:02d}.md" for i in range(1, 6)
    ]
    steps = [(c["brief"], c["step"]) for c in jsonl(run_dir / "calls.jsonl")][:6]
    assert steps == [("brief-01", "outline")] + [
        ("brief-01", f"scene-{i:02d}") for i in range(1, 6)
    ]


def test_scenes_are_written_in_order_from_the_brief_outline_beats_and_story_so_far(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run(tmp_path, story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]}))

    first, second, *_, last = _scene_requests(b_llm)
    assert "lighthouse keeper" in first
    assert "Now write scene 1 of 5, about 400 words" in first
    assert "beat 1a" in first.split("Now write scene")[1]
    assert "Summary 5." in first  # the whole outline is in every scene prompt
    assert "This is the first scene." in first
    assert "1. Summary 1." in second.split("Scenes written so far:")[1]
    assert "Now write scene 5 of 5" in last


def test_the_planner_and_prose_roles_are_used(b_llm: FakeOpenAI, tmp_path: Path) -> None:
    config = story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]})
    config["roles"]["planner"]["model"] = "planner-model"
    config["prices"]["fake/planner-model"] = {"input_per_mtok": 0, "output_per_mtok": 0}

    run(tmp_path, config)

    models = [r.body["model"] for r in b_llm.requests]
    assert models == ["planner-model"] + ["fake-model"] * 5


def test_an_outline_that_needs_a_repair_is_repaired_and_counted(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    broken = outline() | {"scenes": outline()["scenes"][:2]}  # too few scenes
    b_llm.queue(completion(json.dumps(broken)))

    run_dir = run(
        tmp_path, story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]})
    )

    [record] = stories(run_dir)
    assert record["status"] == "ok"
    repair = b_llm.requests[1].body["messages"][-1]["content"]
    assert "scenes: List should have at least 5 items" in repair
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| fake/fake-model | 1 | 0.0% | 100.0% | 100.0% | 0 |" in report


def test_an_outline_still_invalid_after_repairs_fails_only_that_story(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    b_llm.queue(*(completion("{not json") for _ in range(3)))
    config = story_config(b_llm.base_url, [BASELINE_B])

    run_dir = run(tmp_path, config)

    first, second = stories(run_dir)
    assert (first["status"], second["status"]) == ("failed", "ok")
    assert "step 'outline'" in first["error"]


def test_the_outline_scene_count_follows_the_brief(b_llm: FakeOpenAI, tmp_path: Path) -> None:
    run(tmp_path, story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]}))

    planner_prompt = b_llm.requests[0].body["messages"][-1]["content"]
    assert "5 to 8 scenes" in planner_prompt
    schema = planner_prompt.split("JSON Schema:", 1)[1]
    assert '"minItems": 5' in schema
    assert '"maxItems": 8' in schema


def test_length_control_regenerates_the_prose_but_never_the_outline(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    short_scenes = iter([True] * 5)

    def first_attempt_short(body: Json) -> Reply:
        if not _is_planner(body) and next(short_scenes, False):
            return reply_with_story(200)
        return fake_b(body)

    b_llm.default = first_attempt_short

    run_dir = run(
        tmp_path, story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]})
    )

    [record] = stories(run_dir)
    assert [a["words"] for a in record["attempts"]] == [1000, 2000]
    assert sum(_is_planner(r.body) for r in b_llm.requests) == 1
    assert len(_scene_requests(b_llm)) == 10


def test_a_rerun_of_a_and_b_is_free_and_byte_identical(b_llm: FakeOpenAI, tmp_path: Path) -> None:
    config = story_config(b_llm.base_url, [BASELINE_A, BASELINE_B])
    first = run(tmp_path, config)
    calls = b_llm.call_count

    second = run(tmp_path, config)

    assert b_llm.call_count == calls
    assert story_files(second) == story_files(first)


def test_the_shared_prose_steps_can_be_driven_by_a_hand_written_scene_list(
    fake_llm: FakeOpenAI,
) -> None:
    fake_llm.default = lambda body: completion(f"Prose for: {last_user_message(body)[-40:]}")
    brief = Brief.model_validate(
        yaml.safe_load((REPO / "benchmark" / "dev" / "brief-03.yaml").read_text(encoding="utf-8"))
    )
    gateway = Gateway({"fake": ProviderConfig(base_url=fake_llm.base_url)}, environ={})
    ctx = StoryContext(
        gateway=gateway,
        state=RunState(),
        templates={SCENE_PROSE.label: load_template(REPO / "prompts", "scene_prose", 1)},
        roles={"prose": ModelBinding(provider="fake", model="m")},
        structured_config=StructuredConfig(),
        brief=brief,
        condition="C",
        repetition=0,
        seed=11,
    )
    scenes = [
        SceneSpec(summary="Priya finds the forged script.", beats=["She checks the date."]),
        SceneSpec(
            summary="She confronts the customer.",
            beats=["The customer lies.", "Priya lets her go."],
            notes="Tension: rising. Priya's trust in the customer: falls.",
        ),
    ]

    try:
        story = write_scenes(ctx, scenes, seed=11)
    finally:
        gateway.close()

    first, second = (last_user_message(r.body) for r in fake_llm.requests)
    assert "1. Priya finds the forged script." in first
    assert "2. She confronts the customer." in first
    assert "Now write scene 2 of 2" in second
    assert "Tension: rising." in second
    assert story.count("Prose for:") == 2
    assert [c.step for c in ctx.state.calls] == ["scene-01", "scene-02"]
    assert {c.condition for c in ctx.state.calls} == {"C"}


def test_the_committed_ab_config_runs_both_conditions(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for provider in ("DEEPSEEK", "OPENAI"):
        monkeypatch.setenv(f"NDE_{provider}_BASE_URL", fake_llm.base_url)
        monkeypatch.setenv(f"{provider}_API_KEY", "test-key")
    fake_llm.default = fake_b
    runs = tmp_path / "runs"

    assert main(["run", str(REPO / "configs" / "stories-ab.yaml"), "--runs-dir", str(runs)]) == 0

    [run_dir] = list(runs.iterdir())
    assert {r["condition"] for r in stories(run_dir)} == {"A", "B"}


def test_chatter_a_model_adds_inside_a_scene_is_removed_from_the_story(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    def chatty(body: Json) -> Reply:
        if "Now write scene" in last_user_message(body):
            text = story_text(400)
            return completion(
                f"Here is the scene:\n\n**Night Watch**\n\n{text}\n\nI hope this works!"
            )
        return fake_b(body)

    b_llm.default = chatty

    run_dir = run(
        tmp_path, story_config(b_llm.base_url, [BASELINE_B], briefs={"ids": ["brief-01"]})
    )

    [record] = stories(run_dir)
    story = (run_dir / record["path"] / "story.md").read_text(encoding="utf-8")
    assert "Here is" not in story
    assert "Night Watch" not in story
    assert "I hope" not in story
    assert record["words"] == 2000
