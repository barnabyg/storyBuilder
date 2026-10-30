import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from nde.continuity.plan import PLAIN_STYLE_RULES
from nde.continuity.story import NumberedStory
from tests.conftest import REPO, write_config
from tests.continuity_support import (
    calls,
    clean_audits,
    continuity_config,
    fake_continuity,
    flags_of,
    person,
    plan_reply,
    result,
    run_continuity,
)
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, manifest, stories

INTRODUCTION = "character_introduction"


@pytest.fixture
def continuity_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_continuity()
    return fake_llm


def _with_people(*people: dict[str, Any]) -> dict[str, Any]:
    return clean_audits() | {"List every named person": {"people": list(people)}}


def test_one_brief_gives_a_saved_story_sheets_flags_and_cost_in_one_folder(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    run_dir, folder = run_continuity(tmp_path, continuity_llm)

    assert manifest(run_dir)["status"] == "succeeded"
    saved = result(folder)
    assert (folder / saved["story"]).read_text(encoding="utf-8").startswith("Scene1 first")
    assert saved["flagged_problems"] == []
    assert saved["cost"]["calls"] == len(calls(run_dir))
    assert saved["cost"]["usd"] > 0
    assert "Only the owner's reading" in (folder / "summary.md").read_text(encoding="utf-8")


def test_every_sheet_records_appearance_back_story_and_entry(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm)

    sheets = json.loads((folder / "character-sheets.json").read_text(encoding="utf-8"))
    assert sheets == plan_reply()["characters"]
    for field in ("appearance", "back_story", "enters_scene", "introduction"):
        assert all(sheet[field] for sheet in sheets)


def test_a_plan_without_an_entry_is_sent_back_for_repair(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    broken = plan_reply()
    del broken["characters"][0]["enters_scene"]
    continuity_llm.queue(completion(json.dumps(broken)))

    run_dir, _ = run_continuity(tmp_path, continuity_llm)

    assert stories(run_dir)[0]["status"] == "ok"
    assert [c["step"] for c in calls(run_dir)][:2] == ["plan", "plan"]


def test_prose_follows_the_sheets_and_concrete_style_rules(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    run_continuity(tmp_path, continuity_llm)

    scenes = [
        last_user_message(r.body)
        for r in continuity_llm.requests
        if "Now write scene" in last_user_message(r.body)
    ]
    assert len(scenes) == 5
    for scene in scenes:
        assert "Tom Reyes\nAppearance: Tom Reyes is tall" in scene
        assert "Back-story: Tom Reyes has lived" in scene
        assert "Enters in scene 2:" in scene
        assert PLAIN_STYLE_RULES.strip() in scene
    assert "at most one simile or metaphor in a scene" in PLAIN_STYLE_RULES


def test_a_named_character_with_no_sheet_is_flagged_with_the_passage(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    continuity_llm.default = fake_continuity(
        _with_people(person("Elsie Marr", 1), person("Tom", 3), person("Viktor Hale", 6))
    )

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = flags_of(folder, INTRODUCTION)
    assert "Viktor Hale" in flag["reason"]
    assert flag["passage"].startswith("Scene3 second")
    assert set(flag) == {"problem_type", "passage", "reason", "paragraph"}
    assert flag["paragraph"] == 6


def test_a_sheeted_character_with_no_introduction_is_flagged(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    continuity_llm.default = fake_continuity(
        _with_people(person("Elsie Marr", 1), person("Tom", 3, introduced=False))
    )

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = flags_of(folder, INTRODUCTION)
    assert "Tom appears with no introduction" in flag["reason"]
    assert flag["passage"].startswith("Scene2 first")


def test_a_character_entering_in_the_wrong_scene_is_flagged(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    continuity_llm.default = fake_continuity(_with_people(person("Elsie", 1), person("Tom", 1)))

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = flags_of(folder, INTRODUCTION)
    assert "first appears in scene 1" in flag["reason"]
    assert "enter in scene 2" in flag["reason"]


def test_a_planned_character_who_never_appears_is_flagged_at_their_planned_scene(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    continuity_llm.default = fake_continuity(_with_people(person("Elsie Marr", 1)))

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = flags_of(folder, INTRODUCTION)
    assert "Tom Reyes is planned to enter in scene 2 but never appears" in flag["reason"]
    assert flag["passage"].startswith("Scene2 first")


def test_a_story_that_introduces_everyone_as_planned_has_no_such_flag(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm)

    assert flags_of(folder, INTRODUCTION) == []


def test_a_citation_of_a_paragraph_that_does_not_exist_is_sent_back(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    good = fake_continuity()
    wrong = {"people": [person("Elsie Marr", 99)]}
    answered: list[str] = []

    def once_wrong(body: Json) -> Reply:
        message = last_user_message(body)
        if "List every named person" in message and not answered:
            answered.append(message)
            return completion(json.dumps(wrong))
        if "did not validate" in message:
            answered.append(message)
            return completion(json.dumps(clean_audits()["List every named person"]))
        return good(body)

    continuity_llm.default = once_wrong

    run_dir, folder = run_continuity(tmp_path, continuity_llm)

    assert "paragraph 99 does not exist" in answered[1]
    assert stories(run_dir)[0]["status"] == "ok"
    assert flags_of(folder, INTRODUCTION) == []


def test_the_checker_reads_numbered_paragraphs_under_scene_headings() -> None:
    story = NumberedStory.from_scenes("One.\n\nTwo.\n\nThree.\n", ["One.\n\nTwo.", "Three."])

    assert story.scenes == (1, 1, 2)
    assert (
        story.render() == "--- Scene 1 ---\n\n[1] One.\n\n[2] Two.\n\n--- Scene 2 ---\n\n[3] Three."
    )
    assert story.text == "One.\n\nTwo.\n\nThree.\n"


def test_the_command_runs_one_dev_brief_and_reports_where_everything_is(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, continuity_config(continuity_llm.base_url))
    runs = tmp_path / "runs"

    code = main(["continuity", "brief-02", "--config", str(config), "--runs-dir", str(runs)])

    out = capsys.readouterr().out
    assert code == 0
    assert "flagged problems: 0" in out
    assert "cost: US$" in out
    [run_dir] = list(runs.iterdir())
    assert [r["brief"] for r in stories(run_dir)] == ["brief-02"]


def test_the_command_refuses_a_held_out_brief_before_any_call(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path, continuity_config(continuity_llm.base_url))

    code = main(["continuity", "brief-11", "--config", str(config)])

    assert code == 2
    assert "brief-11" in capsys.readouterr().err
    assert continuity_llm.call_count == 0


def test_a_config_naming_a_held_out_brief_is_refused_before_any_call(
    continuity_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = continuity_config(continuity_llm.base_url)
    raw["stories"]["briefs"] = {"ids": ["brief-11"]}
    config = write_config(tmp_path, raw)

    assert main(["run", str(config)]) == 2
    assert "only on dev briefs" in capsys.readouterr().err
    assert continuity_llm.call_count == 0


def test_the_committed_config_is_within_the_live_testing_allowance() -> None:
    raw = yaml.safe_load((REPO / "configs" / "continuity-dev.yaml").read_text(encoding="utf-8"))
    assert raw["budget"]["per_run_usd"] <= 3
    assert "allow_unpriced_models" not in raw["budget"]
    assert {role["provider"] for role in raw["roles"].values()} == {"openai"}
    assert raw["stories"]["briefs"]["split"] == "dev"
