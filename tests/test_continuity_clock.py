import json
from pathlib import Path
from typing import Any

import pytest

from nde.continuity.clock import ClockAudit, check_clock
from nde.continuity.plan import ContinuityPlan, format_minutes
from nde.continuity.story import NumberedStory
from tests.continuity_evidence import POWER_CUT, SOUR_AFTER_FOUR_MINUTES
from tests.continuity_support import (
    clean_audits,
    fake_continuity,
    flags_of,
    plan_reply,
    result,
    run_continuity,
)
from tests.fake_openai import FakeOpenAI

TIMING = "impossible_timing"


@pytest.fixture
def continuity_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_continuity()
    return fake_llm


def _event(paragraph: int, minutes: float, **fields: Any) -> dict[str, Any]:
    return {
        "event": "Food smells sour",
        "paragraph": paragraph,
        "prose_minutes": minutes,
        "realistic_min_minutes": 720,
        "realistic_max_minutes": 4320,
        "why": "A closed fridge stays cold for hours.",
    } | fields


def _power_cut_story() -> NumberedStory:
    return NumberedStory((POWER_CUT, SOUR_AFTER_FOUR_MINUTES), (1, 1))


def _plan() -> ContinuityPlan:
    return ContinuityPlan.model_validate(plan_reply())


def test_the_saved_plan_records_elapsed_time_and_expected_durations(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm)

    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    assert [scene["elapsed_minutes"] for scene in plan["scenes"]] == [60] * 5
    assert plan["scenes"][2]["key_events"][0]["expected_min_minutes"] == 720


def test_the_writer_and_the_checker_both_receive_the_clock(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_continuity(tmp_path, continuity_llm)

    prompts = [str(r.body["messages"][-1]["content"]) for r in continuity_llm.requests]
    [scene_three] = [p for p in prompts if "Now write scene 3 of" in p]
    assert "Scene clock: 1 hour pass" in scene_three
    assert "Food in the fridge smells sour takes 12 hours to 3 days" in scene_three
    [audit] = [p for p in prompts if "List every event whose time" in p]
    assert "K1: Food in the fridge smells sour (12 hours to 3 days)" in audit


def test_food_smelling_sour_four_minutes_after_a_power_cut_is_flagged() -> None:
    audit = ClockAudit.model_validate({"events": [_event(2, 4)], "scenes": []})

    [flag] = check_clock(_plan(), _power_cut_story(), audit)

    assert flag.problem_type == TIMING
    assert flag.passage == SOUR_AFTER_FOUR_MINUTES
    assert "too quickly" in flag.reason
    assert "4 minutes" in flag.reason
    assert "12 hours" in flag.reason


def test_an_event_that_takes_too_long_is_flagged() -> None:
    audit = ClockAudit.model_validate({"events": [_event(2, 20000)], "scenes": []})

    [flag] = check_clock(_plan(), _power_cut_story(), audit)

    assert "too slowly" in flag.reason


def test_the_plans_clock_outranks_the_readers_estimate() -> None:
    lenient = _event(2, 4, key_event="K1", realistic_min_minutes=1)
    audit = ClockAudit.model_validate({"events": [lenient], "scenes": []})

    [flag] = check_clock(_plan(), _power_cut_story(), audit)

    assert "at least 12 hours" in flag.reason


def test_a_scene_whose_stated_time_strays_from_the_plan_is_flagged() -> None:
    reading = {"scene": 1, "paragraph": 1, "prose_minutes": 600}
    audit = ClockAudit.model_validate({"events": [], "scenes": [reading]})

    [flag] = check_clock(_plan(), _power_cut_story(), audit)

    assert flag.passage == POWER_CUT
    assert "Scene 1 says 10 hours" in flag.reason
    assert "1 hour" in flag.reason


def test_plausible_timing_has_no_flags() -> None:
    audit = ClockAudit.model_validate(
        {
            "events": [_event(2, 1440, key_event="K1")],
            "scenes": [{"scene": 1, "paragraph": 1, "prose_minutes": 70}],
        }
    )

    assert check_clock(_plan(), _power_cut_story(), audit) == []


def test_timing_flags_join_the_same_list_with_their_passage_and_reason(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    answers = clean_audits() | {
        "List every event whose time": {"events": [_event(5, 4)], "scenes": []}
    }
    continuity_llm.default = fake_continuity(answers)

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = result(folder)["flagged_problems"]
    assert flag == flags_of(folder, TIMING)[0]
    assert flag["passage"].startswith("Scene3 first")
    assert "too quickly" in flag["reason"]


def test_a_story_with_plausible_timing_has_no_timing_flags(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm)

    assert flags_of(folder, TIMING) == []


@pytest.mark.parametrize(
    ("minutes", "shown"),
    [(4, "4 minutes"), (1, "1 minute"), (90, "1.5 hours"), (60, "1 hour"), (4320, "3 days")],
)
def test_durations_read_plainly(minutes: float, shown: str) -> None:
    assert format_minutes(minutes) == shown
