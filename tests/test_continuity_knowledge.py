import json
from pathlib import Path
from typing import Any

import pytest

from nde.continuity.knowledge import KnowledgeAudit, check_knowledge
from nde.continuity.plan import ContinuityPlan
from nde.continuity.story import NumberedStory
from tests.continuity_evidence import (
    MRS_ALCOTT_AIDE,
    MRS_ALCOTT_TELLS_HER,
    PRIYA_EARLIER,
    PRIYA_KNOWS_THE_LEDGER,
)
from tests.continuity_support import (
    clean_audits,
    fake_continuity,
    flags_of,
    plan_reply,
    run_continuity,
)
from tests.fake_openai import FakeOpenAI

KNOWLEDGE = "unexplained_knowledge"
HIDDEN_FACT = "The boat is already sold."


@pytest.fixture
def continuity_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_continuity()
    return fake_llm


def _entry(acted: int, source: str, **fields: Any) -> dict[str, Any]:
    return {
        "character": "Tom Reyes",
        "fact": HIDDEN_FACT,
        "acted_paragraph": acted,
        "how_learned": "The story never says.",
        "source": source,
    } | fields


def _with_entries(*entries: dict[str, Any]) -> dict[str, Any]:
    return clean_audits() | {"record each specific fact": {"entries": list(entries)}}


def _plan() -> ContinuityPlan:
    return ContinuityPlan.model_validate(plan_reply())


def _story() -> NumberedStory:
    return NumberedStory(("First.", "Second.", "Third."), (1, 1, 2))


def _check(*entries: dict[str, Any]) -> list[str]:
    audit = KnowledgeAudit.model_validate({"entries": list(entries)})
    return [flag.reason for flag in check_knowledge(_plan(), _story(), audit)]


def test_acting_on_a_secret_never_told_is_flagged_with_character_fact_and_passage(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    continuity_llm.default = fake_continuity(_with_entries(_entry(4, "none")))

    _, folder = run_continuity(tmp_path, continuity_llm)

    [flag] = flags_of(folder, KNOWLEDGE)
    assert "Tom Reyes" in flag["reason"]
    assert HIDDEN_FACT in flag["reason"]
    assert flag["passage"].startswith("Scene2 second")


def test_learning_the_fact_on_the_page_first_has_no_flag() -> None:
    assert _check(_entry(3, "on_page", learned_paragraph=1)) == []
    assert _check(_entry(2, "on_page", learned_paragraph=2)) == []


def test_what_a_person_in_their_place_would_simply_know_is_recorded_without_a_flag() -> None:
    assert _check(_entry(1, "ordinary", how_learned="It is their own boat.")) == []


def test_learning_the_fact_only_after_acting_on_it_is_flagged() -> None:
    [reason] = _check(_entry(1, "on_page", learned_paragraph=3))

    assert "act on it in paragraph 1 but only learn it in paragraph 3" in reason


def test_knowledge_the_sheet_lists_has_no_flag_but_a_false_citation_does() -> None:
    assert _check(_entry(1, "sheet", sheet_item=1)) == []

    [reason] = _check(_entry(1, "sheet", sheet_item=9))
    assert "The sheet does not list it" in reason

    [reason] = _check(_entry(1, "sheet", sheet_item=1, character="Nobody Known"))
    assert "The sheet does not list it" in reason


def test_a_real_g1_character_doing_the_sums_before_hearing_them_is_flagged() -> None:
    story = NumberedStory(
        (PRIYA_EARLIER, PRIYA_KNOWS_THE_LEDGER, MRS_ALCOTT_TELLS_HER, MRS_ALCOTT_AIDE),
        (1, 2, 3, 3),
    )
    plan = ContinuityPlan.model_validate(
        plan_reply() | {"characters": [_priya_sheet()]},
    )
    audit = KnowledgeAudit.model_validate(
        {
            "entries": [
                {
                    "character": "Priya Raman",
                    "fact": "The scheme brings in a hundred dollars a month and pays for an aide.",
                    "acted_paragraph": 2,
                    "how_learned": "Mrs. Alcott tells her, but only in paragraphs 3 and 4.",
                    "source": "on_page",
                    "learned_paragraph": 3,
                }
            ]
        }
    )

    [flag] = check_knowledge(plan, story, audit)

    assert flag.passage == PRIYA_KNOWS_THE_LEDGER
    assert "Priya Raman" in flag.reason
    assert "a hundred dollars a month" in flag.reason


def _priya_sheet() -> dict[str, Any]:
    return {
        "name": "Priya Raman",
        "appearance": "Tired, in a white coat.",
        "back_story": "A night-shift pharmacist two weeks from quitting.",
        "enters_scene": 1,
        "introduction": "Behind the counter at night.",
        "knows_at_start": ["How to read a prescription."],
    }


def test_the_run_saves_a_scene_by_scene_record_of_what_each_character_acted_on(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    continuity_llm.default = fake_continuity(
        _with_entries(_entry(1, "sheet", sheet_item=1), _entry(4, "on_page", learned_paragraph=3))
    )

    _, folder = run_continuity(tmp_path, continuity_llm)

    record = json.loads((folder / "knowledge.json").read_text(encoding="utf-8"))
    assert [scene["scene"] for scene in record] == [1, 2, 3, 4, 5]
    assert [e["acted_paragraph"] for e in record[0]["entries"]] == [1]
    assert record[1]["entries"][0]["how_learned"] == "The story never says."
    assert record[2]["entries"] == []
    assert flags_of(folder, KNOWLEDGE) == []


def test_the_sheets_and_writer_carry_what_each_character_knows_at_the_start(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_continuity(tmp_path, continuity_llm)

    prompts = [str(r.body["messages"][-1]["content"]) for r in continuity_llm.requests]
    [scene] = [p for p in prompts if "Now write scene 1 of" in p]
    assert "Knows at the start:\n1. Tom Reyes knows the tide tables." in scene
    assert "A character may act only on what" in scene
    [audit] = [p for p in prompts if "record each specific fact" in p]
    assert "1. Tom Reyes knows the tide tables." in audit


def test_an_on_page_answer_without_its_paragraph_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_page needs learned_paragraph"):
        KnowledgeAudit.model_validate({"entries": [_entry(1, "on_page")]})
