import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from nde.cli import main
from nde.continuity.flags import Flag
from nde.continuity.plan import ContinuityPlan
from nde.continuity.repair import REPAIR_PROMPT, RepairedPassage, group_by_passage, repair_passage
from nde.continuity.story import NumberedStory
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.gateway import Gateway
from nde.runs.config import StructuredConfig
from nde.runs.prompts import load_template
from nde.runs.state import RunState
from nde.schema.brief import Brief
from nde.stories.pipeline import StoryContext
from tests.conftest import REPO, write_config
from tests.continuity_support import (
    calls,
    clean_audits,
    continuity_config,
    fake_continuity,
    person,
    plan_reply,
    result,
    run_continuity,
    scene_text,
)
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, manifest, stories

UNINTRODUCED = clean_audits() | {
    "List every named person": {
        "people": [person("Elsie Marr", 1), person("Tom", 3, introduced=False)]
    }
}


def _audit_requests(fake: FakeOpenAI) -> int:
    return sum("List every named person" in last_user_message(r.body) for r in fake.requests)


def _fixed_after(rounds: int) -> Callable[[Json], Reply]:
    """A fake whose check stops finding the problem after `rounds` flagged rounds."""
    flagged = fake_continuity(UNINTRODUCED)
    clean = fake_continuity()
    seen = [0]

    def reply(body: Json) -> Reply:
        if "List every named person" in last_user_message(body):
            seen[0] += 1
            return (flagged if seen[0] <= rounds else clean)(body)
        return flagged(body)

    return reply


@pytest.fixture
def continuity_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = _fixed_after(1)
    return fake_llm


def _paragraphs(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().split("\n\n")


def test_text_outside_the_repaired_passage_is_unchanged_word_for_word(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm, repair_tries=2)

    before = _paragraphs(folder / "story.unrepaired.md")
    after = _paragraphs(folder / "story.md")
    assert len(before) == len(after) == 10
    changed = [i for i, (old, new) in enumerate(zip(before, after, strict=True)) if old != new]
    assert changed == [2]
    assert after[2] == "A repaired paragraph."
    assert before[2].startswith("Scene2 first")


def test_a_flagged_part_of_a_paragraph_is_repaired_and_the_rest_of_it_kept(
    continuity_llm: FakeOpenAI,
) -> None:

    continuity_llm.queue(completion(json.dumps({"replacement": "It took a day."})))
    gateway = Gateway({"fake": ProviderConfig(base_url=continuity_llm.base_url)}, environ={})
    brief = Brief.model_validate(
        yaml.safe_load((REPO / "benchmark" / "dev" / "brief-01.yaml").read_text(encoding="utf-8"))
    )
    ctx = StoryContext(
        gateway=gateway,
        state=RunState(),
        templates={
            REPAIR_PROMPT.label: load_template(
                REPO / "prompts", REPAIR_PROMPT.template, REPAIR_PROMPT.version
            )
        },
        roles={"prose": ModelBinding(provider="fake", model="m")},
        structured_config=StructuredConfig(),
        brief=brief,
        condition="continuity",
        repetition=0,
        seed=1,
    )
    story = NumberedStory(("Keep me. It took a minute. Keep me too.", "Other."), (1, 1))
    flag = Flag(problem_type="any_new_type", passage="It took a minute.", reason="Too quick.")

    try:
        repaired = repair_passage(
            ctx, ContinuityPlan.model_validate(plan_reply()), story, [flag], attempt=1
        )
    finally:
        gateway.close()

    assert repaired is not None
    assert repaired[0].paragraphs == ("Keep me. It took a day. Keep me too.", "Other.")
    assert repaired[1].paragraph == 1


def test_the_repair_step_receives_the_sheets_the_plan_and_the_problem(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_continuity(tmp_path, continuity_llm, repair_tries=1)

    [request] = [
        last_user_message(r.body)
        for r in continuity_llm.requests
        if "Rewrite only this passage" in last_user_message(r.body)
    ]
    assert "Tom Reyes\nAppearance: Tom Reyes is tall" in request
    assert "Scene 3 (1 hour since the scene before): Summary 3." in request
    assert "character_introduction: Tom appears with no introduction" in request
    assert f"Rewrite only this passage:\n\n{scene_text(2).split(chr(10) * 2)[0]}" in request
    assert "at most one simile or metaphor" in request


def test_repair_stops_at_the_fixed_number_of_tries_and_reports_what_is_left(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    # The story's repaired paragraph changes each round, so give each try a new reply.
    replies = iter(f"Repaired on try {n}." for n in range(1, 9))
    flagged = fake_continuity(UNINTRODUCED)

    def never_fixed(body: Json) -> Reply:
        if "Rewrite only this passage" in last_user_message(body):
            return completion(json.dumps({"replacement": next(replies)}))
        return flagged(body)

    continuity_llm.default = never_fixed

    run_dir, folder = run_continuity(tmp_path, continuity_llm, repair_tries=2)

    saved = result(folder)
    assert _audit_requests(continuity_llm) == 3  # the first check and one after each try
    assert [c["step"] for c in calls(run_dir) if c["step"].startswith("repair")] == [
        "repair-1-p003",
        "repair-2-p003",
    ]
    assert saved["finished"] is True
    [problem] = saved["unresolved_problems"]
    assert problem["passage"] == "Repaired on try 2."
    assert "Tom appears with no introduction" in problem["reason"]
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "## Unresolved problems, last check" in summary
    assert "Tom appears with no introduction" in summary.split("Unresolved problems")[1]


def test_a_repair_that_works_stops_early_and_leaves_nothing_unresolved(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm, repair_tries=3)

    saved = result(folder)
    assert _audit_requests(continuity_llm) == 2
    assert len(saved["flagged_problems"]) == 1
    assert saved["unresolved_problems"] == []


def test_each_try_is_costed_and_the_budget_stops_repair_with_the_work_so_far_saved(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    flagged = fake_continuity(UNINTRODUCED)
    expensive = completion(
        json.dumps({"replacement": "An expensive repair."}), completion_tokens=10_000_000
    )

    def dear_repairs(body: Json) -> Reply:
        if "Rewrite only this passage" in last_user_message(body):
            return expensive
        return flagged(body)

    continuity_llm.default = dear_repairs

    config = continuity_config(continuity_llm.base_url, repair_tries=5)
    config["budget"] = {"per_run_usd": 1.0}
    runs = tmp_path / "runs"

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])

    [run_dir] = list(runs.iterdir())
    assert code == 1
    assert manifest(run_dir)["abort_reason"] == "budget"
    [record] = stories(run_dir)
    assert record["status"] == "failed"
    repairs = [c for c in calls(run_dir) if c["step"].startswith("repair")]
    assert repairs
    assert repairs[0]["cost_usd"] == pytest.approx(2.0, rel=0.01)
    folder = run_dir / record["path"]
    assert result(folder)["finished"] is False
    assert (folder / "story.md").read_text(encoding="utf-8").count("An expensive repair.") == 1
    assert "stopped before" in (folder / "summary.md").read_text(encoding="utf-8")


def test_the_run_saves_each_change_with_its_problem_and_the_text_before_and_after(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    _, folder = run_continuity(tmp_path, continuity_llm, repair_tries=2)

    [change] = json.loads((folder / "repairs.json").read_text(encoding="utf-8"))
    assert change["attempt"] == 1
    assert change["paragraph"] == 3
    assert change["before"].startswith("Scene2 first")
    assert change["after"] == "A repaired paragraph."
    assert change["problems"][0]["problem_type"] == "character_introduction"
    # The saved changes replay the unrepaired story into the final one.
    story = (folder / "story.unrepaired.md").read_text(encoding="utf-8")
    assert story.replace(change["before"], change["after"]) == (folder / "story.md").read_text(
        encoding="utf-8"
    )


def test_every_problem_type_is_repaired_the_same_way_and_shared_passages_go_together() -> None:
    first = Flag(problem_type="future_type", passage="Same.", reason="One.")
    second = Flag(problem_type="other_type", passage="Same.", reason="Two.")
    third = Flag(problem_type="future_type", passage="Different.", reason="Three.")

    assert group_by_passage([first, third, second]) == [[first, second], [third]]


def test_a_replacement_with_line_breaks_is_rejected() -> None:
    with pytest.raises(ValueError, match="one paragraph"):
        RepairedPassage(replacement="One.\n\nTwo.")


def test_word_count_and_checksum_describe_the_repaired_story(
    continuity_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir, folder = run_continuity(tmp_path, continuity_llm, repair_tries=1)

    [record] = stories(run_dir)
    story = (folder / "story.md").read_text(encoding="utf-8")
    assert record["words"] == len(story.split())
    assert "A repaired paragraph." in story
