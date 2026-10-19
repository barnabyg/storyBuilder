from pathlib import Path

import pytest

from tests.fake_openai import FakeOpenAI, completion
from tests.stories import (
    BASELINE_A,
    jsonl,
    manifest,
    reply_with_story,
    run,
    stories,
    story_config,
    story_files,
    story_text,
)


@pytest.fixture
def story_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: reply_with_story(2000)
    return fake_llm


def _one_brief(base_url: str, **stories_section: object) -> dict[str, object]:
    return story_config(base_url, [BASELINE_A], briefs={"ids": ["brief-01"]}, **stories_section)


def test_stories_are_stored_raw_and_normalised(story_llm: FakeOpenAI, tmp_path: Path) -> None:
    raw = "Here is your story:\n\n# The Lamp\n\n" + story_text(2000) + "\n\nThe End."
    story_llm.default = lambda _body: completion(raw)

    run_dir = run(tmp_path, _one_brief(story_llm.base_url))

    [record] = stories(run_dir)
    folder = run_dir / record["path"]
    assert (folder / "story.raw.md").read_text(encoding="utf-8") == raw
    normalised = (folder / "story.md").read_text(encoding="utf-8")
    assert normalised == story_text(2000) + "\n"
    assert record["words"] == 2000


def test_an_out_of_range_story_is_regenerated_on_a_new_seed(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(reply_with_story(1500))

    run_dir = run(tmp_path, _one_brief(story_llm.base_url))

    [record] = stories(run_dir)
    assert [(a["words"], a["in_range"]) for a in record["attempts"]] == [
        (1500, False),
        (2000, True),
    ]
    assert record["chosen_attempt"] == 1
    assert record["in_range"] is True
    first_seed, second_seed = (request.body["seed"] for request in story_llm.requests)
    assert first_seed == 7
    assert second_seed != first_seed
    # Both attempts are kept, costed and attributed.
    folder = run_dir / record["path"]
    assert (folder / "attempts" / "1" / "story.raw.md").is_file()
    assert (folder / "attempts" / "2" / "story.raw.md").is_file()
    calls = jsonl(run_dir / "calls.jsonl")
    assert [c["story_attempt"] for c in calls] == [0, 1]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| brief-01 | A | 1 | 2000 | 2000 | 2 | yes | ok | 2 |" in report


def test_a_story_still_out_of_range_after_the_bound_is_kept_and_flagged(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(reply_with_story(1000), reply_with_story(1700), reply_with_story(1200))

    run_dir = run(tmp_path, _one_brief(story_llm.base_url))

    [record] = stories(run_dir)
    assert len(record["attempts"]) == 3
    # The attempt closest to the target is kept.
    assert (record["chosen_attempt"], record["words"], record["in_range"]) == (1, 1700, False)
    assert record["status"] == "ok"
    assert manifest(run_dir)["stories"]["out_of_range"] == ["A/brief-01#1"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "### Out of length range" in report
    assert "A/brief-01 #1: 1700 words after 3 attempts (target 2000 \u00b1 10%)" in report


def test_the_tolerance_and_attempt_bound_are_configurable(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(reply_with_story(1500), reply_with_story(1500))
    config = _one_brief(story_llm.base_url, length={"tolerance": 0.3, "max_attempts": 1})

    [record] = stories(run(tmp_path, config))

    assert (len(record["attempts"]), record["in_range"]) == (1, True)
    assert story_llm.call_count == 1


def test_regeneration_is_deterministic_so_a_rerun_is_free_and_identical(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(reply_with_story(1500), reply_with_story(900))
    config = _one_brief(story_llm.base_url)
    first = run(tmp_path, config)
    calls = story_llm.call_count

    second = run(tmp_path, config)

    assert story_llm.call_count == calls == 3
    assert story_files(second) == story_files(first)
