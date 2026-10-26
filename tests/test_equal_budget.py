import json
from pathlib import Path
from typing import Any

import pytest

from nde.cli import main
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import (
    BASELINE_A,
    last_user_message,
    manifest,
    run,
    stories,
    story_config,
    story_files,
    story_text,
)
from tests.test_baseline_b import BASELINE_B, outline
from tests.test_plan_generation import fake_plan

# Every reply reports 10 input tokens. A story costs 2,000 output tokens; a B scene
# costs SCENE_TOKENS, so B's budget is 15 (outline) + 5 * (10 + SCENE_TOKENS).
SCENE_TOKENS = 800


def _reply(body: Json, scene_tokens: int = SCENE_TOKENS) -> Reply:
    if "story planner" in str(body["messages"][0]["content"]):
        return completion(json.dumps(outline()))
    if "Now write scene" in last_user_message(body):
        return completion(story_text(400), completion_tokens=scene_tokens)
    return completion(story_text(2000), completion_tokens=2000)


@pytest.fixture
def budget_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = _reply
    return fake_llm


def _config(base_url: str, **equal_budget: Any) -> dict[str, Any]:
    return story_config(
        base_url,
        [BASELINE_A, BASELINE_B],
        equal_budget={"conditions": ["A", "B"]} | equal_budget,
    )


def _revision_requests(fake: FakeOpenAI) -> list[str]:
    return [
        last_user_message(r.body) for r in fake.requests if "Revise" in last_user_message(r.body)
    ]


def test_the_report_shows_tokens_and_calls_per_story_and_condition(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, story_config(budget_llm.base_url, [BASELINE_A, BASELINE_B]))

    a, b = stories(run_dir)[:2]
    assert a["budget"] == {"calls": 1, "input_tokens": 10, "output_tokens": 2000}
    assert b["budget"] == {"calls": 6, "input_tokens": 60, "output_tokens": 5 + 5 * 800}
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Generation budget" in report
    assert "| A | 2 | 2 | 4020 | 2010 |" in report
    assert "| B | 2 | 12 | 8130 | 4065 |" in report


def test_equal_budget_mode_gives_the_cheaper_condition_revision_passes(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, _config(budget_llm.base_url))

    matches = manifest(run_dir)["stories"]["equal_budget"]["matches"]
    assert [(m["brief"], m["revised"], m["passes"], m["matched"]) for m in matches] == [
        ("brief-01", "A", 1, True),
        ("brief-02", "A", 1, True),
    ]
    for match in matches:
        assert match["before"] == {"A": 2010, "B": 4065}
        assert match["after"] == {"A": 4020, "B": 4065}
        after = match["after"]
        assert abs(after["A"] - after["B"]) <= 0.1 * max(after.values())
    a = stories(run_dir)[0]
    assert (a["revision_passes"], a["budget"]["calls"]) == (1, 2)
    revision = _revision_requests(budget_llm)[0]
    assert "lighthouse keeper" in revision
    assert story_text(2000)[:200] in revision


def test_comparison_report_identifies_selected_final_prose_and_actual_spend(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, _config(budget_llm.base_url))
    comparison = json.loads((run_dir / "comparison.json").read_text())
    assert len(comparison["pairs"]) == 2
    pair = comparison["pairs"][0]
    assert pair["comparable"]
    assert pair["conditions"]["A"]["revision_passes"] == 1
    assert pair["conditions"]["A"]["final_sha256"]
    assert pair["conditions"]["A"]["tokens"] == 4020
    assert pair["conditions"]["A"]["new_spend_usd"] > 0
    second = run(tmp_path, _config(budget_llm.base_url))
    replay = json.loads((second / "comparison.json").read_text())
    assert replay["pairs"][0]["conditions"]["A"]["new_spend_usd"] == 0
    assert replay["pairs"][0]["conditions"]["A"]["tokens"] == 4020


def test_aborted_revision_retains_the_incomplete_pair_and_previous_selected_prose(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    budget_llm.default = lambda body: (
        error(400, "revision refused") if "Revise" in last_user_message(body) else _reply(body)
    )
    run_dir = run(tmp_path, _config(budget_llm.base_url), expect=1)
    comparison = json.loads((run_dir / "comparison.json").read_text())
    assert len(comparison["pairs"]) == 2
    first = comparison["pairs"][0]
    assert not first["comparable"]
    assert first["matching"]["reason"] == "revision interrupted: HTTP 400: revision refused"
    assert first["conditions"]["A"]["final_sha256"]
    assert not comparison["pairs"][1]["comparable"]


def test_bc_comparison_counts_multistage_planning_and_uses_revised_final_story(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    def replies(body: Json) -> Reply:
        system = str(body["messages"][0]["content"])
        if "skeleton planner" in system or "scene-channel planner" in system:
            reply = fake_plan(body)
            assert isinstance(reply.body, dict)
            return completion(reply.body["choices"][0]["message"]["content"], prompt_tokens=1000)
        if "Revise" in last_user_message(body):
            return completion(story_text(2000, marker="revised"), completion_tokens=2000)
        return _reply(body)

    budget_llm.default = replies
    config = story_config(
        budget_llm.base_url,
        [{"name": "C", "pipeline": "baseline_c", "params": {"tier": "L"}}, BASELINE_B],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["C", "B"]},
    )
    run_dir = run(tmp_path, config)
    comparison = json.loads((run_dir / "comparison.json").read_text())
    [pair] = comparison["pairs"]
    assert pair["comparable"]
    assert pair["conditions"]["C"]["calls"] == 11
    assert pair["conditions"]["C"]["tokens"] == 10080
    assert pair["conditions"]["B"]["revision_passes"] == 3
    assert pair["matching"]["revised"] == "B"
    assert pair["conditions"]["B"]["tokens"] == 10095
    packet = tmp_path / "realisation"
    assert (
        main(
            [
                "realisation",
                "create",
                str(run_dir),
                "--condition",
                "B",
                "--brief",
                "brief-01",
                "--out",
                str(packet),
            ]
        )
        == 0
    )
    evidence = json.loads((packet / "packet.json").read_text())
    assert evidence["prose"].startswith("revised")
    assert evidence["alignment"] == "unresolved"
    batches = tmp_path / "batches"
    assert (
        main(
            [
                "eval",
                "batch",
                "bc",
                "--from",
                str(run_dir),
                "C",
                "--from",
                str(run_dir),
                "B",
                "--batches-dir",
                str(batches),
                "--rubric-dir",
                str(REPO / "benchmark/rubric"),
            ]
        )
        == 0
    )
    view = json.loads((batches / "bc/view.json").read_text())
    [shown] = view["pairs"]
    assert any(text.startswith("revised") for text in shown.values())


def test_revised_stories_go_through_normalisation_and_length_control(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    replies = iter(
        [
            completion("# Revised\n\n" + story_text(1500), completion_tokens=1000),
            completion("**Revised**\n\n" + story_text(2050), completion_tokens=1000),
        ]
    )

    def revision_replies(body: Json) -> Reply:
        if "Revise" in last_user_message(body):
            return next(replies)
        return _reply(body)

    budget_llm.default = revision_replies
    config = _config(budget_llm.base_url)
    config["stories"]["briefs"] = {"ids": ["brief-01"]}

    run_dir = run(tmp_path, config)

    a = stories(run_dir)[0]
    folder = run_dir / a["path"]
    assert (folder / "story.md").read_text(encoding="utf-8") == story_text(2050) + "\n"
    assert (folder / "story.original.md").read_text(encoding="utf-8") == story_text(2000) + "\n"
    assert (folder / "revisions" / "1" / "attempts" / "2" / "story.raw.md").is_file()
    assert (a["words"], a["in_range"], a["revision_passes"]) == (2050, True, 1)
    [match] = manifest(run_dir)["stories"]["equal_budget"]["matches"]
    # Two revision attempts of 10 + 1,000 tokens each.
    assert match["after"]["A"] == 2010 + 2 * 1010


def test_a_budget_that_cannot_be_matched_is_reported_per_brief(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    budget_llm.default = lambda body: _reply(body, scene_tokens=2000)
    run_dir = run(tmp_path, _config(budget_llm.base_url, max_revision_passes=2))

    matches = manifest(run_dir)["stories"]["equal_budget"]["matches"]
    assert all(not m["matched"] and m["passes"] == 2 for m in matches)
    assert matches[0]["reason"] == "not within 10% after 2 revision passes"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Equal budget" in report
    assert "| brief-01 | 1 | 2010 | 10065 | A | 2 | 6030 | 10065 | no |" in report
    assert "not within 10% after 2 revision passes" in report


def test_a_revision_pass_that_overshoots_is_reported(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    # B costs 15 + 5 * 510 = 2565; one revision takes A from 2010 to 4020.
    budget_llm.default = lambda body: _reply(body, scene_tokens=500)

    run_dir = run(tmp_path, _config(budget_llm.base_url))

    match = manifest(run_dir)["stories"]["equal_budget"]["matches"][0]
    assert (match["passes"], match["matched"]) == (1, False)
    assert match["reason"] == "overshot after 1 revision pass: A costs 4020 tokens against 2565"


def test_revision_passes_are_cached_so_a_rerun_is_free_and_identical(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    config = _config(budget_llm.base_url)
    first = run(tmp_path, config)
    calls = budget_llm.call_count

    second = run(tmp_path, config)

    assert budget_llm.call_count == calls
    assert story_files(second) == story_files(first)


@pytest.mark.parametrize(
    ("equal_budget", "message"),
    [
        ({"conditions": ["A", "C"]}, "equal_budget names unknown condition 'C'"),
        ({"conditions": ["A"]}, "equal_budget.conditions"),
        ({"conditions": ["A", "A"]}, "two different conditions"),
    ],
)
def test_equal_budget_must_name_two_conditions_of_the_run(
    budget_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    equal_budget: dict[str, Any],
    message: str,
) -> None:
    config = story_config(budget_llm.base_url, [BASELINE_A, BASELINE_B], equal_budget=equal_budget)

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(tmp_path / "r")])

    assert code == 2
    assert message in capsys.readouterr().err
    assert budget_llm.call_count == 0


def test_a_pass_that_lands_just_above_the_other_budget_but_within_tolerance_is_matched(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    # B costs 15 + 5 * 722 = 3625; one revision takes A to 4020, 10.9% above B but
    # within 10% of the larger budget, so the brief is matched with no overshoot.
    budget_llm.default = lambda body: _reply(body, scene_tokens=712)

    run_dir = run(tmp_path, _config(budget_llm.base_url))

    match = manifest(run_dir)["stories"]["equal_budget"]["matches"][0]
    assert (match["passes"], match["matched"], match["reason"]) == (1, True, None)


def test_the_report_shows_revision_passes_and_the_final_generations_attempts(
    budget_llm: FakeOpenAI, tmp_path: Path
) -> None:
    replies = iter(
        [
            completion(story_text(1500), completion_tokens=1000),
            completion(story_text(1600), completion_tokens=1000),
            completion(story_text(2050), completion_tokens=1000),
        ]
    )

    def revision_replies(body: Json) -> Reply:
        if "Revise" in last_user_message(body):
            return next(replies)
        return _reply(body)

    budget_llm.default = revision_replies
    config = _config(budget_llm.base_url)
    config["stories"]["briefs"] = {"ids": ["brief-01"]}

    run_dir = run(tmp_path, config)

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    # A: 2,050 words after the revision's third attempt, one revision pass.
    assert "| brief-01 | A | 1 | 2050 | 2000 | 3 | yes | 1 | ok |" in report
    assert "| brief-01 | B | 1 | 2000 | 2000 | 1 | yes | 0 | ok |" in report
