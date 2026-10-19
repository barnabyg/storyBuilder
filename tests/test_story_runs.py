from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, completion, error
from tests.stories import (
    BASELINE_A,
    jsonl,
    manifest,
    reply_with_story,
    run,
    stories,
    story_config,
    story_files,
)


@pytest.fixture
def story_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = lambda _body: reply_with_story(2000)
    return fake_llm


def test_baseline_a_writes_one_story_per_brief(story_llm: FakeOpenAI, tmp_path: Path) -> None:
    run_dir = run(tmp_path, story_config(story_llm.base_url, [BASELINE_A]))

    records = stories(run_dir)
    assert [(r["brief"], r["condition"], r["repetition"]) for r in records] == [
        ("brief-01", "A", 0),
        ("brief-02", "A", 0),
    ]
    assert all(r["words"] == 2000 and r["status"] == "ok" for r in records)
    story = (run_dir / records[0]["path"] / "story.md").read_text(encoding="utf-8")
    assert story.split()[:3] == ["word", "word", "word"]
    # One call per brief, carrying the brief's text and target length.
    assert story_llm.call_count == 2
    prompt = story_llm.requests[0].body["messages"][-1]["content"]
    assert "lighthouse keeper" in prompt
    assert "about 2000 words" in prompt


def test_calls_are_attributed_to_brief_condition_and_step(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, story_config(story_llm.base_url, [BASELINE_A]))

    calls = jsonl(run_dir / "calls.jsonl")
    assert [(c["brief"], c["condition"], c["step"]) for c in calls] == [
        ("brief-01", "A", "story"),
        ("brief-02", "A", "story"),
    ]
    assert calls[0]["call"] == "A/brief-01/story"
    summary = manifest(run_dir)["stories"]
    assert summary["brief_ids"] == ["brief-01", "brief-02"]
    assert summary["conditions"] == [{"name": "A", "pipeline": "baseline_a", "params": {}}]


def test_the_report_lists_stories_with_words_and_cost_per_story_and_condition(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    run_dir = run(tmp_path, story_config(story_llm.base_url, [BASELINE_A]))

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Stories" in report
    assert "| brief-01 | A | 1 | 2000 |" in report
    assert "[story.md](stories/A/brief-01/r1/story.md)" in report
    assert "### Per condition" in report
    # 2 calls of 10 input and 2,000 output tokens at $0.1 / $0.2 per million.
    assert "| A | 2 | 0 | 2 | 20 | 4000 | $0.0008 |" in report


def test_a_rerun_is_free_and_reproduces_the_stories_byte_for_byte(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    config = story_config(story_llm.base_url, [BASELINE_A], repeat=2)
    first = run(tmp_path, config)
    calls = story_llm.call_count

    second = run(tmp_path, config)

    assert story_llm.call_count == calls == 4
    assert manifest(second)["cost"]["usd"] == 0.0
    assert story_files(second) == story_files(first)


def test_repetitions_use_their_own_seeds(story_llm: FakeOpenAI, tmp_path: Path) -> None:
    run_dir = run(tmp_path, story_config(story_llm.base_url, [BASELINE_A], repeat=2))

    seeds = [(r["brief"], r["repetition"], r["seed"]) for r in stories(run_dir)]
    assert seeds == [("brief-01", 0, 7), ("brief-01", 1, 8), ("brief-02", 0, 7), ("brief-02", 1, 8)]
    assert (run_dir / "stories" / "A" / "brief-02" / "r2" / "story.md").is_file()


def test_briefs_can_be_selected_by_split(story_llm: FakeOpenAI, tmp_path: Path) -> None:
    config = story_config(story_llm.base_url, [BASELINE_A], briefs={"split": "dev"})

    run_dir = run(tmp_path, config)

    assert len(stories(run_dir)) == 10
    assert manifest(run_dir)["stories"]["split_checksums"] == {}


def test_the_manifest_records_the_checksum_of_a_frozen_split(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "test").mkdir(parents=True)
    raw = yaml.safe_load((REPO / "benchmark" / "dev" / "brief-01.yaml").read_text("utf-8"))
    (benchmark / "test" / "brief-41.yaml").write_text(
        yaml.safe_dump(raw | {"id": "brief-41", "split": "test"}), encoding="utf-8"
    )
    assert main(["briefs", "freeze", "test", "--dir", str(benchmark)]) == 0
    config = story_config(
        story_llm.base_url, [BASELINE_A], briefs_dir=str(benchmark), briefs={"split": "test"}
    )

    summary = manifest(run(tmp_path, config))["stories"]

    assert summary["brief_ids"] == ["brief-41"]
    assert len(summary["split_checksums"]["test"]) == 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"briefs": {"ids": ["brief-01", "brief-99"]}}, "unknown brief 'brief-99'"),
        ({"conditions": [{"name": "A", "pipeline": "nope"}]}, "unknown pipeline 'nope'"),
        (
            {"conditions": [BASELINE_A | {"params": {"temperature": 2}}]},
            "condition 'A': invalid params",
        ),
        (
            {
                "conditions": [
                    BASELINE_A | {"params": {"prompt": {"template": "story_a", "version": 9}}}
                ]
            },
            "story_a@9 not found",
        ),
        ({"conditions": [BASELINE_A, BASELINE_A]}, "condition names must be unique"),
        ({"briefs": {}}, "name a split or brief ids"),
    ],
    ids=["brief", "pipeline", "params", "template", "duplicate", "empty-selection"],
)
def test_invalid_story_configs_are_rejected_before_any_network_call(
    story_llm: FakeOpenAI,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, Any],
    message: str,
) -> None:
    config = story_config(story_llm.base_url, [BASELINE_A])
    config["stories"] |= change
    runs = tmp_path / "runs"

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(runs)])

    assert code == 2
    assert message in capsys.readouterr().err
    assert story_llm.call_count == 0
    assert not runs.exists()


def test_a_pipeline_role_without_a_model_is_rejected(
    story_llm: FakeOpenAI, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = story_config(story_llm.base_url, [BASELINE_A])
    del config["roles"]["prose"]

    code = main(["run", str(write_config(tmp_path, config)), "--runs-dir", str(tmp_path / "r")])

    assert code == 2
    assert "condition 'A' uses role 'prose', which has no model" in capsys.readouterr().err


def test_a_provider_error_fails_the_run_but_leaves_manifest_and_report(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(reply_with_story(2000), error(400, "bad parameter"))

    run_dir = run(tmp_path, story_config(story_llm.base_url, [BASELINE_A]), expect=1)

    recorded = manifest(run_dir)
    assert recorded["status"] == "failed"
    assert recorded["error"]["call"] == "A/brief-02/story"
    assert [r["brief"] for r in stories(run_dir)] == ["brief-01"]
    assert "## Stories" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_a_budget_stop_aborts_the_run_but_leaves_manifest_and_report(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    config = story_config(story_llm.base_url, [BASELINE_A])
    config["roles"]["prose"]["params"] = {"max_tokens": 3000}
    # Each call reserves at least 3,000 output tokens at $0.2 per million: $0.0006.
    config["budget"] = {"per_run_usd": 0.001}

    run_dir = run(tmp_path, config, expect=1)

    assert manifest(run_dir)["abort_reason"] == "budget"
    assert story_llm.call_count == 1
    assert len(stories(run_dir)) == 1
    assert (run_dir / "report.md").is_file()


def test_call_list_configs_still_run_alongside_story_support(
    story_llm: FakeOpenAI, tmp_path: Path
) -> None:
    story_llm.queue(completion("A line."))
    config = story_config(story_llm.base_url, [BASELINE_A])
    config["calls"] = [
        {
            "id": "line",
            "role": "prose",
            "prompt": {"template": "smoke", "version": 1},
            "variables": {"setting": "a harbour"},
        }
    ]
    del config["stories"]

    run_dir = run(tmp_path, config)

    assert jsonl(run_dir / "outputs.jsonl")[0]["text"] == "A line."
    assert manifest(run_dir)["stories"] is None


def test_the_committed_baseline_a_config_runs_two_dev_briefs(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NDE_DEEPSEEK_BASE_URL", fake_llm.base_url)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    fake_llm.default = lambda _body: reply_with_story(2000)
    runs = tmp_path / "runs"

    code = main(["run", str(REPO / "configs" / "stories-a.yaml"), "--runs-dir", str(runs)])

    assert code == 0
    [run_dir] = list(runs.iterdir())
    assert [r["brief"] for r in stories(run_dir)] == ["brief-01", "brief-02"]
