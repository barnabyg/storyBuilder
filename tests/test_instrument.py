import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde.cli import main
from nde.stories.scrambled import scramble
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, run, stories, story_config
from tests.test_baseline_b import BASELINE_B, fake_b, outline

B_SCRAMBLED = {"name": "B-scrambled", "pipeline": "baseline_b_scrambled"}


@pytest.mark.parametrize("scenes", range(2, 13))
def test_a_scramble_is_never_the_original_order(scenes: int) -> None:
    for seed in range(200):
        order = scramble(scenes, seed)
        assert sorted(order) == list(range(scenes))
        assert order != list(range(scenes))


def test_a_scramble_is_deterministic_and_varies_with_the_seed() -> None:
    assert scramble(6, 11) == scramble(6, 11)
    assert len({tuple(scramble(6, seed)) for seed in range(20)}) > 1


@pytest.fixture
def b_llm(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_b
    return fake_llm


def _scene_prompts(fake: FakeOpenAI) -> list[str]:
    return [
        last_user_message(r.body)
        for r in fake.requests
        if "Now write scene" in last_user_message(r.body)
    ]


def test_b_scrambled_reuses_bs_outline_in_a_shuffled_order(
    b_llm: FakeOpenAI, tmp_path: Path
) -> None:
    config = story_config(b_llm.base_url, [BASELINE_B, B_SCRAMBLED], briefs={"ids": ["brief-01"]})

    run_dir = run(tmp_path, config)

    intact, scrambled = stories(run_dir)
    folder = run_dir / scrambled["path"]
    permutation = json.loads((folder / "permutation.json").read_text(encoding="utf-8"))
    assert permutation != list(range(5))
    # B's outline is asked for once: B-scrambled gets it from the cache.
    planner_calls = [
        r for r in b_llm.requests if "story planner" in r.body["messages"][0]["content"]
    ]
    assert len(planner_calls) == 1
    assert json.loads((folder / "outline.json").read_text("utf-8")) == outline()
    # Its scenes follow the permutation of the outline's scenes.
    scrambled_prompts = _scene_prompts(b_llm)[5:]
    for position, original in enumerate(permutation):
        assert f"beat {original + 1}a" in scrambled_prompts[position].split("Now write scene")[1]
    assert intact["status"] == scrambled["status"] == "ok"


def test_b_scrambled_regenerates_its_prose(b_llm: FakeOpenAI, tmp_path: Path) -> None:
    config = story_config(b_llm.base_url, [BASELINE_B, B_SCRAMBLED], briefs={"ids": ["brief-01"]})

    run_dir = run(tmp_path, config)

    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text("utf-8").splitlines()]
    scenes = [c for c in calls if c["condition"] == "B-scrambled" and c["step"].startswith("scene")]
    assert len(scenes) == 5
    assert not any(c["cached"] for c in scenes)
    assert len(_scene_prompts(b_llm)) == 10


def _instrument_reply(body: Json) -> Reply:
    prompt = last_user_message(body)
    if "Score this short story" in prompt:
        scores = dict.fromkeys(
            [
                "engagement",
                "coherence",
                "emotional_impact",
                "character_believability",
                "pacing",
                "originality",
            ],
            3,
        )
        return completion(json.dumps(scores | {"failure_tags": []}))
    if "Story 1:" in prompt:
        return completion(json.dumps({"preferred": 1, "strength": 2, "reason": "r"}))
    return fake_b(body)


def _point_providers_at(fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> None:
    for provider in ("DEEPSEEK", "OPENAI", "ANTHROPIC", "MISTRAL"):
        monkeypatch.setenv(f"NDE_{provider}_BASE_URL", fake.base_url)
        monkeypatch.setenv(f"{provider}_API_KEY", "test-key")


def test_the_instrument_check_runs_end_to_end(
    fake_llm: FakeOpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_providers_at(fake_llm, monkeypatch)
    fake_llm.default = _instrument_reply
    runs = tmp_path / "runs"
    batches = tmp_path / "batches"

    instrument = REPO / "configs" / "p1-instrument.yaml"
    assert main(["run", str(instrument), "--runs-dir", str(runs)]) == 0
    [run_dir] = list(runs.iterdir())
    records = stories(run_dir)
    assert len(records) == 20
    assert {r["condition"] for r in records} == {"B", "B-scrambled"}
    assert all(r["status"] == "ok" for r in records)

    assert (
        main(
            [
                "eval",
                "batch",
                "p1-instrument",
                "--from",
                str(run_dir),
                "B",
                "--from",
                str(run_dir),
                "B-scrambled",
                "--batches-dir",
                str(batches),
                "--rubric-dir",
                str(REPO / "benchmark" / "rubric"),
            ]
        )
        == 0
    )

    judges: dict[str, Any] = yaml.safe_load(
        (REPO / "configs" / "p1-judges.yaml").read_text(encoding="utf-8")
    )
    judges["judges"]["batches_dir"] = str(batches)
    judges["prompts_dir"] = str(REPO / "prompts")
    assert main(["run", str(write_config(tmp_path, judges)), "--runs-dir", str(runs)]) == 0
    verdicts = list((batches / "p1-instrument" / "judges").glob("*.jsonl"))
    assert len(verdicts) == 1
    lines = verdicts[0].read_text(encoding="utf-8").splitlines()
    # 10 pairs x 2 judges x (2 orders + 2 stories).
    assert len(lines) == 80


def test_the_instrument_config_meets_the_plan() -> None:
    config = yaml.safe_load((REPO / "configs" / "p1-instrument.yaml").read_text(encoding="utf-8"))

    stories_section = config["stories"]
    assert stories_section["briefs"] == {"split": "dev"}
    assert [c["pipeline"] for c in stories_section["conditions"]] == [
        "baseline_b",
        "baseline_b_scrambled",
    ]
    assert config["budget"]["per_run_usd"] > 0


def test_the_pre_registration_is_complete() -> None:
    text = (REPO / "benchmark" / "prereg" / "g-instr.md").read_text(encoding="utf-8")

    for heading in (
        "Hypothesis",
        "Primary metric",
        "Sample size",
        "Pass thresholds",
        "Fail actions",
        "Status",
    ):
        assert f"## {heading}" in text
    assert "n = 10" in text
    assert "9 of 10" in text
    assert "80%" in text
    assert "TBD" not in text
