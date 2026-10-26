import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from nde import briefs, tiers
from nde.cli import main
from nde.eval.batch import Source
from nde.eval.readiness import GateCohort, check_readiness
from nde.runs.records import read_manifest
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import last_user_message, reply_with_story, story_config, story_text
from tests.test_baseline_c import fake_c
from tests.test_baseline_d import BASELINE_D, critic_provider, fake_d
from tests.test_tiers import tier_provider


@pytest.fixture
def cd_inputs(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = tier_provider
    value = story_config(
        fake_llm.base_url,
        [{"name": "B", "pipeline": "baseline_b"}, {"name": "C", "pipeline": "baseline_c"}],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["B", "C"]},
    )
    (tmp_path / "benchmark/dev").mkdir(parents=True)
    shutil.copyfile(REPO / "benchmark/dev/brief-01.yaml", tmp_path / "benchmark/dev/brief-01.yaml")
    briefs.freeze(tmp_path / "benchmark", "dev")
    value["stories"]["briefs_dir"] = str(tmp_path / "benchmark")
    for binding in value["roles"].values():
        binding["family"] = "generator"
    source = write_config(tmp_path, value, "tiers.yaml")
    packet = tiers.compare(source, tmp_path / "tiers")
    tiers.review(
        tmp_path / "tiers",
        tiers.DevReview(
            comparison_sha256=packet.id,
            reviewer="synthetic owner",
            test_only=True,
            judgements=[
                tiers.DevJudgement(
                    brief="brief-01",
                    tier="L",
                    coherence=4,
                    originality=4,
                    note="Synthetic test fixture",
                )
            ],
        ),
    )
    selection = tmp_path / "selection.json"
    tiers.select(
        tmp_path / "tiers",
        tiers.Choice(
            tier="L",
            owner="synthetic owner",
            rationale="Synthetic test-only selection",
            test_only=True,
        ),
        selection,
    )
    value["stories"]["conditions"] = [
        {"name": "C", "pipeline": "baseline_c", "params": {"tier": "L"}},
        BASELINE_D | {"params": {"critic_iterations": 1}},
    ]
    value["stories"]["equal_budget"]["conditions"] = ["C", "D"]
    value["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "family": "critic"}
    fake_llm.default = cd_provider
    return write_config(tmp_path, value, "cd.yaml"), selection


def cd_provider(body: Json) -> Reply:
    system = str(body["messages"][0]["content"])
    if "global intentions planner" in system or "coarse-conditioned skeleton planner" in system:
        reply = fake_d(body)
        assert isinstance(reply.body, dict)
        return completion(reply.body["choices"][0]["message"]["content"], prompt_tokens=1000)
    if "structural revision critic" in system or "masked structural revision planner" in system:
        return critic_provider(body)
    if "Revise" in last_user_message(body):
        return completion(story_text(2000, marker="revised"), completion_tokens=2000)
    return fake_c(body)


def test_cd_command_matches_complete_attribution_selects_final_prose_and_replays_free(
    cd_inputs: tuple[Path, Path],
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, selection = cd_inputs
    monkeypatch.setenv("NDE_CACHE", str(tmp_path / "cd-cache.sqlite"))

    def execute(folder: Path) -> dict[str, Any]:
        assert (
            main(
                [
                    "compare-cd",
                    "run",
                    str(config),
                    "--selection",
                    str(selection),
                    "--out",
                    str(folder),
                ]
            )
            == 0
        )
        return json.loads((folder / "report.json").read_text())

    first = execute(tmp_path / "cd")
    [pair] = first["pairs"]
    assert pair["comparable"]
    assert pair["matching"]["revised"] == "C"
    assert pair["conditions"]["D"]["calls"] == 14
    assert pair["conditions"]["C"]["revision_passes"] == 1
    assert pair["conditions"]["C"]["calls"] == 12
    run = Path(json.loads((tmp_path / "cd/cd-comparison.json").read_text())["run"])
    assert "revised" in (run / pair["conditions"]["C"]["path"]).read_text()
    assert first["test_only_selection"]
    summary = read_manifest(run).stories
    assert summary is not None
    cohort = GateCohort(
        id="synthetic-cd-fixture",
        brief_checksums=first["briefs"],
        split_checksums=summary.split_checksums,
        conditions=("D", "C"),
        length_tolerance=0.1,
        budget_tolerance=0.1,
    )
    sources = (Source(run, "D"), Source(run, "C"))
    assert check_readiness(cohort, tmp_path / "benchmark", sources).status == "ready"
    before = fake_llm.call_count
    second = execute(tmp_path / "replay")
    assert fake_llm.call_count == before
    assert all(row["new_spend_usd"] == 0 for row in second["pairs"][0]["conditions"].values())
    assert second["pairs"][0]["conditions"]["D"]["tokens"] == pair["conditions"]["D"]["tokens"]
    original = (tmp_path / "cd/report.json").read_bytes()
    assert main(["compare-cd", "report", str(tmp_path / "cd")]) == 0
    assert (tmp_path / "cd/report.json").read_bytes() == original
    selected = run / pair["conditions"]["C"]["path"]
    selected.write_text(selected.read_text() + " changed final artifact")
    assert check_readiness(cohort, tmp_path / "benchmark", sources).status == "incomplete"
    assert main(["compare-cd", "report", str(tmp_path / "cd")]) == 1
    assert json.loads((tmp_path / "cd/report.json").read_text())["problems"]


@pytest.mark.parametrize("change", ["tier", "prose", "planner"])
def test_cd_rejects_substituted_selected_baseline_before_new_calls(
    cd_inputs: tuple[Path, Path],
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    change: str,
) -> None:
    config, selection = cd_inputs
    value = yaml.safe_load(config.read_text())
    if change == "tier":
        value["stories"]["conditions"][0]["params"]["tier"] = "M"
    else:
        value["roles"][change]["params"]["temperature"] = 0.99
    config.write_text(yaml.safe_dump(value))
    before = fake_llm.call_count
    assert (
        main(
            [
                "compare-cd",
                "run",
                str(config),
                "--selection",
                str(selection),
                "--out",
                str(tmp_path / "bad"),
            ]
        )
        == 2
    )
    assert fake_llm.call_count == before


@pytest.mark.parametrize("failure", ["overshot", "interrupted", "omitted", "length"])
def test_cd_keeps_expected_pair_for_unmatched_failed_or_out_of_range_stories(
    cd_inputs: tuple[Path, Path],
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    failure: str,
) -> None:
    config, selection = cd_inputs
    if failure == "omitted":
        value = yaml.safe_load(config.read_text())
        value["stories"]["conditions"].reverse()
        config.write_text(yaml.safe_dump(value))

    def broken(body: Json) -> Reply:
        if failure == "overshot" and "Revise" in last_user_message(body):
            return completion(story_text(2000), completion_tokens=10000)
        if failure in {"interrupted", "omitted"} and any(
            label in str(body["messages"][0]["content"])
            for label in ("structural revision critic", "masked structural revision planner")
        ):
            return critic_provider(body, "provider")
        if failure == "length" and "Now write scene" in last_user_message(body):
            return reply_with_story(100)
        return cd_provider(body)

    fake_llm.default = broken
    out = tmp_path / "failed"
    assert (
        main(["compare-cd", "run", str(config), "--selection", str(selection), "--out", str(out)])
        == 1
    )
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "incomplete"
    [pair] = report["pairs"]
    assert set(pair["conditions"]) == {"C", "D"}
    assert not pair["comparable"]
    assert pair["conditions"]["D"]["calls"] >= 8
    assert all(
        "token_cost_usd" in row and "new_spend_usd" in row for row in pair["conditions"].values()
    )
    if failure == "omitted":
        assert pair["conditions"]["C"]["missing"]
        assert pair["conditions"]["C"]["token_cost_usd"] == 0
