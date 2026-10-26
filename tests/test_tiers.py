import json
import shutil
from pathlib import Path

import pytest
import yaml

from nde.cli import main
from tests.conftest import REPO, write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, completion, error
from tests.stories import jsonl, last_user_message, reply_with_story, story_config
from tests.test_baseline_b import outline
from tests.test_plan_generation import skeleton


def tier_provider(body: Json) -> Reply:
    system = str(body["messages"][0]["content"])
    request = last_user_message(body)
    if "whole-story skeleton planner" in system:
        plan = skeleton()
        plan["states"] = [
            {
                "id": "belief",
                "scene_id": "scene-0",
                "character_id": "keeper",
                "kind": "belief",
                "key": "duty",
                "before": {"status": "unknown"},
                "after": {"value": "BELIEF_CHANNEL_MARKER"},
            }
        ]
        plan["motifs"] = [
            {"id": "motif", "scene_ids": ["scene-0"], "reading": {"value": "MOTIF_CHANNEL_MARKER"}}
        ]
        return completion(json.dumps(plan))
    if "narrative scene-channel planner" in system:
        return completion(
            request.split("Accepted plan:\n", 1)[1].split("\nCURRENT_SCENE_ID=", 1)[0]
        )
    if "story planner" in system:
        return completion(json.dumps(outline()))
    return reply_with_story(400 if "Now write scene" in request else 2000)


@pytest.fixture
def tier_config(fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = tier_provider
    config = story_config(
        fake_llm.base_url,
        [{"name": "B", "pipeline": "baseline_b"}, {"name": "C", "pipeline": "baseline_c"}],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["B", "C"], "tolerance": 0.1, "max_revision_passes": 2},
    )
    shutil.copytree(REPO / "prompts", tmp_path / "prompts")
    (tmp_path / "benchmark/dev").mkdir(parents=True)
    shutil.copyfile(REPO / "benchmark/dev/brief-01.yaml", tmp_path / "benchmark/dev/brief-01.yaml")
    config["prompts_dir"] = str(tmp_path / "prompts")
    config["stories"]["briefs_dir"] = str(tmp_path / "benchmark")
    return write_config(tmp_path, config)


def test_tier_comparison_refuses_held_out_inputs_before_generation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = story_config(
        "http://127.0.0.1:1",
        [{"name": "B", "pipeline": "baseline_b"}, {"name": "C", "pipeline": "baseline_c"}],
        briefs={"ids": ["brief-11"]},
    )
    out = tmp_path / "comparison"
    assert main(["tiers", "compare", str(write_config(tmp_path, config)), "--out", str(out)]) == 2
    assert "dev-only" in capsys.readouterr().err
    assert not out.exists()


def test_tier_comparison_projects_all_channels_and_reports_pending_owner_evidence(
    tier_config: Path,
    fake_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    out = tmp_path / "comparison"
    assert main(["tiers", "compare", str(tier_config), "--out", str(out)]) == 0
    packet = json.loads((out / "comparison.json").read_text())
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["owner_status"] == "pending"
    assert set(packet["runs"]) == {"S", "M", "L"}
    for tier, run_path in packet["runs"].items():
        run = Path(run_path)
        [c] = [item for item in jsonl(run / "stories.jsonl") if item["condition"] == "C"]
        plan = json.loads((run / c["path"] / "plan.json").read_text())
        assert plan["tier"] == tier
        assert c["budget"]["calls"] == 11
        inputs = (run / c["path"] / "attempts/1/scene-01.inputs.json").read_text()
        assert ("BELIEF_CHANNEL_MARKER" in inputs) == (tier in {"M", "L"})
        assert ("MOTIF_CHANNEL_MARKER" in inputs) == (tier == "L")
        assert report["tiers"][tier]["realisation"][0]["owner_status"] == "pending"
    # Repeat report generation is byte-for-byte deterministic, including absent owner evidence.
    original = (out / "report.json").read_bytes()
    assert main(["tiers", "report", str(out)]) == 0
    assert (out / "report.json").read_bytes() == original
    prose_requests = [
        last_user_message(item.body)
        for item in fake_llm.requests
        if "Now write scene" in last_user_message(item.body)
    ]
    assert any("BELIEF_CHANNEL_MARKER" in request for request in prose_requests)


def test_synthetic_owner_selection_is_immutable_and_detects_changed_settings(
    tier_config: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "comparison"
    assert main(["tiers", "compare", str(tier_config), "--out", str(out)]) == 0
    packet = json.loads((out / "comparison.json").read_text())
    review = {
        "comparison_sha256": packet["id"],
        "reviewer": "synthetic owner",
        "test_only": True,
        "judgements": [
            {
                "brief": "brief-01",
                "tier": "L",
                "coherence": 4,
                "originality": 4,
                "note": "Synthetic test-only assessment",
            }
        ],
    }
    file = tmp_path / "review.json"
    file.write_text(json.dumps(review))
    selection = tmp_path / "selection.json"
    choice = [
        "tiers",
        "select",
        str(out),
        "--tier",
        "L",
        "--owner",
        "synthetic owner",
        "--rationale",
        "Synthetic test-only selection",
        "--test-only",
        "--out",
        str(selection),
    ]
    assert main(choice) == 2
    assert not selection.exists()
    assert main(["tiers", "review", str(out), str(file)]) == 0
    assert main(choice) == 0
    actual_choice = [value for value in choice if value != "--test-only"]
    actual_choice[-1] = str(tmp_path / "actual-selection.json")
    assert main(actual_choice) == 2
    chosen = json.loads(selection.read_text())
    assert chosen["test_only"] is True
    assert chosen["owner_sample"] == ["brief-01"]
    original = selection.read_bytes()
    assert main(choice) == 2
    assert selection.read_bytes() == original
    assert main(["tiers", "check", str(selection), str(tier_config)]) == 0
    config = yaml.safe_load(tier_config.read_text())
    config["roles"]["prose"]["params"]["temperature"] = 0.8
    write_config(tmp_path, config)
    assert main(["tiers", "check", str(selection), str(tier_config)]) == 1


@pytest.mark.parametrize("changed", ["dev-brief", "prompt", "selected-story"])
def test_tier_packet_refuses_selection_after_its_evidence_changes(
    tier_config: Path,
    tmp_path: Path,
    changed: str,
) -> None:
    out = tmp_path / "comparison"
    assert main(["tiers", "compare", str(tier_config), "--out", str(out)]) == 0
    packet = json.loads((out / "comparison.json").read_text())
    review = {
        "comparison_sha256": packet["id"],
        "reviewer": "synthetic owner",
        "test_only": True,
        "judgements": [
            {
                "brief": "brief-01",
                "tier": "L",
                "coherence": 4,
                "originality": 4,
                "note": "Synthetic test-only assessment",
            }
        ],
    }
    file = tmp_path / "review.json"
    file.write_text(json.dumps(review))
    assert main(["tiers", "review", str(out), str(file)]) == 0
    selection = tmp_path / "selection.json"
    assert (
        main(
            [
                "tiers",
                "select",
                str(out),
                "--tier",
                "L",
                "--owner",
                "synthetic owner",
                "--rationale",
                "Synthetic test-only selection",
                "--test-only",
                "--out",
                str(selection),
            ]
        )
        == 0
    )
    if changed == "dev-brief":
        path = tmp_path / "benchmark/dev/brief-01.yaml"
    elif changed == "prompt":
        path = tmp_path / "prompts/scene_prose/v1.yaml"
    else:
        run = Path(packet["runs"]["L"])
        [record] = [item for item in jsonl(run / "stories.jsonl") if item["condition"] == "C"]
        path = run / record["path"] / "story.md"
    path.write_text(path.read_text() + "\n# changed evidence\n")
    assert main(["tiers", "check", str(selection), str(tier_config)]) == 1
    if changed != "prompt":
        assert main(["tiers", "report", str(out)]) == 1
    if changed != "prompt":
        assert main(["tiers", "review", str(out), str(file)]) == 2


def test_tier_comparison_keeps_failed_pairs_and_does_not_select_them(
    tier_config: Path,
    fake_llm: FakeOpenAI,
    tmp_path: Path,
) -> None:
    fake_llm.queue(error(400, "synthetic provider failure"))
    out = tmp_path / "comparison"
    assert main(["tiers", "compare", str(tier_config), "--out", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert any("incomplete" in item for item in report["problems"])
    assert report["tiers"]["S"]["stories"][0]["status"] == "failed"
    assert (
        main(
            [
                "tiers",
                "select",
                str(out),
                "--tier",
                "L",
                "--owner",
                "synthetic owner",
                "--rationale",
                "Synthetic test-only selection",
                "--test-only",
                "--out",
                str(tmp_path / "selection.json"),
            ]
        )
        == 2
    )


def test_repeated_tier_comparison_reuses_cache_without_losing_experimental_costs(
    tier_config: Path,
    fake_llm: FakeOpenAI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NDE_CACHE", str(tmp_path / "shared-cache.sqlite"))
    first, second = tmp_path / "first", tmp_path / "second"
    assert main(["tiers", "compare", str(tier_config), "--out", str(first)]) == 0
    calls = fake_llm.call_count
    assert main(["tiers", "compare", str(tier_config), "--out", str(second)]) == 0
    assert fake_llm.call_count == calls
    a, b = [json.loads((folder / "report.json").read_text()) for folder in (first, second)]
    for tier in ("S", "M", "L"):
        assert [row["attributed_budget"] for row in a["tiers"][tier]["stories"]] == [
            row["attributed_budget"] for row in b["tiers"][tier]["stories"]
        ]
        assert all(row["new_spend_usd"] == 0 for row in b["tiers"][tier]["stories"])
