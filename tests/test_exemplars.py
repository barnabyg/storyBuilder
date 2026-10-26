"""Development exemplars use public configurations and inspect actual provider inputs."""

from pathlib import Path

import pytest

from nde.cli import main
from nde.corpus import exemplars
from tests.conftest import write_config
from tests.fake_openai import FakeOpenAI
from tests.stories import jsonl, run, stories, story_config
from tests.test_baseline_c import BASELINE_C, fake_c


def test_disabled_exemplars_preserve_existing_c_requests_and_accounting(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    fake_llm.default = fake_c
    config = story_config(
        fake_llm.base_url,
        [BASELINE_C | {"params": {"exemplars": None}}],
        briefs={"ids": ["brief-01"]},
    )
    result = run(tmp_path, config)
    [record] = stories(result)
    assert record["status"] == "ok"
    assert record["budget"]["calls"] == 11
    assert all("CORPUS_EXEMPLARS" not in str(request.body) for request in fake_llm.requests)
    assert {call["template"] for call in jsonl(result / "calls.jsonl")} == {
        "plan_skeleton",
        "plan_channels",
        "scene_prose",
    }


def test_missing_exemplar_export_rejected_before_any_provider_call(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    config = story_config(
        fake_llm.base_url,
        [
            BASELINE_C
            | {
                "params": {
                    "exemplars": {
                        "dataset": str(tmp_path / "missing/dataset.json"),
                        "dataset_sha256": "0" * 64,
                        "work_ids": ["human-story"],
                    }
                }
            }
        ],
        briefs={"ids": ["brief-01"]},
    )
    assert main(["run", str(write_config(tmp_path, config))]) == 2
    assert fake_llm.call_count == 0


@pytest.mark.parametrize("change", ["held_out", "different_parameters", "different_pipeline"])
def test_comparison_rejects_uncontrolled_inputs_before_opening_export_or_provider(
    fake_llm: FakeOpenAI, tmp_path: Path, change: str
) -> None:
    config = story_config(
        fake_llm.base_url,
        [
            {"name": "baseline", "pipeline": "baseline_c"},
            {"name": "exemplars", "pipeline": "baseline_c"},
        ],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["baseline", "exemplars"]},
    )
    if change == "held_out":
        config["stories"]["briefs"] = {"ids": ["brief-11"]}
    elif change == "different_parameters":
        config["stories"]["conditions"][1]["params"] = {"tier": "M"}
    else:
        config["stories"]["conditions"][1]["pipeline"] = "baseline_d"
    with pytest.raises(ValueError, match=r"same C or D|dev-only"):
        exemplars.compare(
            write_config(tmp_path, config),
            tmp_path / "missing-export",
            ["human"],
            tmp_path / "comparison",
        )
    assert fake_llm.call_count == 0
    assert not (tmp_path / "comparison").exists()
