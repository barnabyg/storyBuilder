"""Approved corpus exports reach only development structural requests."""

import json
from pathlib import Path

import pytest

from nde.corpus import exemplars
from nde.corpus.exemplar_records import file_digest
from nde.corpus.records_io import create_record
from tests.conftest import write_config
from tests.fake_openai import FakeOpenAI, Json, Reply, error
from tests.g2_support import approved_export
from tests.stories import story_config
from tests.test_baseline_c import fake_c
from tests.test_baseline_d import critic_provider


@pytest.mark.parametrize("pipeline", ["baseline_c", "baseline_d"])
def test_approved_exemplars_reach_structural_requests_and_keep_full_experimental_budget(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pipeline: str
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    (tmp_path / "corpus").mkdir()
    dataset = approved_export(tmp_path / "corpus", fake_llm)
    monkeypatch.setenv("NDE_CACHE", str(tmp_path / "comparison-cache.sqlite"))
    start = fake_llm.call_count

    def provider(body: Json) -> Reply:
        value = critic_provider(body) if pipeline == "baseline_d" else fake_c(body)
        assert isinstance(value.body, dict)
        if "CORPUS_EXEMPLARS" in str(body["messages"]):
            value.body["usage"]["prompt_tokens"] = 60
        return value

    fake_llm.default = provider
    params = {"critic_iterations": 1} if pipeline == "baseline_d" else {}
    value = story_config(
        fake_llm.base_url,
        [
            {"name": name, "pipeline": pipeline, "params": params}
            for name in ("baseline", "exemplars")
        ],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["baseline", "exemplars"], "tolerance": 0.4},
    )
    for binding in value["roles"].values():
        binding["family"] = "generator"
    value["roles"]["critic"] = {"provider": "fake", "model": "fake-model", "family": "critic"}
    config = write_config(tmp_path, value)
    out = tmp_path / "comparison"
    result = exemplars.compare(config, dataset, ["lamp"], out)
    assert result["status"] == "complete"
    assert result["literary_improvement"] == "pending_owner_evidence"
    assert result["test_only"]
    [example] = result["examples"]["examples"]
    assert example["family"] == "alpha"
    assert set(example["targets"]["hard"]) == {"plot", "causality", "information", "goal", "belief"}
    assert "interpretation" not in example["targets"]["hard"]
    assert all("text" not in source for source in example["context"]["sources"])
    [pair] = result["pairs"]
    baseline, augmented = pair["conditions"]["baseline"], pair["conditions"]["exemplars"]
    assert augmented["calls"] == baseline["calls"] == (14 if pipeline == "baseline_d" else 11)
    assert baseline["input_tokens"] == (140 if pipeline == "baseline_d" else 110)
    assert augmented["input_tokens"] == (590 if pipeline == "baseline_d" else 410)
    requests = fake_llm.requests[start:]
    augmented_requests = [r for r in requests if "CORPUS_EXEMPLARS" in str(r.body)]
    assert len(augmented_requests) == (9 if pipeline == "baseline_d" else 6)
    assert all(
        '"family": "alpha"' in str(r.body["messages"][0]["content"]) for r in augmented_requests
    )
    assert all("Now write scene" not in str(r.body["messages"]) for r in augmented_requests)
    packet = json.loads((out / "comparison.json").read_text())
    run = Path(packet["run"])
    requests_saved = list(run.glob("stories/exemplars/*/*/planning/*/*.request.json"))
    assert len(requests_saved) == (9 if pipeline == "baseline_d" else 6)
    before = fake_llm.call_count
    replay = exemplars.compare(config, dataset, ["lamp"], tmp_path / "replay")
    assert fake_llm.call_count == before
    assert (
        replay["pairs"][0]["conditions"]["exemplars"]["input_tokens"] == augmented["input_tokens"]
    )
    assert replay["pairs"][0]["conditions"]["exemplars"]["new_spend_usd"] == 0
    if pipeline == "baseline_c":
        assert_synthetic_owner_review_pending(out)


def assert_synthetic_owner_review_pending(out: Path) -> None:
    result = json.loads((out / "report.json").read_text())
    [pair] = result["pairs"]
    baseline, augmented = pair["conditions"]["baseline"], pair["conditions"]["exemplars"]
    judgement = exemplars.DevJudgement(
        brief="brief-01",
        repetition=0,
        baseline_sha256=baseline["final_sha256"],
        exemplars_sha256=augmented["final_sha256"],
        baseline_coherence=3,
        exemplars_coherence=3,
        baseline_originality=3,
        exemplars_originality=3,
        notes="Synthetic review only.",
    )
    review = exemplars.DevReview(
        comparison_sha256=file_digest(out / "comparison.json"),
        reviewer="synthetic fixture",
        judgements=[judgement],
    )
    with pytest.raises(ValueError, match="test-only"):
        exemplars.review(out, review)
    review.test_only = True
    reviewed = exemplars.review(out, review)
    assert reviewed["literary_improvement"] == "pending_owner_evidence"
    assert len(reviewed["owner_reviews"]) == 1


def test_changed_export_annotation_is_rejected_without_provider_calls(
    fake_llm: FakeOpenAI, tmp_path: Path
) -> None:
    (tmp_path / "corpus").mkdir()
    dataset = approved_export(tmp_path / "corpus", fake_llm)
    saved = json.loads(dataset.read_text())
    annotation = Path(saved["records"][0]["annotation_path"])
    annotation.write_text(annotation.read_text() + "\n")
    config = story_config(
        fake_llm.base_url,
        [{"name": name, "pipeline": "baseline_c"} for name in ("baseline", "exemplars")],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["baseline", "exemplars"]},
    )
    before = fake_llm.call_count
    with pytest.raises(ValueError, match=r"missing or changed|changed|reproduce"):
        exemplars.compare(write_config(tmp_path, config), dataset, ["lamp"], tmp_path / "bad")
    assert fake_llm.call_count == before


@pytest.mark.parametrize("changed", ["author_id", "split"])
def test_export_author_or_protected_split_substitution_cannot_reach_a_provider(
    fake_llm: FakeOpenAI, tmp_path: Path, changed: str
) -> None:
    (tmp_path / "corpus").mkdir()
    dataset = approved_export(tmp_path / "corpus", fake_llm)
    saved = json.loads(dataset.read_text())
    saved["records"][0][changed] = "substituted-author" if changed == "author_id" else "validation"
    forged = tmp_path / "forged.json"
    create_record(forged, saved)
    config = story_config(
        fake_llm.base_url,
        [{"name": name, "pipeline": "baseline_c"} for name in ("baseline", "exemplars")],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["baseline", "exemplars"]},
    )
    before = fake_llm.call_count
    with pytest.raises(ValueError, match="protected/stale"):
        exemplars.compare(write_config(tmp_path, config), forged, ["lamp"], tmp_path / "bad")
    assert fake_llm.call_count == before


def test_failed_augmented_plan_keeps_incomplete_comparison_and_expected_pair(
    fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    (tmp_path / "corpus").mkdir()
    dataset = approved_export(tmp_path / "corpus", fake_llm)

    def fail_augmented(body: Json) -> Reply:
        return error(401, "synthetic failure") if "CORPUS_EXEMPLARS" in str(body) else fake_c(body)

    fake_llm.default = fail_augmented
    config = story_config(
        fake_llm.base_url,
        [{"name": name, "pipeline": "baseline_c"} for name in ("baseline", "exemplars")],
        briefs={"ids": ["brief-01"]},
        equal_budget={"conditions": ["baseline", "exemplars"]},
    )
    out = tmp_path / "comparison"
    result = exemplars.compare(write_config(tmp_path, config), dataset, ["lamp"], out)
    assert result["status"] == "incomplete"
    assert result["pairs"][0]["conditions"]["baseline"]["status"] == "ok"
    assert result["pairs"][0]["conditions"]["exemplars"]["status"] == "failed"
    assert result["pairs"][0]["problems"]
    assert exemplars.report(out)["status"] == "incomplete"
