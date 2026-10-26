import json
from pathlib import Path

import pytest
import yaml

from nde.corpus import families
from nde.corpus.extraction_records import ExtractionOptions
from tests.fake_openai import FakeOpenAI, completion
from tests.test_extraction import extract_command, extraction_inputs, plan_fixture


def family_inputs(tmp_path: Path, fake: FakeOpenAI) -> tuple[Path, Path, Path]:
    work, segments, single = extraction_inputs(tmp_path, fake)
    config = yaml.safe_load(single.read_text())
    first = config.pop("annotator")
    second = {**first, "model": "second-model", "family": "two"}
    config["annotators"] = [first, second]
    config["prices"]["fake/second-model"] = config["prices"]["fake/fake-model"]
    path = tmp_path / "families.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return work, segments, path


def test_families_are_independent_and_align_renamed_entities(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    left, right = plan_fixture(work, segments), plan_fixture(work, segments)
    right["id"] = "other-plan"
    right["characters"][0]["id"] = "renamed"
    right["scenes"][0]["character_ids"] = ["renamed"]
    fake_llm.queue(*(completion(json.dumps(plan)) for plan in [left] * 3 + [right] * 3))
    out = tmp_path / "families"
    result = families.compare_families(
        path,
        work,
        segments,
        ExtractionOptions(out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite"),
    )
    assert result.status == "complete"
    report = families.family_report(out)
    assert [item["family"] for item in report["families"]] == ["one", "two"]
    plot = next(item for item in report["agreement"] if item["channel"] == "plot")
    assert plot["agreed"] == 1
    assert plot["unmatched"] == 0
    messages = fake_llm.requests[3].body["messages"]
    assert "keeper-plan" not in json.dumps(messages)
    assert report["owner_approved"] is False


def test_causality_agreement_includes_setup_claims(tmp_path: Path, fake_llm: FakeOpenAI) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    left, right = plan_fixture(work, segments), plan_fixture(work, segments)
    for plan, description in ((left, "A lamp promises rescue"), (right, "A lamp promises danger")):
        plan["setups"] = [
            {"id": "setup", "scene_id": plan["scenes"][0]["id"], "description": description}
        ]
    fake_llm.queue(*(completion(json.dumps(plan)) for plan in [left] * 3 + [right] * 3))
    out = tmp_path / "families"
    assert (
        families.compare_families(
            path,
            work,
            segments,
            ExtractionOptions(out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite"),
        ).status
        == "complete"
    )
    report = families.family_report(out)
    row = next(row for row in report["agreement"] if row["channel"] == "causality")
    assert (row["agreed"], row["total"]) == (0, 2)


@pytest.mark.parametrize(
    "fault", ["same-family", "missing-family", "same-binding", "provider-alias"]
)
def test_invalid_independence_cannot_make_calls(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    fault: str,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    config = yaml.safe_load(path.read_text())
    second = config["annotators"][1]
    if fault == "same-family":
        second["family"] = " ONE "
    elif fault == "missing-family":
        second.pop("family")
    elif fault == "same-binding":
        second["model"] = "fake-model"
    else:
        config["providers"]["alias"] = config["providers"]["fake"]
        second.update(provider="alias", model="fake-model")
        config["prices"]["alias/fake-model"] = config["prices"]["fake/fake-model"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert (
        extract_command(
            [
                "families",
                str(work),
                "--segments",
                str(segments),
                "--config",
                str(path),
                "--out",
                str(tmp_path / "families"),
            ]
        )
        == 2
    )
    assert fake_llm.call_count == 0


def test_conflicting_claims_remain_disagreements_and_reports_reproduce(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    structural = plan_fixture(work, segments)
    left, right = json.loads(json.dumps(structural)), json.loads(json.dumps(structural))
    for plan, value in ((left, "Save the boat"), (right, "Abandon the boat")):
        plan["states"] = [
            {
                "id": "goal",
                "scene_id": plan["scenes"][0]["id"],
                "character_id": "mara",
                "kind": "goal",
                "key": "boat",
                "before": {"status": "unknown"},
                "after": {"value": value},
            }
        ]
    fake_llm.queue(
        *(
            completion(json.dumps(plan))
            for plan in [structural, left, left, structural, right, right]
        )
    )
    out = tmp_path / "families"
    options = ExtractionOptions(out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite")
    assert families.compare_families(path, work, segments, options).status == "complete"
    report = families.family_report(out)
    goals = next(row for row in report["agreement"] if row["channel"] == "goal")
    assert (goals["agreed"], goals["unmatched"], goals["total"]) == (0, 2, 2)
    assert (
        len(next(row for row in report["disagreements"] if row["channel"] == "goal")["left_only"])
        == 1
    )
    assert families.compare_families(path, work, segments, options).status == "complete"
    assert families.family_report(out) == report
    assert fake_llm.call_count == 6


def test_partial_family_failure_never_becomes_complete(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    fake_llm.queue(
        *(completion(json.dumps(good)) for _ in range(3)), completion("{}"), completion("{}")
    )
    out = tmp_path / "families"
    options = ExtractionOptions(out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite")
    assert families.compare_families(path, work, segments, options).status == "incomplete"
    report = families.family_report(out)
    assert [row["status"] for row in report["families"]] == ["complete", "exhausted"]
    assert report["agreement"] == []
    assert families.compare_families(path, work, segments, options).status == "incomplete"
    assert fake_llm.call_count == 5


def test_nested_story_cap_survives_resumption_under_larger_cohort_cap(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    config = yaml.safe_load(path.read_text())
    config["budget"]["per_run_usd"] = 0.004
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    good = plan_fixture(work, segments)
    fake_llm.queue(
        *(
            completion(json.dumps(good), prompt_tokens=10000, completion_tokens=5000)
            for _ in range(6)
        )
    )
    out = tmp_path / "families"
    options = ExtractionOptions(
        out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite", "pilot", 1
    )
    assert families.compare_families(path, work, segments, options).status == "incomplete"
    calls = fake_llm.call_count
    assert calls == 2
    assert families.compare_families(path, work, segments, options).status == "incomplete"
    assert fake_llm.call_count == calls


def test_report_retains_unstarted_family_after_interruption(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work, segments, path = family_inputs(tmp_path, fake_llm)
    good = plan_fixture(work, segments)
    fake_llm.queue(*(completion(json.dumps(good)) for _ in range(3)))
    out = tmp_path / "families"
    options = ExtractionOptions(out, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite")
    mkdir = Path.mkdir

    def interrupted(
        path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if path == out / "family-1":
            raise OSError("simulated interruption before second family")
        mkdir(path, mode, parents, exist_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "mkdir", interrupted)
        with pytest.raises(OSError, match="interruption"):
            families.compare_families(path, work, segments, options)
    report = families.family_report(out)
    assert report["status"] != "complete"
    assert [row["status"] for row in report["families"]] == ["complete", "planned"]
    assert [row["name"] for row in report["families"][1]["passes"]] == [
        "structural",
        "state",
        "interpretive",
    ]
