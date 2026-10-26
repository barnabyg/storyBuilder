import argparse
import copy
import json
from pathlib import Path

import pytest

from nde.narrative.models import NarrativePlan
from nde.reconstruction import codec, tasks
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export


def command(*arguments: str) -> int:
    parser = argparse.ArgumentParser()
    tasks.configure(parser)
    return tasks.run(parser.parse_args(arguments))


def test_saved_tasks_keep_answers_outside_visible_information(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    export = approved_export(tmp_path, fake_llm)
    encoded = tmp_path / "encoded"
    codec.create_encoding(export, encoded)
    config = tmp_path / "schedules.json"
    config.write_text(
        json.dumps(
            {
                "seed": 17,
                "schedules": [
                    {"kind": kind, "rate": 0.3} for kind in ("fields", "scenes", "channels", "span")
                ],
            }
        )
    )
    out = tmp_path / "tasks"
    assert command("create", str(encoded), "--schedule", str(config), "--out", str(out)) == 0
    assert command("inspect", str(out)) == 0
    saved = tasks.load_tasks(out)
    task = saved["tasks"][0]
    original = copy.deepcopy(task)
    predictions = {item["position"]: item["value"] for item in task["targets"]}
    assert tasks.candidate_check(task, predictions)["valid"] is True
    assert (
        tasks.reconstruct(task, predictions) == codec.load_encoding(encoded)["records"][0]["view"]
    )
    poisoned = copy.deepcopy(task)
    for target in poisoned["targets"]:
        target["value"] = "hidden answer sentinel"
    poisoned = codec.seal(poisoned)
    assert tasks.model_input(poisoned) == tasks.model_input(task)
    assert "hidden answer sentinel" not in codec.canonical_json(
        codec.detokenize(tasks.model_input(task))
    )
    assert task == original
    assert task["metadata"]["actual_rate"] != 0.3
    assert all(
        task["visible"]["slots"][position]["masked"] for position in task["allowed_positions"]
    )
    assert len(saved["tasks"]) == 8
    scene_task = saved["tasks"][1]
    assert all(
        slot["masked"]
        for slot in scene_task["visible"]["slots"]
        if slot["eligible"] and slot["channel"] == "causality"
    )
    config.write_text("{}")
    assert command("inspect", str(out)) == 2


def test_corruption_boundaries_empty_channels_and_immutable_observations() -> None:
    plan = NarrativePlan.model_validate(
        {
            "id": "p",
            "title": "t",
            "scenes": [
                {"id": f"s{index}", "title": f"Title {index}", "summary": f"Summary {index}"}
                for index in range(3)
            ],
        }
    )
    record = codec.encode_plan(plan, {"plot": "hard"}, {"work_id": "one", "split": "test"})
    source = copy.deepcopy(record)
    first = tasks.make_task(record, {"kind": "span", "rate": 0.1, "start": 0}, seed=3)
    assert first == tasks.make_task(record, {"kind": "span", "rate": 0.1, "start": 0}, seed=3)
    assert first["metadata"]["eligible_fields"] == 6
    assert first["metadata"]["masked_fields"] == 1
    assert first["metadata"]["actual_rate"] == 1 / 6
    last = tasks.make_task(record, {"kind": "span", "rate": 0.1, "start": 5})
    middle = tasks.make_task(record, {"kind": "span", "rate": 0.3, "start": 2})
    assert middle["metadata"]["masked_fields"] == 2
    assert (
        first["allowed_positions"][0]
        < middle["allowed_positions"][0]
        < last["allowed_positions"][0]
    )
    assert last["allowed_positions"][-1] > first["allowed_positions"][0]
    whole = tasks.make_task(record, {"kind": "span", "rate": 0.7, "count": 6})
    assert whole["metadata"]["actual_rate"] == 1.0
    empty = tasks.make_task(record, {"kind": "fields", "rate": 0.3, "channel": "information"})
    assert empty["allowed_positions"] == []
    assert empty["metadata"]["actual_rate"] == 0.0
    assert tasks.reconstruct(empty, {}) == record["view"]
    with pytest.raises(ValueError, match="observations immutable"):
        tasks.reconstruct(first, {999: "changed context"})
    assert record == source
    changed = copy.deepcopy(first)
    changed["codec_id"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        tasks.model_input(changed)
    for schedule in (
        {"kind": "fields", "rate": 0.09},
        {"kind": "fields", "rate": 0.71},
        {"kind": "span", "rate": 0.7, "start": 6},
        {"kind": "span", "rate": 0.7, "count": 7},
        {"kind": "fields", "rate": 0.3, "channel": "unregistered"},
    ):
        with pytest.raises(ValueError, match=r"mask rate|span|channel"):
            tasks.make_task(record, schedule)


def test_soft_loss_and_invalid_references_remain_explicit() -> None:
    plan = NarrativePlan.model_validate(
        {
            "id": "p",
            "title": "t",
            "scenes": [{"id": "start", "title": "Start"}, {"id": "finish", "title": "Finish"}],
            "causal_links": [
                {"id": "link", "cause_scene_id": "start", "effect_scene_id": "finish"}
            ],
        }
    )
    record = codec.encode_plan(plan, {"causality": "soft"}, {})
    task = tasks.make_task(record, {"kind": "fields", "rate": 0.7})
    assert all(
        target["weight"] == 0.25 and target["policy"] == "soft" for target in task["targets"]
    )
    predictions = {target["position"]: target["value"] for target in task["targets"]}
    reference = next(
        index
        for index in task["allowed_positions"]
        if task["visible"]["slots"][index]["path"][-1] == "effect_scene_id"
    )
    predictions[reference] = "missing-scene"
    checked = tasks.candidate_check(task, predictions)
    assert checked["valid"] is False
    assert checked["diagnostics"]
