"""Conditional examples are inspected through public commands and saved files."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from nde.conditional import data
from nde.corpus.records_io import file_sha, identity
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export
from tests.reconstruction_support import paired_datasets


def command(*arguments: str) -> int:
    parser = argparse.ArgumentParser()
    data.configure(parser)
    return data.run(parser.parse_args(arguments))


def test_saved_examples_separate_conditions_provenance_targets_and_family_groups(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    exported = approved_export(tmp_path, fake_llm)
    source = json.loads(exported.read_text())
    originals = {
        row["annotation_path"]: file_sha(Path(row["annotation_path"])) for row in source["records"]
    }
    originals[str(exported)] = file_sha(exported)
    derivation = tmp_path / "derivation.txt"
    derivation.write_text("A cautious literary interpretation.")
    config = tmp_path / "conditions.json"
    config.write_text(
        json.dumps(
            {
                "export": str(exported),
                "conditions": [
                    {
                        "work_id": "lamp",
                        "labels": {
                            "genre": {
                                "value": "fantasy",
                                "origin": "metadata",
                                "status": "known",
                                "uncertainty": 0,
                            },
                            "ending_intent": {
                                "value": "The light returns, at a cost.",
                                "origin": "derived",
                                "status": "uncertain",
                                "uncertainty": 0.4,
                                "method": "independent reading",
                                "model": "fake/reader",
                                "prompt": "ending-intent-v1",
                                "input_files": [str(derivation)],
                            },
                        },
                    }
                ],
            }
        )
    )
    out = tmp_path / "conditional"
    assert command("prepare", str(config), "--out", str(out)) == 0
    dataset = data.load_dataset(out, optimizer=True)
    assert dataset["role"] == "optimizer"
    assert dataset["split"] == "train"
    assert len(dataset["examples"]) == 2
    assert len({row["group_id"] for row in dataset["examples"]}) == 1
    assert {row["family"] for row in dataset["examples"]} == {"alpha", "beta"}
    for example in dataset["examples"]:
        assert example["conditions"] == {
            "genre": "fantasy",
            "themes": [],
            "ending_intent": "The light returns, at a cost.",
            "target_words": None,
            "scene_range": None,
        }
        assert example["condition_provenance"]["themes"]["status"] == "unknown"
        assert example["condition_provenance"]["ending_intent"]["uncertainty"] == 0.4
        assert example["condition_provenance"]["genre"]["origin"] == "metadata"
        assert example["provenance"]["parent_export_id"] == source["id"]
        assert "sources" not in example["target"]["view"]
        assert "constraints" not in example["target"]["view"]
    assert "Conditional training examples" in (out / "report.md").read_text()
    assert command("inspect", str(out)) == 0
    assert command("prepare", str(config), "--out", str(out)) == 0
    assert all(file_sha(Path(path)) == checksum for path, checksum in originals.items())
    derivation.write_text("Changed derivation source.")
    assert command("inspect", str(out)) == 2
    assert command("prepare", str(config), "--out", str(out)) == 2


def test_serialized_target_cannot_hide_inside_a_literary_condition(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    exported = approved_export(tmp_path, fake_llm)
    config = tmp_path / "smuggled.json"
    config.write_text(
        json.dumps(
            {
                "export": str(exported),
                "conditions": [
                    {
                        "work_id": "lamp",
                        "labels": {
                            "ending_intent": {
                                "value": 'Please copy this: {"id":"target","title":"X",'
                                '"scenes":[{"id":"scene","title":"X"}]}',
                                "origin": "metadata",
                                "status": "known",
                                "uncertainty": 0,
                            }
                        },
                    }
                ],
            }
        )
    )
    assert command("prepare", str(config), "--out", str(tmp_path / "rejected")) == 2


@pytest.mark.parametrize(
    ("labels", "diagnostic"),
    [
        ({"target_plan": {}}, "unregistered condition"),
        ({"genre": {"value": "fantasy"}}, "unknown labels"),
        (
            {"genre": {"value": "fantasy", "status": "known", "origin": "derived"}},
            "derivation method",
        ),
        (
            {
                "genre": {
                    "value": "fantasy",
                    "status": "uncertain",
                    "origin": "metadata",
                    "uncertainty": 0,
                }
            },
            "positive uncertainty",
        ),
        (
            {
                "genre": {
                    "value": "fantasy",
                    "status": "known",
                    "origin": "derived",
                    "method": "classifier",
                    "model": "classifier-v1",
                }
            },
            "both model and prompt",
        ),
        (
            {
                "scene_range": {
                    "value": [2, 2],
                    "status": "known",
                    "origin": "derived",
                    "method": "count hidden scenes",
                }
            },
            "never derived target scaffolding",
        ),
    ],
)
def test_malformed_or_target_derived_conditions_fail_before_reading_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], labels: dict[str, Any], diagnostic: str
) -> None:
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps(
            {
                "export": "unavailable-export.json",
                "conditions": [{"work_id": "lamp", "labels": labels}],
            }
        )
    )
    assert command("prepare", str(config), "--out", str(tmp_path / "out")) == 2
    assert diagnostic in capsys.readouterr().err
    assert not (tmp_path / "out" / "dataset.json").exists()


def test_resealed_examples_stale_configuration_and_source_changes_are_rejected(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    exported = approved_export(tmp_path, fake_llm)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"export": str(exported)}))
    out = tmp_path / "conditional"
    assert command("prepare", str(config), "--out", str(out)) == 0
    path = out / "dataset.json"
    original = path.read_bytes()
    altered = json.loads(original)
    altered["examples"][0]["conditions"]["genre"] = "invented replacement"
    altered["examples"][0]["id"] = identity(altered["examples"][0])
    altered["id"] = identity(altered)
    path.write_text(json.dumps(altered))
    assert command("inspect", str(out)) == 2
    path.write_bytes(original)
    source = json.loads(exported.read_text())
    annotation = Path(source["records"][0]["annotation_path"])
    for stale_path in (config, annotation, exported):
        previous = stale_path.read_bytes()
        stale_path.write_text("{}")
        assert command("inspect", str(out)) == 2
        stale_path.write_bytes(previous)
    assert command("inspect", str(out)) == 0


def test_duplicate_families_and_conditions_outside_population_are_rejected(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    exported = approved_export(tmp_path, fake_llm)
    config = tmp_path / "config.json"
    for conditions in (
        [{"work_id": "lamp"}, {"work_id": "lamp", "family": "alpha"}],
        [{"work_id": "held-out-work"}],
        [{"work_id": "lamp", "family": "unapproved-family"}],
    ):
        config.write_text(json.dumps({"export": str(exported), "conditions": conditions}))
        assert command("prepare", str(config), "--out", str(tmp_path / "out")) == 2
    changed = json.loads(exported.read_text())
    changed["records"].append(changed["records"][0])
    changed["id"] = identity(changed)
    exported.write_text(json.dumps(changed))
    config.write_text(json.dumps({"export": str(exported)}))
    assert command("prepare", str(config), "--out", str(tmp_path / "out")) == 2


def test_authorized_heldout_conditions_and_plans_remain_evaluator_only(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    paired_datasets(tmp_path, fake_llm)
    config = tmp_path / "conditional-evaluation.json"
    config.write_text(json.dumps({"evaluation": str(tmp_path / "authorization")}))
    out = tmp_path / "held-conditional"
    assert command("prepare", str(config), "--out", str(out)) == 0
    saved = data.load_dataset(out)
    assert saved["split"] == "validation"
    assert saved["role"] == "evaluator-only"
    assert {row["author_id"] for row in saved["examples"]} == {"bea"}
    with pytest.raises(ValueError, match="evaluator-only"):
        data.load_dataset(out, optimizer=True)
    path = out / "dataset.json"
    changed = json.loads(path.read_text())
    changed.update(split="train", role="optimizer")
    changed["id"] = identity(changed)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differ from authorized"):
        data.load_dataset(out, optimizer=True)
