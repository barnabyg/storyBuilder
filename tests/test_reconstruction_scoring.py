"""Known-answer saved-artifact scoring workflows."""

import copy
from pathlib import Path
from typing import Any

import pytest

from nde.corpus.records_io import create_record, identity
from nde.narrative.models import NarrativePlan
from nde.reconstruction.codec import encode_plan, seal
from nde.reconstruction.scoring import score_pair


def score_inputs(
    directory: Path,
    *,
    missing: bool = False,
    mutation: bool = False,
    duplicate: bool = False,
    soft: bool = False,
) -> tuple[Path, Path, Path]:
    plan = NarrativePlan.model_validate(
        {
            "id": "p",
            "title": "Plan",
            "scenes": [{"id": "s1", "title": "Start"}, {"id": "s2", "title": "End"}],
            "causal_links": [{"id": "edge", "cause_scene_id": "s1", "effect_scene_id": "s2"}],
        }
    )
    if duplicate:
        content = plan.model_dump(mode="json")
        content["scenes"].append({"id": "s3", "title": "Later"})
        content["causal_links"] = [
            {"id": "edge", "cause_scene_id": "s1", "effect_scene_id": "s3"},
            {"id": "edge2", "cause_scene_id": "s2", "effect_scene_id": "s3"},
        ]
        plan = NarrativePlan.model_validate(content)
    provenance = {
        "work_id": "work",
        "author_id": "author",
        "source_sha256": "source",
        "split": "test",
    }
    policy = "soft" if soft else "hard"
    record = encode_plan(plan, {"plot": "hard", "causality": policy}, provenance)
    slots = copy.deepcopy(record["slots"])
    positions = [
        index
        for index, slot in enumerate(slots)
        if (slot["path"] == ["causal_links", 0, "effect_scene_id"] and not duplicate)
        or (
            duplicate and slot["path"][0] == "causal_links" and slot["path"][-1] == "cause_scene_id"
        )
    ]
    targets: list[dict[str, Any]] = []
    for position in positions:
        targets.append(
            {"position": position, "value": slots[position]["value"], "policy": policy, "weight": 1}
        )
        slots[position]["value"] = None
        slots[position]["masked"] = True
    task = seal(
        {
            "codec_id": record["codec_id"],
            "record_id": record["id"],
            "visible": {"slots": slots},
            "allowed_positions": positions,
            "targets": targets,
            "metadata": {"kind": "fields", "requested_rate": 0.5, "actual_rate": 0.5},
            "provenance": provenance,
        }
    )
    data = create_record(
        directory / "dataset.json",
        {
            "codec_id": record["codec_id"],
            "vocabulary_id": record["vocabulary_id"],
            "role": "evaluator-only",
            "split": "test",
            "test_only": True,
            "examples": [{"id": "example", "record": record, "tasks": [task]}],
            "inputs": {},
        },
    )
    checkpoint = create_record(
        directory / "checkpoint.json", {"kind": "training-checkpoint", "inputs": {}}
    )
    matching = {
        "base_id": "base",
        "initial_parameter_sha256": "initial",
        "parameter_count": 100,
        "dataset_id": "train",
        "task_set_id": "training-tasks",
        "vocabulary_id": record["vocabulary_id"],
        "budget_policy": {"max_steps": 1},
        "seed": 7,
        "training_budget": {"max_steps": 1},
        "inference_budget": {"max_tokens": 256},
    }
    for objective in ("denoising", "ar"):
        replacements = [
            {"position": target["position"], "value": target["value"]} for target in targets
        ]
        if duplicate and objective == "denoising":
            replacements[1]["value"] = replacements[0]["value"]
        if mutation and objective == "denoising":
            replacements.append({"position": 0, "value": "mutated"})
        predictions = (
            []
            if missing and objective == "ar"
            else [
                {
                    "task_id": task["id"],
                    "status": "complete",
                    "replacements": replacements,
                    "diagnostics": [],
                }
            ]
        )
        create_record(
            directory / f"{objective}.json",
            {
                "kind": "reconstruction-predictions",
                "dataset_id": data["id"],
                "task_set_id": identity({"task_ids": [task["id"]]}),
                "codec_id": record["codec_id"],
                "vocabulary_id": record["vocabulary_id"],
                "checkpoint_id": checkpoint["id"],
                "checkpoint_path": str((directory / "checkpoint.json").resolve()),
                "objective": objective,
                "matching": matching,
                "work": {"training_steps": 1},
                "test_only": True,
                "predictions": predictions,
                "inputs": {},
            },
        )
    return directory / "dataset.json", directory / "denoising.json", directory / "ar.json"


def test_scores_only_masked_answers_and_keeps_missing_outputs_in_denominator(
    tmp_path: Path,
) -> None:
    inputs = score_inputs(tmp_path, missing=True)
    report = score_pair(*inputs, tmp_path / "scores")
    assert report["summary"]["candidate"]["hard"] == {"correct": 1, "total": 1, "accuracy": 1}
    assert report["summary"]["comparator"]["hard"] == {"correct": 0, "total": 1, "accuracy": 0}
    assert report["summary"]["comparator"]["validator_pass_rate"] == 0
    assert report["rows"][0]["candidate"]["edges"]["causal"]["f1"] == 1
    assert report["owner_review"]["status"] == "missing"
    assert (tmp_path / "scores" / "meaning-review.json").is_file()


def test_unmasked_mutations_fail_validation_even_with_correct_hidden_answer(tmp_path: Path) -> None:
    inputs = score_inputs(tmp_path, mutation=True)
    report = score_pair(*inputs, tmp_path / "scores")
    assert report["summary"]["candidate"]["validator_pass_rate"] == 0
    assert "unmasked-content" in " ".join(report["rows"][0]["candidate"]["diagnostics"])


def test_reverse_comparison_and_stale_predictions_are_rejected(tmp_path: Path) -> None:
    data, candidate, comparator = score_inputs(tmp_path)
    with pytest.raises(ValueError, match="positive direction"):
        score_pair(data, comparator, candidate, tmp_path / "reverse")
    candidate.write_text(candidate.read_text().replace('"denoising"', '"changed"'))
    with pytest.raises(ValueError, match="identity"):
        score_pair(data, candidate, comparator, tmp_path / "stale")


def test_local_edges_cannot_count_as_long_range_recovery(tmp_path: Path) -> None:
    inputs = score_inputs(tmp_path)
    report = score_pair(*inputs, tmp_path / "scores")
    assert report["rows"][0]["candidate"]["long_range_edges"]["causal"]["expected"] == 0
    assert report["rows"][0]["candidate"]["long_range_edges"]["causal"]["f1"] is None


def test_duplicate_edges_are_false_positives_and_partial_recovery_is_fractional(
    tmp_path: Path,
) -> None:
    inputs = score_inputs(tmp_path, duplicate=True)
    report = score_pair(*inputs, tmp_path / "scores")
    edge = report["rows"][0]["candidate"]["edges"]["causal"]
    assert edge["true_positive"] == 1
    assert edge["expected"] == 2
    assert edge["predicted"] == 2
    assert edge["f1"] == 0.5
    assert report["rows"][0]["comparator"]["long_range_edges"]["causal"]["f1"] == 1


def test_soft_targets_never_become_hard_accuracy_labels(tmp_path: Path) -> None:
    inputs = score_inputs(tmp_path, soft=True)
    report = score_pair(*inputs, tmp_path / "scores")
    assert report["rows"][0]["candidate"]["hard"]["total"] == 0
    assert report["rows"][0]["candidate"]["soft_diagnostic"]["accuracy"] == 1
    assert report["rows"][0]["candidate"]["edges"]["causal"]["expected"] == 0
