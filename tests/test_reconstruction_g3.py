"""Synthetic metrics test gate mechanics, never literary competence or scientific passage."""

import json
from pathlib import Path
from typing import Any

import pytest

from nde.corpus.records_io import create_record, file_sha, identity, read_sealed
from nde.reconstruction.g3 import assess, readiness, register, write_decision


def g3_inputs(
    directory: Path,
    *,
    margin: float = 0.1,
    candidate: float = 0.7,
    comparator: float = 0.6,
    valid: bool = True,
) -> tuple[Path, dict[str, Any]]:
    train = create_record(
        directory / "train.json",
        {
            "role": "optimizer",
            "split": "train",
            "codec_id": "codec",
            "test_only": True,
            "examples": [],
            "inputs": {},
        },
    )
    task = {
        "id": "task",
        "metadata": {"actual_rate": 0.4, "requested_rate": 0.4},
        "provenance": {"work_id": "work", "author_id": "author"},
    }
    evaluation = create_record(
        directory / "evaluation.json",
        {
            "role": "evaluator-only",
            "split": "test",
            "codec_id": "codec",
            "test_only": True,
            "examples": [{"tasks": [task]}],
            "inputs": {},
        },
    )
    weights = create_record(directory / "weights.json", {"weights": {"test": [0]}})
    base = create_record(
        directory / "model.json",
        {
            "test_only": True,
            "weights_sha256": file_sha(directory / "weights.json"),
            "parameter_sha256": identity(weights["weights"]),
            "parameter_count": 1,
        },
    )
    report = directory / "pair" / "scores.json"
    create_record(
        directory / "experiment.json",
        {
            "kind": "reconstruction-experiment",
            "test_only": True,
            "config": {
                "base_model": str(directory / "model.json"),
                "evaluation_dataset": str(directory / "evaluation.json"),
                "train_datasets": [str(directory / "train.json")],
                "selection_split": "validation",
                "training": {"inference_max_tokens": 256},
            },
            "jobs": [{"id": "d", "objective": "denoising"}, {"id": "a", "objective": "ar"}],
            "pairs": [
                {
                    "id": "pair",
                    "job_ids": ["d", "a"],
                    "score_path": str(report.resolve()),
                    "training_dataset_path": str(directory / "train.json"),
                    "training_dataset_id": train["id"],
                    "training_task_set_id": "training-tasks",
                    "seed": 7,
                    "training_budget": {
                        "max_steps": 1,
                        "max_tokens": 100,
                        "max_attention_cells": 100,
                        "max_seconds": 1,
                    },
                }
            ],
            "inputs": {},
        },
    )
    (directory / "reference.json").write_text('{"condition":"C"}')
    config = directory / "g3.json"
    config.write_text(
        json.dumps(
            {
                "training": str(directory / "train.json"),
                "evaluation": str(directory / "evaluation.json"),
                "base_model": str(directory / "model.json"),
                "experiment": str(directory / "experiment.json"),
                "score_reports": [str(report)],
                "overlap_report": str(directory / "overlap.json"),
                "reference": str(directory / "reference.json"),
                "positive_margin": margin,
                "validator_target": 1,
                "test_only": True,
            }
        )
    )

    def side(score: float, good: bool) -> dict[str, Any]:
        return {
            "status": "complete",
            "valid": good,
            "long_range_edges": {
                edge: {"expected": 1, "f1": score}
                for edge in ("causal", "setup_payoff", "knowledge")
            },
        }

    score: dict[str, Any] = {
        "kind": "paired-reconstruction-scores",
        "dataset_id": evaluation["id"],
        "codec_id": "codec",
        "matching_mismatches": [],
        "matching": {
            "base_id": base["id"],
            "initial_parameter_sha256": base["parameter_sha256"],
            "parameter_count": 1,
            "dataset_id": train["id"],
            "task_set_id": "training-tasks",
            "seed": 7,
            "training_budget": {
                "max_steps": 1,
                "max_tokens": 100,
                "max_attention_cells": 100,
                "max_seconds": 1,
            },
            "inference_budget": {"max_tokens": 256},
        },
        "test_only": True,
        "work": {
            side: {
                "training_steps": 1,
                "training_tokens": 1,
                "training_attention_cells": 1,
                "training_elapsed_seconds": 0.01,
                "inference_tokens": 1,
            }
            for side in ("candidate", "comparator")
        },
        "rows": [
            {
                "task_id": "task",
                "mask": task["metadata"],
                "provenance": task["provenance"],
                "candidate": side(candidate, valid),
                "comparator": side(comparator, True),
            }
        ],
        "inputs": {},
    }
    return config, {
        "score": score,
        "train_id": train["id"],
        "eval_id": evaluation["id"],
        "base_id": base["id"],
    }


def complete(directory: Path, fixture: dict[str, Any]) -> None:
    create_record(directory / "pair" / "scores.json", fixture["score"])
    create_record(
        directory / "overlap.json",
        {
            "training_id": fixture["train_id"],
            "query_id": fixture["eval_id"],
            "unresolved_overlap": False,
            "inputs": {},
        },
    )


def test_synthetic_margin_boundary_pass_remains_scientifically_pending(tmp_path: Path) -> None:
    config, fixture = g3_inputs(tmp_path)
    status = readiness(config)
    assert status["ready"] is False
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    complete(tmp_path, fixture)
    result = write_decision(protocol, tmp_path / "decision")
    assert result["criteria_outcome"] == "passed"
    assert result["outcome"] == "pending"
    assert result["owner_review"]["status"] == "missing"
    assert read_sealed(tmp_path / "decision" / "decision.json")["id"] == result["id"]


@pytest.mark.parametrize(
    ("candidate", "comparator", "valid"), [(0.69, 0.6, True), (0.5, 0.7, True), (0.8, 0.6, False)]
)
def test_small_reverse_margin_and_validator_failures_retain_reference(
    tmp_path: Path, candidate: float, comparator: float, valid: bool
) -> None:
    config, fixture = g3_inputs(tmp_path, candidate=candidate, comparator=comparator, valid=valid)
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    complete(tmp_path, fixture)
    result = assess(protocol)
    assert result["criteria_outcome"] == "failed"
    assert "Stop before Phase 7" in result["action"]
    assert (tmp_path / "reference.json").is_file()


def test_missing_results_and_stale_registered_data_are_pending(tmp_path: Path) -> None:
    config, fixture = g3_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    assert assess(protocol)["criteria_outcome"] == "pending"
    complete(tmp_path, fixture)
    (tmp_path / "weights.json").write_text("{}")
    assert "stale readiness" in " ".join(assess(protocol)["pending"])


def test_no_real_registration_without_passed_gates_and_no_late_registration(tmp_path: Path) -> None:
    config, fixture = g3_inputs(tmp_path)
    value = json.loads(config.read_text())
    value["test_only"] = False
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="not scientifically ready"):
        register(config, tmp_path / "real-protocol.json")
    value["test_only"] = True
    config.write_text(json.dumps(value))
    complete(tmp_path, fixture)
    with pytest.raises(ValueError, match="before expected scores exist"):
        register(config, tmp_path / "late-protocol.json")


def test_zero_advantage_cannot_pass_even_a_tiny_positive_margin(tmp_path: Path) -> None:
    config, fixture = g3_inputs(tmp_path, margin=1e-14, candidate=0.6, comparator=0.6)
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    complete(tmp_path, fixture)
    assert assess(protocol)["criteria_outcome"] == "failed"


@pytest.mark.parametrize("problem", ["budget", "coverage", "overlap"])
def test_unmatched_or_incomplete_evidence_cannot_pass(tmp_path: Path, problem: str) -> None:
    config, fixture = g3_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    if problem == "budget":
        fixture["score"]["matching_mismatches"] = ["training_budget"]
    if problem == "coverage":
        fixture["score"]["rows"] = []
    complete(tmp_path, fixture)
    if problem == "overlap":
        path = tmp_path / "overlap.json"
        value = read_sealed(path)
        path.unlink()
        create_record(path, {**value, "unresolved_overlap": True})
    assert assess(protocol)["criteria_outcome"] == "pending"


def test_synthetic_owner_review_cannot_satisfy_registered_meaning_requirement(
    tmp_path: Path,
) -> None:
    config, fixture = g3_inputs(tmp_path)
    value = json.loads(config.read_text())
    value.update({"require_meaning_review": True, "owner_review": str(tmp_path / "owner.json")})
    config.write_text(json.dumps(value))
    protocol = tmp_path / "protocol.json"
    register(config, protocol)
    complete(tmp_path, fixture)
    score = read_sealed(tmp_path / "pair" / "scores.json")
    create_record(
        tmp_path / "owner.json",
        {
            "kind": "owner-meaning-review",
            "reviewer": "fixture",
            "test_only": True,
            "score_ids": [score["id"]],
            "samples": [{"task_id": "task", "judgement": "preserved"}],
            "inputs": {},
        },
    )
    result = assess(protocol)
    assert result["owner_review"]["status"] == "test-only"
    assert result["criteria_outcome"] == "pending"


def test_g3_rejects_matrix_populations_outside_its_registered_overlap_audit(
    tmp_path: Path,
) -> None:
    config, _ = g3_inputs(tmp_path)
    experiment = tmp_path / "experiment.json"
    matrix = read_sealed(experiment)
    other = tmp_path / "unaudited-train.json"
    create_record(other, {"role": "optimizer", "split": "train", "inputs": {}})
    matrix["config"]["train_datasets"].append(str(other))
    experiment.unlink()
    create_record(experiment, matrix)
    with pytest.raises(ValueError, match="one audited training population"):
        readiness(config)
