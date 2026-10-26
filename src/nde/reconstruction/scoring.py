"""Paired masked-only scores with complete task denominators and separate owner review."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.narrative.validation import validate_plan
from nde.reconstruction.codec import decode_record, materialize_view, view_from_slots
from nde.reconstruction.overlap import canonical_view, load_view

EDGE_FIELDS = {
    "causal": ("causal_links",),
    "setup_payoff": ("payoff_links",),
    "knowledge": ("reveal_links", "knowledge_uses"),
}
LONG_RANGE_SCENE_DISTANCE = 2
MATCHING_FIELDS = (
    "base_id",
    "initial_parameter_sha256",
    "parameter_count",
    "dataset_id",
    "task_set_id",
    "vocabulary_id",
    "budget_policy",
    "seed",
    "training_budget",
    "inference_budget",
)


def dataset_tasks(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        decode_record(example["record"])
        for task in example["tasks"]:
            if task.get("id") != identity(task):
                raise ValueError("stale corruption task identity")
            if (
                task.get("record_id") != example["record"]["id"]
                or task.get("codec_id") != dataset["codec_id"]
            ):
                raise ValueError("task refers to a different encoded source or codec")
            result.append(task)
    ids = [task["id"] for task in result]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate corruption task identities")
    return result


def _candidate(task: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    slots = copy.deepcopy(task["visible"]["slots"])
    diagnostics: list[str] = []
    supplied: dict[int, Any] = {}
    if prediction is None or prediction.get("status") != "complete":
        diagnostics.append("missing or incomplete prediction")
    else:
        for item in prediction.get("replacements", []):
            position = item.get("position")
            if not isinstance(position, int) or position not in task["allowed_positions"]:
                diagnostics.append("unmasked-content change or invalid prediction position")
            elif position in supplied:
                diagnostics.append("duplicate prediction position")
            else:
                supplied[position] = item.get("value")
    if set(supplied) != set(task["allowed_positions"]):
        diagnostics.append("missing allowed prediction positions")
    for position, value in supplied.items():
        slots[position]["value"] = value
    view: dict[str, Any] | None = None
    valid = False
    try:
        view = view_from_slots(slots)
        validation = validate_plan(materialize_view(view))
        diagnostics.extend(f"{item.code}: {item.message}" for item in validation.diagnostics)
        valid = validation.valid and not diagnostics
    except (ValueError, TypeError, KeyError, IndexError) as error:
        diagnostics.append(f"invalid decode: {error}")
    return {
        "view": view,
        "valid": valid,
        "diagnostics": diagnostics,
        "supplied": supplied,
        "status": prediction.get("status", "missing") if prediction else "missing",
    }


def _fraction(correct: int, total: int) -> dict[str, Any]:
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def _edge_counts(expected: list[str], predicted: list[str]) -> dict[str, Any]:
    gold = set(expected)
    # Repeated predictions contribute false positives rather than extra true positives.
    true_positive = len(gold & set(predicted))
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall and precision
        else 0.0
        if gold
        else None
    )
    return {
        "true_positive": true_positive,
        "predicted": len(predicted),
        "expected": len(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _scene_index(reference: str) -> int:
    return int(reference.rsplit("/", 1)[-1])


def _long_range(view: dict[str, Any], field: str, index: int) -> bool:
    edge = view[field][index]
    try:
        if field == "causal_links":
            start = _scene_index(edge["cause_scene_id"])
            end = _scene_index(edge["effect_scene_id"])
        elif field == "payoff_links":
            setup = view["setups"][int(edge["setup_id"].rsplit("/", 1)[-1])]
            start, end = _scene_index(setup["scene_id"]), _scene_index(edge["payoff_scene_id"])
        else:
            end = _scene_index(edge["scene_id"])
            origins = [
                _scene_index(reveal["scene_id"])
                for reveal in view.get("reveal_links", [])
                if reveal["fact_id"] == edge["fact_id"] and _scene_index(reveal["scene_id"]) < end
            ]
            fact = view["knowledge_facts"][int(edge["fact_id"].rsplit("/", 1)[-1])]
            if fact["initial_holder_ids"]:
                origins.append(-1)
            if not origins:
                return False
            start = min(origins)
        return end - start >= LONG_RANGE_SCENE_DISTANCE
    except (ValueError, KeyError, IndexError, TypeError):
        return False


def _score_edges(
    gold: dict[str, Any],
    predicted_view: dict[str, Any],
    eligible: dict[str, set[int]],
    *,
    long_range: bool = False,
) -> dict[str, Any]:
    edges: dict[str, Any] = {}
    for name, fields in EDGE_FIELDS.items():
        expected: list[str] = []
        predicted: list[str] = []
        for field in fields:
            for index in sorted(eligible[field]):
                if long_range and not _long_range(gold, field, index):
                    continue
                if index < len(gold.get(field, [])):
                    expected.append(json.dumps([field, gold[field][index]], sort_keys=True))
                if index < len(predicted_view.get(field, [])):
                    predicted.append(
                        json.dumps([field, predicted_view[field][index]], sort_keys=True)
                    )
        edges[name] = _edge_counts(expected, predicted)
    return edges


def _score_task(task: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    candidate = _candidate(task, prediction)
    channels: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    gold_slots = copy.deepcopy(task["visible"]["slots"])
    eligible_edges: dict[str, set[int]] = defaultdict(set)
    for target in task["targets"]:
        position = target["position"]
        slot = gold_slots[position]
        slot["value"] = target["value"]
        policy = target["policy"]
        if policy not in {"hard", "soft"} or not slot["eligible"] or slot["policy"] != policy:
            raise ValueError("excluded targets cannot be authoritative scores")
        correct = (
            position in candidate["supplied"] and candidate["supplied"][position] == target["value"]
        )
        channels[slot["channel"]][policy].append(correct)
        path = slot["path"]
        if policy == "hard" and len(path) > 1 and isinstance(path[1], int):
            eligible_edges[str(path[0])].add(path[1])
    gold_view = canonical_view(view_from_slots(gold_slots))
    predicted_view = canonical_view(candidate["view"]) if candidate["view"] is not None else {}
    channel_scores = {
        channel: {
            policy: _fraction(sum(values), len(values)) for policy, values in policies.items()
        }
        for channel, policies in channels.items()
    }
    all_hard = [value for policies in channels.values() for value in policies["hard"]]
    all_soft = [value for policies in channels.values() for value in policies["soft"]]
    return {
        "status": candidate["status"],
        "valid": candidate["valid"],
        "diagnostics": candidate["diagnostics"],
        "hard": _fraction(sum(all_hard), len(all_hard)),
        "soft_diagnostic": _fraction(sum(all_soft), len(all_soft)),
        "channels": channel_scores,
        "edges": _score_edges(gold_view, predicted_view, eligible_edges),
        "long_range_edges": _score_edges(
            gold_view, predicted_view, eligible_edges, long_range=True
        ),
    }


def _read_predictions(path: Path, dataset: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    result = read_sealed(path)
    check_snapshot(result.get("inputs", {}))
    if (
        result.get("kind") != "reconstruction-predictions"
        or result.get("dataset_id") != dataset["id"]
        or result.get("codec_id") != dataset["codec_id"]
        or result.get("task_set_id") != identity({"task_ids": sorted(task_ids)})
    ):
        raise ValueError("predictions differ from exact evaluation dataset/tasks/codec")
    if result.get("vocabulary_id") != dataset.get("vocabulary_id"):
        raise ValueError("prediction vocabulary differs")
    checkpoint = result.get("checkpoint_path")
    checkpoint_record = read_sealed(Path(checkpoint)) if checkpoint else {}
    if checkpoint_record.get("id") != result.get("checkpoint_id"):
        raise ValueError("prediction checkpoint missing or changed")
    check_snapshot(checkpoint_record.get("binding", {}).get("inputs", {}))
    if "model_state" in checkpoint_record and identity(
        checkpoint_record["model_state"]
    ) != checkpoint_record.get("parameter_sha256"):
        raise ValueError("checkpoint parameter identity changed")
    predicted_ids = [item["task_id"] for item in result["predictions"]]
    if len(predicted_ids) != len(set(predicted_ids)) or set(predicted_ids) - set(task_ids):
        raise ValueError("duplicate or unexpected prediction task identities")
    return result


def score_pair(
    dataset_path: Path, candidate_path: Path, comparator_path: Path, out: Path
) -> dict[str, Any]:
    """Score every expected task, including invalid or absent model outputs."""
    dataset = load_view(dataset_path)
    if (
        dataset.get("split") not in {"validation", "test"}
        or dataset.get("role") != "evaluator-only"
    ):
        raise ValueError("paired scoring requires separately protected evaluation data")
    tasks = dataset_tasks(dataset)
    task_ids = [task["id"] for task in tasks]
    candidate = _read_predictions(candidate_path, dataset, task_ids)
    comparator = _read_predictions(comparator_path, dataset, task_ids)
    if candidate["objective"] != "denoising" or comparator["objective"] != "ar":
        raise ValueError(
            "candidate/comparator must be denoising/autoregressive in positive direction"
        )
    mismatches = [
        key
        for key in MATCHING_FIELDS
        if candidate["matching"].get(key) != comparator["matching"].get(key)
    ]
    if any(key not in candidate["matching"] for key in MATCHING_FIELDS):
        raise ValueError("incomplete matched-training provenance")
    by_side = [
        {item["task_id"]: item for item in result["predictions"]}
        for result in (candidate, comparator)
    ]
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item["id"]):
        rows.append(
            {
                "task_id": task["id"],
                "provenance": task.get("provenance", {}),
                "mask": task["metadata"],
                "candidate": _score_task(task, by_side[0].get(task["id"])),
                "comparator": _score_task(task, by_side[1].get(task["id"])),
            }
        )
    summary: dict[str, Any] = {}
    for side in ("candidate", "comparator"):
        summary[side] = {
            "expected": len(rows),
            "valid": sum(row[side]["valid"] for row in rows),
            "validator_pass_rate": sum(row[side]["valid"] for row in rows) / len(rows),
            "hard": _fraction(
                sum(row[side]["hard"]["correct"] for row in rows),
                sum(row[side]["hard"]["total"] for row in rows),
            ),
        }
    dataset_file = dataset_path / "dataset.json" if dataset_path.is_dir() else dataset_path
    result = create_record(
        out / "scores.json",
        {
            "kind": "paired-reconstruction-scores",
            "dataset_id": dataset["id"],
            "task_set_id": candidate["task_set_id"],
            "codec_id": dataset["codec_id"],
            "vocabulary_id": dataset["vocabulary_id"],
            "candidate_id": candidate["id"],
            "comparator_id": comparator["id"],
            "checkpoint_ids": [candidate["checkpoint_id"], comparator["checkpoint_id"]],
            "matching": candidate["matching"],
            "matching_mismatches": mismatches,
            "work": {"candidate": candidate["work"], "comparator": comparator["work"]},
            "rows": rows,
            "summary": summary,
            "test_only": dataset.get("test_only", False)
            or candidate.get("test_only", False)
            or comparator.get("test_only", False),
            "validator_scope": (
                "retained canonical view with excluded channels empty; "
                "not complete tier-L validation"
            ),
            "owner_review": {"status": "missing", "diagnostic_judges": "not owner evidence"},
            "inputs": snapshot([dataset_file, candidate_path, comparator_path]),
        },
    )
    create_record(
        out / "meaning-review.json",
        {
            "score_id": result["id"],
            "test_only": result["test_only"],
            "status": "missing",
            "samples": [
                {
                    "task_id": row["task_id"],
                    "source": row["provenance"],
                    "candidate_checkpoint": candidate["checkpoint_id"],
                    "comparator_checkpoint": comparator["checkpoint_id"],
                    "judgement": None,
                }
                for row in rows
            ],
        },
    )
    report_files(out, "Paired reconstruction scores", result)
    return result


def configure_score(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("comparator", type=Path)
    parser.add_argument("--out", type=Path, required=True)


def run_score(args: argparse.Namespace) -> int:
    try:
        score_pair(args.dataset, args.candidate, args.comparator, args.out)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde reconstruction-score: {error}", file=sys.stderr)
        return 2
