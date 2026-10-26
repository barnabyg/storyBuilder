"""Frozen, author-grouped CPU learning curves with one resumable training engine."""

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import Field, model_validator

from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    read_json,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.llm.config import Strict
from nde.reconstruction.codec import seal
from nde.reconstruction.dataset import load_dataset
from nde.reconstruction.scoring import score_pair
from nde.reconstruction.training import (
    TrainingConfig,
    dataset_tasks,
    inspect_checkpoint,
    inspect_model,
    preflight,
    reconstruct,
    task_set_id,
    train,
)
from nde.reconstruction.training_data import create_subset, dataset_file, load_training_dataset


class ExperimentConfig(Strict):
    version: Literal["1"] = "1"
    base_model: Path
    train_datasets: list[Path] = Field(min_length=1)
    evaluation_dataset: Path
    seeds: list[int] = Field(default_factory=lambda: [7], min_length=1)
    subsets: list[Literal[25, 50, 100]] = Field(default_factory=lambda: [25, 50, 100])
    training: dict[str, Any] = Field(default_factory=dict[str, Any])
    max_jobs: int = Field(default=100, ge=1, le=1000)
    max_total_tokens: int = Field(default=100000000, ge=1)
    max_total_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    selection_split: Literal["validation"] = "validation"
    test_only: bool = False

    @model_validator(mode="after")
    def unique_axes(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or not self.subsets:
            raise ValueError("seeds and subset axes must be nonempty and unique")
        if sorted(set(self.subsets)) != self.subsets:
            raise ValueError("subsets must be unique and increasing")
        if len(set(self.train_datasets)) != len(self.train_datasets):
            raise ValueError("training datasets must be unique")
        return self


def register_experiment(config_path: Path, out: Path) -> dict[str, Any]:
    config = ExperimentConfig.model_validate(read_json(config_path))
    cardinality = len(config.train_datasets) * len(config.seeds) * len(config.subsets) * 2
    if cardinality > config.max_jobs:
        raise ValueError("matrix exceeds the registered maximum job count")
    evaluation = load_dataset(config.evaluation_dataset)
    if evaluation["role"] != "evaluator-only" or evaluation["split"] not in {"validation", "test"}:
        raise ValueError("sweep evaluation must use explicitly authorized held-out labels")
    if {"dataset", "base_model", "seed", "objective", "test_only"} & set(config.training):
        raise ValueError("training overrides cannot replace frozen matrix identities")
    base = inspect_model(config.base_model)
    pairs: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    files = [
        config_path,
        dataset_file(config.evaluation_dataset),
        config.base_model / "model.json",
        config.base_model / "weights.json",
    ]
    phase_seconds = math.nextafter(config.max_total_seconds / (cardinality * 2), 0.0)
    out.mkdir(parents=True, exist_ok=True)
    for dataset_index, training_path in enumerate(config.train_datasets):
        original = load_training_dataset(training_path)
        files.append(dataset_file(training_path))
        for seed in config.seeds:
            for percent in config.subsets:
                pair_id = f"data-{dataset_index}-seed-{seed}-subset-{percent}"
                subset_dir = out / "subsets" / pair_id
                subset = create_subset(training_path, subset_dir, percent, seed=seed)
                subset_path = (subset_dir / "dataset.json").resolve()
                files.append(subset_path)
                template = TrainingConfig.model_validate(
                    {
                        "dataset": str(subset_path),
                        "base_model": str(config.base_model.resolve()),
                        "seed": seed,
                        "test_only": config.test_only,
                        **config.training,
                        "max_seconds": min(
                            float(config.training.get("max_seconds", 3600)), phase_seconds
                        ),
                        "inference_max_seconds": min(
                            float(config.training.get("inference_max_seconds", 3600)), phase_seconds
                        ),
                    }
                )
                pair = {
                    "id": pair_id,
                    "job_ids": [f"{pair_id}-denoising", f"{pair_id}-ar"],
                    "score_path": str((out / "pairs" / pair_id / "scores.json").resolve()),
                    "seed": seed,
                    "subset_percent": percent,
                    "training_dataset_path": str(subset_path),
                    "training_dataset_id": subset["id"],
                    "training_task_set_id": task_set_id(dataset_tasks(subset)),
                    "granularity": original["granularity"],
                    "groups": subset["groups"],
                    "training_budget": {
                        key: template.model_dump(mode="json")[key]
                        for key in (
                            "max_steps",
                            "max_tokens",
                            "max_attention_cells",
                            "max_seconds",
                            "max_output_tokens",
                            "learning_rate",
                        )
                    },
                    "inference_budget": {
                        "max_tokens": template.inference_max_tokens,
                        "max_seconds": template.inference_max_seconds,
                    },
                    "source_dataset_id": original["id"],
                    "generation_costs": original.get("generation_costs", {}),
                    "source_groups": original.get("source_groups", subset["groups"]),
                    "human_control": original.get("human_control"),
                }
                pairs.append(pair)
                for objective in ("denoising", "ar"):
                    job_id = f"{pair_id}-{objective}"
                    job_dir = out / "jobs" / job_id
                    job_dir.mkdir(parents=True)
                    job_config = template.model_copy(update={"objective": objective})
                    save_json(job_dir / "config.json", job_config.model_dump(mode="json"))
                    checked = preflight(job_dir / "config.json")
                    files.append(job_dir / "config.json")
                    jobs.append(
                        {
                            "id": job_id,
                            "pair_id": pair_id,
                            "objective": objective,
                            "config_path": str((job_dir / "config.json").resolve()),
                            "prediction_path": str(
                                (job_dir / "predictions/predictions.json").resolve()
                            ),
                            "binding": checked,
                            "seed": seed,
                            "subset_percent": percent,
                            "granularity": original["granularity"],
                            "time_reservation_seconds": template.max_seconds
                            + template.inference_max_seconds,
                        }
                    )
    return create_record(
        out / "experiment.json",
        {
            "kind": "reconstruction-experiment",
            "config": config.model_dump(mode="json"),
            "base_id": base["id"],
            "evaluation_dataset_id": evaluation["id"],
            "evaluation_task_set_id": task_set_id(dataset_tasks(evaluation)),
            "pairs": pairs,
            "jobs": jobs,
            "expected_scores": [pair["score_path"] for pair in pairs],
            "inputs": snapshot(files),
            "test_only": config.test_only,
            "selection_policy": "validation only; test evidence never selects a frozen recipe",
            "subset_policy": "nested seeded author prefixes; ceil rounding; whole related families",
        },
    )


def load_experiment(path: Path) -> dict[str, Any]:
    path = path / "experiment.json" if path.is_dir() else path
    protocol = read_sealed(path)
    if protocol.get("kind") != "reconstruction-experiment":
        raise ValueError("not a reconstruction experiment protocol")
    check_snapshot(protocol["inputs"])
    config = ExperimentConfig.model_validate(protocol["config"])
    if inspect_model(config.base_model)["id"] != protocol["base_id"]:
        raise ValueError("sweep base model changed")
    if load_dataset(config.evaluation_dataset)["id"] != protocol["evaluation_dataset_id"]:
        raise ValueError("sweep evaluation changed")
    for job in protocol["jobs"]:
        if preflight(Path(job["config_path"])) != job["binding"]:
            raise ValueError("sweep training inputs changed")
    return protocol


def _job_result(job: dict[str, Any], protocol: dict[str, Any], steps: int | None) -> dict[str, Any]:
    config_path = Path(job["config_path"])
    directory = config_path.parent
    training = train(
        config_path,
        directory / "training",
        resume=(directory / "training/latest.json").exists(),
        max_steps_this_call=steps,
    )
    result = {
        "id": job["id"],
        "status": training["status"],
        "training_tokens": training["tokens"],
        "training_steps": training["step"],
        "evaluation_task_set_id": protocol["evaluation_task_set_id"],
        "pair_id": job["pair_id"],
        "objective": job["objective"],
    }
    result["reserved_tokens"] = _reservation(job)
    result["reserved_seconds"] = job["time_reservation_seconds"]
    if training["status"] != "complete":
        return result
    checkpoint = directory / "training/checkpoints" / f"step-{training['step']:08d}.json"
    prediction_path = directory / "predictions/predictions.json"
    if prediction_path.exists():
        predictions = read_sealed(prediction_path)
        check_snapshot(predictions["inputs"])
    else:
        predictions = reconstruct(
            checkpoint, Path(protocol["config"]["evaluation_dataset"]), directory / "predictions"
        )
    result.update(
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_id": predictions["checkpoint_id"],
            "predictions_path": str(prediction_path.resolve()),
            "prediction_id": predictions["id"],
            "work": predictions["work"],
        }
    )
    result["reserved_tokens"] = training["tokens"] + predictions["work"]["inference_tokens"]
    return result


def _score_pairs(protocol: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    indexed = {job["id"]: job for job in jobs}
    reports: list[str] = []
    for pair in protocol["pairs"]:
        sides = [indexed[job_id] for job_id in pair["job_ids"]]
        if not all(side["status"] == "complete" for side in sides):
            continue
        path = Path(pair["score_path"])
        if not path.exists():
            score_pair(
                Path(protocol["config"]["evaluation_dataset"]),
                Path(sides[0]["predictions_path"]),
                Path(sides[1]["predictions_path"]),
                path.parent,
            )
        score = read_sealed(path)
        check_snapshot(score["inputs"])
        reports.append(str(path))
    return reports


def run_experiment(
    path: Path, *, max_jobs_this_call: int | None = None, steps_per_call: int | None = None
) -> dict[str, Any]:
    protocol = load_experiment(path)
    directory = path if path.is_dir() else path.parent
    config = ExperimentConfig.model_validate(protocol["config"])
    if max_jobs_this_call is not None and max_jobs_this_call < 1:
        raise ValueError("max-jobs-this-call must be positive")
    # The OS releases this advisory lock when an interrupted process exits.
    lock_api = cast(Any, importlib.import_module("msvcrt" if os.name == "nt" else "fcntl"))
    with (directory / ".running").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            lock_api.locking(stream.fileno(), lock_api.LK_NBLCK, 1)
        else:
            lock_api.flock(stream.fileno(), lock_api.LOCK_EX | lock_api.LOCK_NB)
        try:
            return _run_locked(protocol, directory, config, max_jobs_this_call, steps_per_call)
        finally:
            if os.name == "nt":
                stream.seek(0)
                lock_api.locking(stream.fileno(), lock_api.LK_UNLCK, 1)
            else:
                lock_api.flock(stream.fileno(), lock_api.LOCK_UN)


def _reservation(job: dict[str, Any]) -> int:
    budget = job["binding"]["config"]
    return int(budget["max_tokens"]) + int(budget["inference_max_tokens"])


def _failed_job(job: dict[str, Any], protocol: dict[str, Any], error: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": job["id"],
        "status": "failed",
        "error": str(error),
        "evaluation_task_set_id": protocol["evaluation_task_set_id"],
        "reserved_tokens": _reservation(job),
        "reserved_seconds": job["time_reservation_seconds"],
        "accounting": "full reservation retained for failed/unknown work",
    }
    try:
        checkpoints = sorted(
            (Path(job["config_path"]).parent / "training/checkpoints").glob("step-*.json")
        )
        if checkpoints:
            checkpoint = inspect_checkpoint(checkpoints[-1])
            result.update(
                {
                    "training_steps": checkpoint["step"],
                    "training_tokens": checkpoint["tokens"],
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_path": str(checkpoints[-1]),
                }
            )
    except (ValueError, OSError, KeyError):
        result["accounting"] = "checkpoint unavailable; full reservation retained"
    return result


def _run_locked(
    protocol: dict[str, Any],
    directory: Path,
    config: ExperimentConfig,
    job_limit: int | None,
    steps: int | None,
) -> dict[str, Any]:
    progress: dict[str, Any]
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = read_sealed(progress_path)
        if progress["protocol_id"] != protocol["id"]:
            raise ValueError("resume experiment protocol changed")
    else:
        progress = {
            "protocol_id": protocol["id"],
            "elapsed_seconds": 0.0,
            "jobs": [
                {
                    "id": job["id"],
                    "status": "planned",
                    "evaluation_task_set_id": protocol["evaluation_task_set_id"],
                }
                for job in protocol["jobs"]
            ],
        }
    attempted = 0
    for index, job in enumerate(protocol["jobs"]):
        if progress["jobs"][index]["status"] in {"complete", "failed", "resource-stop"}:
            continue
        reserved = sum(row.get("reserved_tokens", 0) for row in progress["jobs"])
        additional = 0 if progress["jobs"][index].get("reserved_tokens") else _reservation(job)
        seconds = math.fsum(float(row.get("reserved_seconds", 0.0)) for row in progress["jobs"])
        new_seconds = (
            0.0
            if progress["jobs"][index].get("reserved_seconds")
            else job["time_reservation_seconds"]
        )
        if (
            (job_limit is not None and attempted >= job_limit)
            or progress["elapsed_seconds"] >= config.max_total_seconds
            or reserved + additional > config.max_total_tokens
            or seconds + new_seconds > config.max_total_seconds
        ):
            break
        progress["jobs"][index]["reserved_tokens"] = _reservation(job)
        progress["jobs"][index]["reserved_seconds"] = job["time_reservation_seconds"]
        progress["jobs"][index]["status"] = "running"
        save_json(progress_path, seal(progress))
        started = time.monotonic()
        try:
            result = _job_result(job, protocol, steps)
        except (ValueError, OSError, RuntimeError) as error:
            result = _failed_job(job, protocol, error)
        progress["jobs"][index] = result
        progress["elapsed_seconds"] += time.monotonic() - started
        attempted += 1
        save_json(progress_path, seal(progress))
    reports = _score_pairs(protocol, progress["jobs"])
    report = {
        "protocol_id": protocol["id"],
        "jobs": progress["jobs"],
        "complete_jobs": sum(row["status"] == "complete" for row in progress["jobs"]),
        "expected_jobs": len(protocol["jobs"]),
        "reserved_tokens": sum(row.get("reserved_tokens", 0) for row in progress["jobs"]),
        "reserved_seconds": math.fsum(
            float(row.get("reserved_seconds", 0.0)) for row in progress["jobs"]
        ),
        "actual_training_tokens": sum(row.get("training_tokens", 0) for row in progress["jobs"]),
        "actual_inference_tokens": sum(
            row.get("work", {}).get("inference_tokens", 0) for row in progress["jobs"]
        ),
        "source_groups_and_generation": [
            {
                key: pair[key]
                for key in (
                    "id",
                    "source_dataset_id",
                    "source_groups",
                    "generation_costs",
                    "human_control",
                )
            }
            for pair in protocol["pairs"]
        ],
        "score_reports": reports,
        "elapsed_seconds": progress["elapsed_seconds"],
        "test_only": protocol["test_only"],
        "scientific_status": "pending",
        "status": "complete"
        if all(row["status"] == "complete" for row in progress["jobs"])
        else "partial",
        "failures": sum(row["status"] == "failed" for row in progress["jobs"]),
        "budget_policy": (
            "reserve training+inference caps before launch; "
            "failures retain reservation; completed jobs release unused allowance"
        ),
        "selection_split": "validation",
    }
    report_files(directory, "Paired reconstruction learning curves", report)
    return report


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="experiment_command", required=True)
    freeze = commands.add_parser("register")
    freeze.add_argument("config", type=Path)
    freeze.add_argument("--out", type=Path, required=True)
    for name in ("preflight", "run", "resume", "report"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
        if name in {"run", "resume"}:
            command.add_argument("--max-jobs-this-call", type=int)
            command.add_argument("--steps-per-call", type=int)


def run(args: argparse.Namespace) -> int:
    try:
        if args.experiment_command == "register":
            result = register_experiment(args.config, args.out)
        elif args.experiment_command in {"run", "resume"}:
            result = run_experiment(
                args.path,
                max_jobs_this_call=args.max_jobs_this_call,
                steps_per_call=args.steps_per_call,
            )
        elif args.experiment_command == "report":
            load_experiment(args.path)
            result = read_json(
                (args.path if args.path.is_dir() else args.path.parent) / "report.json"
            )
        else:
            result = load_experiment(args.path)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"nde reconstruction-experiment: {error}", file=sys.stderr)
        return 2
