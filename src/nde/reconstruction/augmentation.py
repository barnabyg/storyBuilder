"""Registered, explicitly synthetic training variants with unchanged human controls."""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.llm.config import Strict
from nde.reconstruction import codec
from nde.reconstruction.dataset import load_dataset
from nde.reconstruction.scoring import score_pair
from nde.reconstruction.tasks import make_task
from nde.reconstruction.training_data import load_training_dataset

RECIPE = "training-label-template-v1"
GENERATOR = "nde-local-textual-perturbation-v1"
PROMPT = (
    "Produce an explicitly synthetic textual variant using only these retained training labels."
)
TEXT_FIELDS = {"title", "summary", "description", "value"}
type Record = dict[str, Any]


class AugmentationConfig(Strict):
    dataset: Path
    recipe: Literal["training-label-template-v1"] = RECIPE
    seed: int = 7
    synthetic_per_human: int = Field(default=1, ge=0, le=4)
    source_example_ids: list[str] | None = None
    selection_basis: Literal["pre-registered", "validation"] = "pre-registered"
    selection_validation: Path | None = None
    max_generated_examples: int = Field(default=10000, ge=0, le=100000)
    max_generation_tokens: int = Field(default=1000000, ge=1)

    @model_validator(mode="after")
    def selection_is_explicit(self) -> "AugmentationConfig":
        if (self.selection_basis == "validation") != (self.selection_validation is not None):
            raise ValueError("validation recipe selection requires separate validation evidence")
        return self


class AblationConfig(Strict):
    augmentation: Path
    human_scores: Path
    augmented_scores: Path


def _file(path: Path) -> Path:
    return path / "dataset.json" if path.is_dir() else path


def _parent(path: Path) -> Record:
    source = load_training_dataset(path)
    if source["split"] != "train" or source["role"] != "optimizer":
        raise ValueError("evaluator-only or protected data cannot generate training variants")
    if any(row.get("synthetic", False) for row in source["examples"]):
        raise ValueError("nested synthetic augmentation is not a registered recipe")
    return source


def _synthetic(example: Record, variant: int, seed: int) -> tuple[Record, Record]:
    parent = example["record"]
    labels = [
        {"path": slot["path"], "value": slot["value"]}
        for slot in parent["slots"]
        if slot["eligible"]
        and slot["path"][-1] in TEXT_FIELDS
        and isinstance(slot["value"], str)
        and slot["value"]
    ]
    if not labels:
        raise ValueError("registered recipe requires an eligible textual training label")
    generation = {
        "recipe": RECIPE,
        "generator": GENERATOR,
        "prompt_version": "1",
        "prompt": PROMPT,
        "permitted_labels": labels,
        "parent_example_id": example["id"],
        "parent_record_id": parent["id"],
        "seed": seed,
        "variant": variant,
        "source_eligibility": "approved optimizer training view only",
    }
    generation["id"] = identity(generation)
    tag = f"Synthetic {generation['id'][:8]}: "
    slots = copy.deepcopy(parent["slots"])
    permitted = {tuple(label["path"]) for label in labels}
    for slot in slots:
        if tuple(slot["path"]) in permitted:
            slot["value"] = tag + slot["value"]
    view = codec.view_from_slots(slots)
    provenance = {
        **parent["provenance"],
        "synthetic": True,
        "generation_id": generation["id"],
        "parent_example_id": example["id"],
    }
    record = codec.encode_plan(
        codec.materialize_view(view),
        parent["target_policy"],
        provenance,
        max_scenes=parent["capacity"]["max_scenes"],
        max_tokens=parent["capacity"]["max_tokens"],
        ineligible_channels=tuple(parent["ineligible_channels"]),
    )
    tasks = [
        make_task(
            record,
            task["metadata"]["schedule"],
            seed=task["metadata"]["seed"],
            repetition=task["metadata"]["repetition"],
        )
        for task in example["tasks"]
    ]
    generated = {
        **copy.deepcopy(example),
        "record": record,
        "tasks": tasks,
        "synthetic": True,
        "generation_id": generation["id"],
        "parent_example_id": example["id"],
    }
    generated["id"] = identity(generated)
    return generated, generation


def _derive(source: Record, config: AugmentationConfig) -> Record:
    selected = config.source_example_ids
    known = {example["id"] for example in source["examples"]}
    if selected is not None and (len(set(selected)) != len(selected) or not set(selected) <= known):
        raise ValueError("augmentation sources must be unique eligible training example IDs")
    eligible = [
        example for example in source["examples"] if selected is None or example["id"] in selected
    ]
    if len(eligible) * config.synthetic_per_human > config.max_generated_examples:
        raise ValueError("augmentation exceeds registered example resource limit")
    generated: list[Record] = []
    generations: list[Record] = []
    for example in eligible:
        for variant in range(config.synthetic_per_human):
            synthetic, generation = _synthetic(example, variant, config.seed)
            generated.append(synthetic)
            generations.append(generation)
    input_tokens = sum(
        len(codec.tokenize({"prompt": PROMPT, "permitted_labels": generation["permitted_labels"]}))
        for generation in generations
    )
    output_tokens = sum(len(row["record"]["tokens"]) for row in generated)
    if input_tokens + output_tokens > config.max_generation_tokens:
        raise ValueError("augmentation exceeds registered generation-token resource limit")
    human_count = len(source["examples"])
    synthetic_count = len(generated)
    return {
        "examples": [*copy.deepcopy(source["examples"]), *generated],
        "generation": generations,
        "synthetic_example_ids": [row["id"] for row in generated],
        "source_groups": sorted({row["author_id"] for row in eligible}),
        "source_works": sorted({row["work_id"] for row in eligible}),
        "independent_works": source["independent_works"],
        "independent_authors": source["independent_authors"],
        "mixture": {
            "human_examples": human_count,
            "synthetic_examples": synthetic_count,
            "synthetic_fraction": synthetic_count / (human_count + synthetic_count),
        },
        "generation_costs": {
            "provider_calls": 0,
            "usd": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "token_unit": "fixed UTF-8 byte codec",
            "local_text_transformations": sum(len(row["permitted_labels"]) for row in generations),
        },
    }


def _selection(config: AugmentationConfig) -> list[Path]:
    if config.selection_validation is None:
        return []
    evidence = load_dataset(config.selection_validation)
    if evidence["split"] != "validation" or evidence["role"] != "evaluator-only":
        raise ValueError("held-out test labels cannot select an augmentation recipe")
    return [_file(config.selection_validation)]


def create_augmentation(config_path: Path, out: Path) -> Record:
    config = AugmentationConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    source_path = _file(config.dataset)
    source = _parent(source_path)
    selection = _selection(config)
    record = create_record(
        out / "dataset.json",
        {
            "kind": "reconstruction-augmentation",
            "role": "optimizer",
            "split": "train",
            "parent_path": str(source_path.resolve()),
            "parent_id": source["id"],
            "config": config.model_dump(mode="json"),
            "recipe": RECIPE,
            "generator": GENERATOR,
            "codec_id": source["codec_id"],
            "vocabulary_id": source["vocabulary_id"],
            "granularity": source["granularity"],
            "test_only": source["test_only"],
            "inputs": snapshot([source_path, config_path, *selection]),
            "human_control": {
                "dataset_path": str(source_path.resolve()),
                "dataset_id": source["id"],
                "policy": "unchanged; report human-only results separately",
            },
            "scientific_status": "unestablished; augmentation cannot authorize G3 or Phase 7",
            **_derive(source, config),
        },
    )
    report_files(out, "Synthetic training augmentation (separate ablation)", record)
    return record


def load_augmentation(path: Path) -> Record:
    record = read_sealed(_file(path))
    if record.get("kind") != "reconstruction-augmentation":
        raise ValueError("not a registered augmentation artifact")
    check_snapshot(record["inputs"])
    config = AugmentationConfig.model_validate(record["config"])
    source = _parent(Path(record["parent_path"]))
    _selection(config)
    expected = _derive(source, config)
    if (
        record["parent_id"] != source["id"]
        or Path(record["parent_path"]).resolve() != _file(config.dataset).resolve()
        or record["role"] != "optimizer"
        or record["split"] != "train"
        or any(
            record[field] != source[field]
            for field in ("codec_id", "vocabulary_id", "granularity", "test_only")
        )
        or any(record[field] != value for field, value in expected.items())
    ):
        raise ValueError("augmentation escaped its frozen source, mixture or recipe")
    return record


def _score_inputs(path: Path) -> tuple[Record, dict[str, Path]]:
    score = read_sealed(path)
    if score.get("kind") != "paired-reconstruction-scores":
        raise ValueError("ablation requires paired score reports")
    check_snapshot(score["inputs"])
    paths: dict[str, Path] = {}
    for name in score["inputs"]:
        bound = read_sealed(Path(name))
        if bound.get("kind") == "reconstruction-dataset":
            paths["dataset"] = Path(name)
        elif bound.get("kind") == "reconstruction-predictions":
            paths[bound["objective"]] = Path(name)
    if set(paths) != {"dataset", "denoising", "ar"}:
        raise ValueError("score report lacks complete dataset/objective provenance")
    return score, paths


def _lineage(prediction_path: Path) -> tuple[list[str], Record, bool, Record]:
    prediction = read_sealed(prediction_path)
    check_snapshot(prediction["inputs"])
    checkpoint = read_sealed(Path(prediction["checkpoint_path"]))
    check_snapshot(checkpoint["binding"]["inputs"])
    path = Path(checkpoint["binding"]["config"]["dataset"])
    dataset = load_training_dataset(path)
    synthetic = any(row.get("synthetic", False) for row in dataset["examples"])
    groups = {
        "authors": sorted({row["author_id"] for row in dataset["examples"]}),
        "works": sorted({row["work_id"] for row in dataset["examples"]}),
    }
    lineage = [dataset["id"]]
    while "parent_path" in dataset:
        dataset = load_training_dataset(Path(dataset["parent_path"]))
        lineage.append(dataset["id"])
    return lineage, prediction, synthetic, groups


def report_augmentation(config_path: Path, out: Path) -> Record:
    """Compare frozen human-only and augmented pairs on identical held-out tasks."""
    config = AblationConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    augmentation = load_augmentation(config.augmentation)
    human, human_paths = _score_inputs(config.human_scores)
    augmented, augmented_paths = _score_inputs(config.augmented_scores)
    shared = ("dataset_id", "task_set_id", "codec_id", "vocabulary_id")
    if any(human[field] != augmented[field] for field in shared):
        raise ValueError("ablation evaluation tasks/representation must remain fixed")
    matched = (
        "base_id",
        "initial_parameter_sha256",
        "parameter_count",
        "vocabulary_id",
        "budget_policy",
        "seed",
        "training_budget",
        "inference_budget",
    )
    if (
        human["matching_mismatches"]
        or augmented["matching_mismatches"]
        or any(human["matching"][key] != augmented["matching"][key] for key in matched)
    ):
        raise ValueError("ablation base model and resource policies differ")
    models: dict[str, list[Record]] = {"human_only": [], "augmented": []}
    for side, paths, expected in (
        ("human_only", human_paths, augmentation["parent_id"]),
        ("augmented", augmented_paths, augmentation["id"]),
    ):
        for objective in ("denoising", "ar"):
            lineage, prediction, synthetic, groups = _lineage(paths[objective])
            if expected not in lineage or (side == "human_only" and synthetic):
                raise ValueError(
                    "ablation checkpoint does not use the registered training population"
                )
            models[side].append(
                {
                    "objective": objective,
                    "prediction_id": prediction["id"],
                    "checkpoint_id": prediction["checkpoint_id"],
                    "training_lineage": lineage,
                    "source_groups": groups,
                    "work": prediction["work"],
                }
            )
    if any(
        models["human_only"][index]["source_groups"] != models["augmented"][index]["source_groups"]
        for index in range(len(models["human_only"]))
    ):
        raise ValueError("ablation source work/author subsets differ")
    # Recompute scores from their bound predictions; saved summary values are not authoritative.
    for label, saved, paths in (
        ("human", human, human_paths),
        ("augmented", augmented, augmented_paths),
    ):
        checked = score_pair(paths["dataset"], paths["denoising"], paths["ar"], out / label)
        if checked["id"] != saved["id"]:
            raise ValueError("ablation score differs from its original bound evidence")
    result = create_record(
        out / "ablation.json",
        {
            "kind": "reconstruction-augmentation-ablation",
            "augmentation_id": augmentation["id"],
            "human_score_id": human["id"],
            "augmented_score_id": augmented["id"],
            "models": models,
            "generation_costs": augmentation["generation_costs"],
            "source_groups": augmentation["source_groups"],
            "source_works": augmentation["source_works"],
            "mixture": augmentation["mixture"],
            "independent_works": augmentation["independent_works"],
            "summary": {"human_only": human["summary"], "augmented": augmented["summary"]},
            "test_only": augmentation["test_only"] or human["test_only"] or augmented["test_only"],
            "scientific_status": (
                "unestablished; separate ablation, no automatic gate authorization"
            ),
            "g3_authorized": False,
            "phase7_authorized": False,
            "inputs": snapshot(
                [
                    config_path,
                    _file(config.augmentation),
                    config.human_scores,
                    config.augmented_scores,
                ]
            ),
        },
    )
    report_files(out, "Human-only versus synthetic augmentation", result)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="augmentation_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("config", type=Path)
    create.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("path", type=Path)
    report = commands.add_parser("report")
    report.add_argument("config", type=Path)
    report.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        if args.augmentation_command == "report":
            record = report_augmentation(args.config, args.out)
        else:
            record = (
                create_augmentation(args.config, args.out)
                if args.augmentation_command == "create"
                else load_augmentation(args.path)
            )
        print(json.dumps({"id": record["id"], "mixture": record["mixture"]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde reconstruction-augmentation: {error}", file=sys.stderr)
        return 2
