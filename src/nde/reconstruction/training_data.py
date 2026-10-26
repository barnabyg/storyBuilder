"""Strict derived optimizer views retain the approved parent split and complete groups."""

import importlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from nde.corpus.records_io import check_snapshot, create_record, identity, read_sealed, snapshot
from nde.reconstruction.dataset import load_dataset


def dataset_file(path: Path) -> Path:
    return path / "dataset.json" if path.is_dir() else path


def nested_groups(examples: list[dict[str, Any]], percent: int, *, seed: int) -> list[str]:
    if percent not in {25, 50, 100}:
        raise ValueError("registered subset must be 25, 50 or 100 percent")
    authors = sorted(
        {str(row["author_id"]) for row in examples},
        key=lambda author: identity({"seed": seed, "author": author}),
    )
    return authors[: math.ceil(len(authors) * percent / 100)]


def create_subset(parent: Path, out: Path, percent: int, *, seed: int) -> dict[str, Any]:
    parent = dataset_file(parent)
    dataset = load_training_dataset(parent)
    groups = nested_groups(dataset["examples"], percent, seed=seed)
    examples = [example for example in dataset["examples"] if example["author_id"] in groups]
    return create_record(
        out / "dataset.json",
        {
            "kind": "reconstruction-subset",
            "role": "optimizer",
            "split": "train",
            "parent_path": str(parent.resolve()),
            "parent_id": dataset["id"],
            "percent": percent,
            "seed": seed,
            "groups": groups,
            "examples": examples,
            "test_only": dataset["test_only"],
            "codec_id": dataset["codec_id"],
            "vocabulary_id": dataset["vocabulary_id"],
            "granularity": dataset["granularity"],
            "independent_works": len({row["work_id"] for row in examples}),
            "independent_authors": len(groups),
            "inputs": snapshot([parent]),
        },
    )


def load_training_dataset(path: Path) -> dict[str, Any]:
    path = dataset_file(path)
    record = read_sealed(path)
    if record.get("kind") == "reconstruction-dataset":
        return load_dataset(path, optimizer=True)
    if record.get("kind") == "reconstruction-augmentation":
        loader = cast(
            Callable[[Path], dict[str, Any]],
            importlib.import_module("nde.reconstruction.augmentation").load_augmentation,
        )
        return loader(path)
    if record.get("kind") != "reconstruction-subset":
        raise ValueError("unsupported derived optimizer dataset")
    check_snapshot(record["inputs"])
    parent = load_training_dataset(Path(record["parent_path"]))
    groups = nested_groups(parent["examples"], record["percent"], seed=record["seed"])
    examples = [example for example in parent["examples"] if example["author_id"] in groups]
    if (
        record["parent_id"] != parent["id"]
        or record["groups"] != groups
        or record["examples"] != examples
        or record["role"] != "optimizer"
        or record["split"] != "train"
        or record["test_only"] != parent["test_only"]
        or record["codec_id"] != parent["codec_id"]
        or record["vocabulary_id"] != parent["vocabulary_id"]
    ):
        raise ValueError("derived subset escaped its registered parent/group policy")
    return record
