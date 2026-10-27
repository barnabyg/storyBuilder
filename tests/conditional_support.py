"""Approved synthetic data and public command adapters for CPU acceptance."""

import argparse
import json
from pathlib import Path
from typing import Any

from nde.cli import main
from nde.conditional.data import prepare_dataset
from nde.conditional.training import configure, run
from nde.reconstruction.training import initialize_model
from tests.corpus_support import corpus_inputs, label_packet, queue_annotation, review_config
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export
from tests.pilot_support import pilot_config
from tests.test_corpus_acquisition import acquire


def training_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    configure(parser)
    return run(parser.parse_args(arguments))


def _three_scene_export(directory: Path, fake: FakeOpenAI) -> Path:
    """Acquire and approve a distinct source before freezing any target artifacts."""
    family_source = directory / "family-configuration"
    family_source.mkdir()
    inputs = corpus_inputs(family_source, fake)
    work = acquire(
        directory,
        "BEGIN LAMP\nAnn saw the lamp.\n\n***\n\nAnn lit the lamp.\n\n***\n\n"
        "Ann carried the lamp home.\nEND LAMP",
    )
    segments = directory / "segments"
    assert main(["corpus", "segment", str(work), "--out", str(segments)]) == 0
    config = directory / "three-scene-cohort.json"
    config.write_text(
        json.dumps(
            {"id": "three-scene-fixture", "works": [str(work)], "author_splits": {"ann": "train"}}
        )
    )
    cohort = directory / "cohort"
    assert main(["corpus", "split", "freeze", str(config), "--out", str(cohort)]) == 0
    inputs.update(work=work, segments=segments / "segmentation.json", cohort=cohort / "cohort.json")
    queue_annotation(inputs, fake)
    pilot = directory / "pilot"
    config = pilot_config(directory, inputs)
    assert main(["corpus-pilot", "run", str(config), "--out", str(pilot)]) == 0
    review = directory / "review"
    config = review_config(
        directory, inputs, pilot / "stories/lamp", pilot / "features/lamp/features.json"
    )
    assert main(["corpus-review", "freeze", str(config), "--out", str(review)]) == 0
    config = directory / "g2-config.json"
    config.write_text(
        json.dumps(
            {
                "pilot": str(pilot),
                "review": str(review),
                "min_stories": 1,
                "min_owner_per_channel": 1,
                "scaling_budget_usd": 1,
            }
        )
    )
    protocol = directory / "g2-protocol.json"
    assert main(["corpus-g2", "register", str(config), "--out", str(protocol)]) == 0
    label_packet(directory, review)
    gate, dataset = directory / "gate", directory / "dataset"
    assert main(["corpus-g2", "report", str(protocol), "--out", str(gate)]) == 0
    assert main(["corpus-export", "create", str(gate), "--out", str(dataset), "--test-only"]) == 0
    return dataset / "dataset.json"


def tiny_conditional_dataset(directory: Path, fake: FakeOpenAI, *, scene_count: int = 2) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exported = (
        _three_scene_export(directory, fake)
        if scene_count == 3
        else approved_export(directory, fake)
    )
    config = directory / "conditions.json"
    config.write_text(
        json.dumps(
            {
                "export": str(exported),
                "conditions": [
                    {
                        "work_id": "lamp",
                        "labels": {
                            "genre": {
                                "origin": "metadata",
                                "status": "known",
                                "value": "parable",
                                "uncertainty": 0,
                            }
                        },
                    }
                ],
            }
        )
    )
    prepare_dataset(config, directory / "conditional")
    return directory / "conditional/dataset.json"


def training_config(directory: Path, dataset: Path, **options: Any) -> Path:
    initialize_model(directory / "base", width=16, max_positions=4096, seed=7)
    config = directory / "training.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(directory / "base"),
                "capacity": {
                    "max_scenes": 4,
                    "max_characters": 4,
                    "max_records": 16,
                    "max_tokens": 2048,
                },
                "max_steps": 2,
                "test_only": True,
                **options,
            }
        )
    )
    return config


def trained_checkpoint(
    directory: Path, fake: FakeOpenAI, *, scene_count: int = 2, **options: Any
) -> Path:
    dataset = tiny_conditional_dataset(directory / "data", fake, scene_count=scene_count)
    config = training_config(
        directory,
        dataset,
        **{
            "max_steps": 100,
            "max_condition_tokens": 512,
            "learning_rate": 0.03,
            "max_tokens": 2000000,
            "max_attention_cells": 2000000000,
            **options,
        },
    )
    assert training_command(["train", str(config), "--out", str(directory / "trained")]) == 0
    settings = json.loads(config.read_text())
    return directory / "trained/checkpoints" / f"step-{settings['max_steps']:08d}.json"
