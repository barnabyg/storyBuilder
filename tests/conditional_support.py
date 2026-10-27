"""Approved synthetic data and public command adapters for CPU acceptance."""

import argparse
import json
from pathlib import Path
from typing import Any

from nde.conditional.data import prepare_dataset
from nde.conditional.training import configure, run
from nde.reconstruction.training import initialize_model
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export


def training_command(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    configure(parser)
    return run(parser.parse_args(arguments))


def tiny_conditional_dataset(directory: Path, fake: FakeOpenAI) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exported = approved_export(directory, fake)
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


def trained_checkpoint(directory: Path, fake: FakeOpenAI, **options: Any) -> Path:
    dataset = tiny_conditional_dataset(directory / "data", fake)
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
