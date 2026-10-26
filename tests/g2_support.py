"""Synthetic evidence exercises G2 software without claiming a scientific pass."""

import json
from pathlib import Path

from nde.cli import main
from tests.corpus_support import (
    corpus_inputs,
    label_packet,
    queue_annotation,
    review_config,
)
from tests.fake_openai import FakeOpenAI
from tests.pilot_support import pilot_config


def g2_packet(directory: Path, fake: FakeOpenAI) -> tuple[Path, Path, Path]:
    inputs = corpus_inputs(directory, fake)
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
    return pilot, review, protocol


def approved_export(directory: Path, fake: FakeOpenAI) -> Path:
    _, review, protocol = g2_packet(directory, fake)
    label_packet(directory, review)
    decision, dataset = directory / "gate", directory / "dataset"
    assert main(["corpus-g2", "report", str(protocol), "--out", str(decision)]) == 0
    assert (
        main(["corpus-export", "create", str(decision), "--out", str(dataset), "--test-only"]) == 0
    )
    return dataset / "dataset.json"
