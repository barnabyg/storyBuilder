"""Public, approved test-only Phase 6 datasets; no synthetic scientific approval."""

import json
from pathlib import Path
from typing import Any

from nde.cli import main
from nde.corpus.acquisition import load_work
from nde.reconstruction.codec import create_encoding
from nde.reconstruction.dataset import authorize_evaluation, prepare_dataset
from tests.corpus_support import corpus_inputs, label_packet, queue_annotation, review_config
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export
from tests.pilot_support import pilot_config
from tests.test_corpus_acquisition import acquire


def training_dataset(directory: Path, fake: FakeOpenAI, **options: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    exported = approved_export(directory, fake)
    encoded = directory / "encoded"
    create_encoding(exported, encoded)
    config = directory / "prepare.json"
    config.write_text(json.dumps({"encoding": str(encoded), **options}), encoding="utf-8")
    output = directory / "prepared"
    prepare_dataset(config, output)
    return output / "dataset.json"


def paired_datasets(directory: Path, fake: FakeOpenAI) -> tuple[Path, Path]:
    """Training plus an explicitly authorized, disjoint held-out author."""
    directory.mkdir(parents=True, exist_ok=True)
    inputs = corpus_inputs(directory, fake)
    held_directory = directory / "held-source"
    held_directory.mkdir()
    held = acquire(
        held_directory,
        "BEGIN LAMP\nBea opened a door.\nEND LAMP",
        work_id="door",
        author_id="bea",
        author="Bea Writer",
        source_id="door-edition",
    )
    config = directory / "paired-cohort-config.json"
    config.write_text(
        json.dumps(
            {
                "id": "paired-fixture",
                "works": [str(inputs["work"]), str(held)],
                "author_splits": {"ann": "train", "bea": "validation"},
            }
        )
    )
    cohort = directory / "paired-cohort"
    assert main(["corpus", "split", "freeze", str(config), "--out", str(cohort)]) == 0
    inputs["cohort"] = cohort / "cohort.json"
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
    protocol = directory / "protocol.json"
    assert main(["corpus-g2", "register", str(config), "--out", str(protocol)]) == 0
    label_packet(directory, review)
    gate, exported = directory / "gate", directory / "exported"
    assert main(["corpus-g2", "report", str(protocol), "--out", str(gate)]) == 0
    assert main(["corpus-export", "create", str(gate), "--out", str(exported), "--test-only"]) == 0
    encoded = directory / "encoded"
    create_encoding(exported, encoded)
    config = directory / "prepare-training.json"
    config.write_text(json.dumps({"encoding": str(encoded)}))
    train = directory / "train"
    prepare_dataset(config, train)
    work, text = load_work(held)
    annotation = directory / "held-plan.json"
    annotation.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "id": "held-plan",
                "title": "Door",
                "tier": "L",
                "sources": [
                    {
                        "id": work.source_id,
                        "title": work.title,
                        "text": text,
                        "sha256": work.text_sha256,
                        "provenance": work.source,
                    }
                ],
                "scenes": [
                    {
                        "id": "door-scene",
                        "title": "An opening",
                        "evidence": [
                            {
                                "source_id": work.source_id,
                                "source_sha256": work.text_sha256,
                                "start": 0,
                                "end": len(text),
                            }
                        ],
                    }
                ],
            }
        )
    )
    config = directory / "authorize.json"
    config.write_text(
        json.dumps(
            {
                "export": str(exported),
                "cohort": str(cohort),
                "split": "validation",
                "authorized_by": "synthetic software fixture",
                "test_only": True,
                "annotations": [{"work_id": "door", "family": "fixture", "path": str(annotation)}],
            }
        )
    )
    authorization = directory / "authorization"
    authorize_evaluation(config, authorization)
    config = directory / "prepare-held.json"
    config.write_text(json.dumps({"evaluation": str(authorization)}))
    evaluation = directory / "evaluation"
    prepare_dataset(config, evaluation)
    return train / "dataset.json", evaluation / "dataset.json"
