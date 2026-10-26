"""Registered review evidence is exercised through saved packets and CLI reports."""

import json
from pathlib import Path

import pytest

from nde.cli import main
from nde.corpus.extraction_records import ExtractionOptions
from nde.corpus.families import compare_families
from tests.corpus_support import corpus_inputs, label_packet, queue_annotation, review_config
from tests.fake_openai import FakeOpenAI


def test_review_rejects_a_packet_without_a_frozen_identity(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "packet.json").write_text(json.dumps({"version": "1", "samples": []}))
    assert main(["corpus-review", "report", str(packet)]) == 2


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_review_registration_rejects_invalid_thresholds(tmp_path: Path, rate: float) -> None:
    config = tmp_path / "review.json"
    config.write_text(json.dumps({"cohort": "missing", "reliable_agreement": rate}))
    assert main(["corpus-review", "freeze", str(config), "--out", str(tmp_path / "review")]) == 2


def test_review_freezes_sampling_records_labels_and_invalidates_stale_evidence(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    inputs = corpus_inputs(tmp_path, fake_llm)
    queue_annotation(inputs, fake_llm)
    families, features = tmp_path / "families", tmp_path / "features"
    assert (
        main(
            [
                "extract",
                "families",
                str(inputs["work"]),
                "--config",
                str(inputs["family_config"]),
                "--segments",
                str(inputs["segments"]),
                "--out",
                str(families),
                "--cache",
                str(tmp_path / "cache.sqlite"),
                "--ledger",
                str(tmp_path / "ledger.sqlite"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "corpus",
                "features",
                str(inputs["work"]),
                str(inputs["segments"]),
                "--out",
                str(features),
            ]
        )
        == 0
    )
    config = review_config(tmp_path, inputs, families, features / "features.json")
    packet = tmp_path / "review"
    assert main(["corpus-review", "freeze", str(config), "--out", str(packet)]) == 0
    first = json.loads((packet / "packet.json").read_text())
    second = tmp_path / "same-review"
    assert main(["corpus-review", "freeze", str(config), "--out", str(second)]) == 0
    assert json.loads((second / "packet.json").read_text())["samples"] == first["samples"]
    assert len(first["samples"]) == 5
    label_packet(tmp_path, packet)
    assert main(["corpus-review", "report", str(packet)]) == 0
    report = json.loads((packet / "report.json").read_text())
    assert report["channels"]["goal"]["classification"] == "reliable"
    assert report["channels"]["goal"]["agreement_rate"] == 1
    assert report["scientific_status"] == "pending"
    assert report["channels"]["goal"]["gold_works"] == 0
    (inputs["work"] / "text.txt").write_text("changed")
    assert main(["corpus-review", "report", str(packet)]) == 0
    stale = json.loads((packet / "report.json").read_text())
    assert stale["status"] == "pending"
    assert stale["channels"]["goal"]["classification"] == "pending"
    assert len(list((packet / "judgements").glob("*.json"))) == 5


def test_review_cannot_attribute_another_works_annotations_to_named_source(
    tmp_path: Path,
    fake_llm: FakeOpenAI,
) -> None:
    inputs = corpus_inputs(tmp_path, fake_llm)
    other = tmp_path / "other"
    other.mkdir()
    different = corpus_inputs(other, fake_llm, work_id="other")
    queue_annotation(different, fake_llm)
    families = tmp_path / "unrelated-families"
    assert (
        compare_families(
            different["family_config"],
            different["work"],
            different["segments"],
            ExtractionOptions(families, tmp_path / "cache.sqlite", tmp_path / "ledger.sqlite"),
        ).status
        == "complete"
    )
    features = tmp_path / "features"
    assert (
        main(
            [
                "corpus",
                "features",
                str(inputs["work"]),
                str(inputs["segments"]),
                "--out",
                str(features),
            ]
        )
        == 0
    )
    config = review_config(tmp_path, inputs, families, features / "features.json")
    assert main(["corpus-review", "freeze", str(config), "--out", str(tmp_path / "review")]) == 2
