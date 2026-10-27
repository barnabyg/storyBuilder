import importlib.util
import json
import os
from pathlib import Path

import pytest

from nde.corpus.records_io import read_json, read_sealed
from tests.conditional_support import tiny_conditional_dataset, training_command, training_config
from tests.fake_openai import FakeOpenAI

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
        reason="optional torch runtime; CPU ML lane requires conditional learning",
    ),
]


def test_actual_layout_training_resume_and_sampling_do_not_require_hidden_targets(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = tiny_conditional_dataset(tmp_path / "data", fake_llm)
    config = training_config(tmp_path, dataset)
    run = tmp_path / "run"
    assert training_command(["preflight", str(config)]) == 0
    assert (
        training_command(["train", str(config), "--out", str(run), "--steps-this-call", "1"]) == 0
    )
    first = read_sealed(run / "checkpoints/step-00000001.json")
    assert first["parameter_sha256"] != first["binding"]["initial_parameter_sha256"]
    assert training_command(["resume", str(config), "--out", str(run)]) == 0
    assert training_command(["train", str(config), "--out", str(tmp_path / "continuous")]) == 0
    resumed = read_sealed(run / "checkpoints/step-00000002.json")
    continuous = read_sealed(tmp_path / "continuous/checkpoints/step-00000002.json")
    assert resumed["parameter_sha256"] == continuous["parameter_sha256"]
    assert resumed["losses"] == continuous["losses"]
    assert resumed["optimizer_state"]["state"]
    assert resumed["sampler_position"] == 2
    (tmp_path / "data").rename(tmp_path / "unavailable-training-source")
    checkpoint = run / "checkpoints/step-00000002.json"
    assert training_command(["sample", str(checkpoint), "--out", str(tmp_path / "sample")]) == 0
    sampled = read_sealed(tmp_path / "sample/sample.json")
    assert sampled["work"]["tokens"] > 0
    assert sampled["status"] in {"invalid", "complete"}
    assert training_command(["resume", str(config), "--out", str(run)]) == 2


def test_training_refuses_capacity_configuration_changes_and_retains_resource_caps(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = tiny_conditional_dataset(tmp_path / "data", fake_llm)
    config = training_config(tmp_path, dataset, max_tokens=1)
    run = tmp_path / "run"
    assert training_command(["train", str(config), "--out", str(run)]) == 0
    assert read_json(run / "report.json")["status"] == "resource-stop"
    assert training_command(["resume", str(config), "--out", str(run)]) == 0
    assert read_json(run / "report.json")["tokens"] == 0
    config.write_text(json.dumps({**json.loads(config.read_text()), "max_tokens": 100000}))
    assert training_command(["resume", str(config), "--out", str(run)]) == 2
