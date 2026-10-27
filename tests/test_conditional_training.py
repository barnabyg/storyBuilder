import importlib.util
import json
import os
from pathlib import Path

import pytest

from nde.conditional.data import Conditions
from nde.conditional.training import candidate_cost, inspect_checkpoint, sample_candidate
from nde.corpus.records_io import create_record, identity, read_json, read_sealed
from tests.conditional_support import (
    tiny_conditional_dataset,
    trained_checkpoint,
    training_command,
    training_config,
)
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
    no_conditioning = {**resumed, "conditioning_modes": ["unconditional", "unconditional"]}
    unconditioned_checkpoint = tmp_path / "no-conditioning.json"
    create_record(unconditioned_checkpoint, no_conditioning)
    assert (
        training_command(
            ["sample", str(unconditioned_checkpoint), "--out", str(tmp_path / "unconditioned")]
        )
        == 2
    )
    no_unconditional = {**resumed, "conditioning_modes": ["conditional", "conditional"]}
    conditional_checkpoint = tmp_path / "conditional-only.json"
    create_record(conditional_checkpoint, no_unconditional)
    assert (
        training_command(
            [
                "sample",
                str(conditional_checkpoint),
                "--guidance-scale",
                "2",
                "--out",
                str(tmp_path / "untrained-guidance"),
            ]
        )
        == 2
    )
    (tmp_path / "data").rename(tmp_path / "unavailable-training-source")
    checkpoint = run / "checkpoints/step-00000002.json"
    assert training_command(["sample", str(checkpoint), "--out", str(tmp_path / "sample")]) == 0
    sampled = read_sealed(tmp_path / "sample/sample.json")
    assert sampled["work"]["tokens"] > 0
    assert sampled["status"] in {"invalid", "complete"}
    assert training_command(["sample", str(checkpoint), "--out", str(tmp_path / "sample")]) == 2
    assert read_sealed(tmp_path / "sample/sample.json")["id"] == sampled["id"]
    assert training_command(["resume", str(config), "--out", str(run)]) == 2


def test_training_refuses_capacity_configuration_changes_and_retains_resource_caps(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = tiny_conditional_dataset(tmp_path / "data", fake_llm)
    config = training_config(tmp_path, dataset, max_tokens=1)
    run = tmp_path / "run"
    assert training_command(["train", str(config), "--out", str(run)]) == 0
    assert read_json(run / "report.json")["status"] == "resource-stop"
    assert (
        training_command(
            ["sample", str(run / "checkpoints/step-00000000.json"), "--out", str(tmp_path / "zero")]
        )
        == 2
    )
    assert training_command(["resume", str(config), "--out", str(run)]) == 0
    assert read_json(run / "report.json")["tokens"] == 0
    config.write_text(json.dumps({**json.loads(config.read_text()), "max_tokens": 100000}))
    assert training_command(["resume", str(config), "--out", str(run)]) == 2


def test_self_contained_checkpoint_rejects_unsupported_architecture(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = tiny_conditional_dataset(tmp_path / "data", fake_llm)
    config = training_config(tmp_path, dataset)
    assert training_command(["train", str(config), "--out", str(tmp_path / "run")]) == 0
    saved = read_sealed(tmp_path / "run/checkpoints/step-00000002.json")
    saved["binding"]["architecture"] = "unsupported-backbone"
    saved["id"] = identity(saved)
    checkpoint = tmp_path / "unsupported.json"
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="architecture"):
        inspect_checkpoint(checkpoint)


@pytest.mark.parametrize("scene_count", [2, 3], ids=["two-scenes", "three-scenes"])
def test_cpu_conditional_denoising_learns_valid_layout_and_accounts_guidance_passes(
    tmp_path: Path, fake_llm: FakeOpenAI, scene_count: int
) -> None:
    checkpoint = trained_checkpoint(tmp_path, fake_llm, scene_count=scene_count)
    state = inspect_checkpoint(checkpoint)
    assert set(state["conditioning_modes"]) == {"conditional", "unconditional"}
    assert state["parameter_sha256"] != state["binding"]["initial_parameter_sha256"]
    (tmp_path / "data").rename(tmp_path / "unavailable-layout-targets")
    conditions = Conditions(genre="parable", scene_range=(1, 4))
    candidate = sample_candidate(checkpoint, conditions, seed=41)
    assert candidate["status"] == "complete", candidate["diagnostics"]
    assert len(candidate["view"]["scenes"]) == scene_count
    assert candidate["work"]["passes"] == 1
    same = sample_candidate(checkpoint, conditions, seed=41)
    assert candidate["generated_tokens"] == same["generated_tokens"]
    guided = sample_candidate(checkpoint, conditions, seed=41, guidance_scale=2)
    assert guided["work"]["passes"] == 2
    assert guided["work"]["tokens"] == candidate["work"]["tokens"] * 2
    assert (
        guided["pass_logit_sha256"]["conditional"] != guided["pass_logit_sha256"]["unconditional"]
    )
    assert candidate_cost(state, conditions, guidance_scale=2) == (
        guided["work"]["tokens"],
        guided["work"]["attention_cells"],
    )
    dropped = sample_candidate(checkpoint, conditions, guidance_scale=0)
    assert set(dropped["pass_logit_sha256"]) == {"unconditional"}
    with pytest.raises(ValueError, match="guidance scale"):
        sample_candidate(checkpoint, conditions, guidance_scale=float("inf"))
    expired = sample_candidate(checkpoint, conditions, deadline=0.00001)
    assert expired["status"] == "resource-stop"
    assert expired["work"]["passes"] == 0
