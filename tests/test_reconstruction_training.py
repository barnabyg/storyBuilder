import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from nde.corpus.records_io import read_sealed
from nde.reconstruction.dataset import load_dataset
from nde.reconstruction.tasks import model_input
from nde.reconstruction.training import (
    initialize_model,
    inspect_attention,
    inspect_checkpoint,
    inspect_model,
    matched_configs,
    reconstruct,
    register_local_model,
    train,
)
from tests.fake_openai import FakeOpenAI
from tests.reconstruction_support import training_dataset

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
        reason="optional torch runtime; CPU ML lane requires these tests",
    ),
]


def test_local_base_model_has_immutable_weights_and_deterministic_identity(tmp_path: Path) -> None:
    first = initialize_model(tmp_path / "first", width=8, max_positions=512, seed=19)
    second = initialize_model(tmp_path / "second", width=8, max_positions=512, seed=19)
    assert first["parameter_sha256"] == second["parameter_sha256"]
    assert first["parameter_count"] > 1000
    assert inspect_model(tmp_path / "first")["id"] == first["id"]
    (tmp_path / "first" / "weights.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        inspect_model(tmp_path / "first")


def test_cpu_training_updates_parameters_and_resume_restores_optimizer_progress(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096, seed=19)
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(tmp_path / "base"),
                "max_steps": 2,
                "max_output_tokens": 128,
                "test_only": True,
            }
        )
    )
    first = train(config, tmp_path / "run", max_steps_this_call=1)
    assert first["status"] == "partial"
    assert first["step"] == 1
    assert first["parameter_sha256"] != first["binding"]["initial_parameter_sha256"]
    resumed = train(config, tmp_path / "run", resume=True)
    uninterrupted = train(config, tmp_path / "continuous")
    assert resumed["step"] == 2
    assert resumed["parameter_sha256"] == uninterrupted["parameter_sha256"]
    assert resumed["losses"] == uninterrupted["losses"]
    checkpoint = tmp_path / "run/checkpoints/step-00000002.json"
    state = inspect_checkpoint(checkpoint)
    assert state["sampler_position"] == 2
    assert state["optimizer_state"]["state"]
    result = reconstruct(checkpoint, dataset, tmp_path / "predictions")
    assert len(result["predictions"]) == 2
    assert result["work"]["inference_tokens"] > 0
    config.write_text(config.read_text().replace('"max_steps": 2', '"max_steps": 3'))
    with pytest.raises(ValueError, match="changed"):
        train(config, tmp_path / "run", resume=True)
    config.write_text(config.read_text().replace('"max_steps": 3', '"max_steps": 2'))
    exported = json.loads(Path(load_dataset(dataset)["export_path"]).read_text())
    source_annotation = Path(exported["records"][0]["annotation_path"])
    source_annotation.write_text("{}")
    with pytest.raises(ValueError, match="bound evidence"):
        inspect_checkpoint(checkpoint)


def test_pretrained_import_binds_original_local_weights_and_provenance(tmp_path: Path) -> None:
    initialize_model(tmp_path / "source", width=8, max_positions=128)
    provenance = tmp_path / "pretraining.txt"
    provenance.write_text("Offline native backbone pretraining fixture; no scientific claims.")
    config = tmp_path / "import.json"
    config.write_text(
        json.dumps(
            {
                "width": 8,
                "max_positions": 128,
                "model_id": "fixture-native",
                "pretraining_description": "fixture pretraining mechanics",
                "artifacts": [str(provenance)],
                "test_only": True,
            }
        )
    )
    imported = register_local_model(tmp_path / "source/weights.json", config, tmp_path / "imported")
    assert imported["initialization"] == "registered-local-pretrained"
    assert imported["test_only"] is True
    assert inspect_model(tmp_path / "imported")["id"] == imported["id"]
    provenance.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        inspect_model(tmp_path / "imported")


def test_cpu_resource_stop_preserves_consumed_budget_on_resume(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    saved = load_dataset(dataset)
    length = (
        max(len(model_input(task)) for example in saved["examples"] for task in example["tasks"])
        + 128
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096)
    config = tmp_path / "limited.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(tmp_path / "base"),
                "max_steps": 5,
                "max_tokens": length,
                "max_output_tokens": 128,
                "test_only": True,
            }
        )
    )
    stopped = train(config, tmp_path / "run")
    assert stopped["status"] == "resource-stop"
    assert stopped["step"] == 1
    resumed = train(config, tmp_path / "run", resume=True)
    assert resumed["step"] == 1
    assert resumed["tokens"] == stopped["tokens"]
    assert inspect_checkpoint(tmp_path / "run/checkpoints/step-00000001.json")["step"] == 1


def test_checkpoint_publication_interruption_recovers_published_optimizer_work(
    tmp_path: Path, fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096)
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(tmp_path / "base"),
                "max_steps": 2,
                "max_output_tokens": 128,
                "test_only": True,
            }
        )
    )
    replace = Path.replace
    writes = 0

    def interrupted_replace(path: Path, target: str | Path) -> Path:
        nonlocal writes
        if Path(target).name == "latest.json":
            writes += 1
            if writes == 2:
                raise OSError("simulated interrupted pointer update")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted"):
        train(config, tmp_path / "run")
    monkeypatch.setattr(Path, "replace", replace)
    assert inspect_checkpoint(tmp_path / "run/checkpoints/step-00000001.json")["step"] == 1
    resumed = train(config, tmp_path / "run", resume=True)
    continuous = train(config, tmp_path / "continuous")
    assert resumed["parameter_sha256"] == continuous["parameter_sha256"]
    assert resumed["tokens"] == continuous["tokens"]


def test_cpu_ar_uses_matched_initialization_resumes_and_reconstructs_same_tasks(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096)
    common = {
        "dataset": str(dataset),
        "base_model": str(tmp_path / "base"),
        "max_steps": 2,
        "max_output_tokens": 128,
        "test_only": True,
    }
    denoiser, ar = tmp_path / "denoiser.json", tmp_path / "ar.json"
    denoiser.write_text(json.dumps(common))
    ar.write_text(json.dumps({**common, "objective": "ar"}))
    assert matched_configs(denoiser, ar)["matched"] is True
    first = train(ar, tmp_path / "ar-run", max_steps_this_call=1)
    resumed = train(ar, tmp_path / "ar-run", resume=True)
    assert first["step"] == 1
    assert resumed["step"] == 2
    assert resumed["parameter_sha256"] != resumed["binding"]["initial_parameter_sha256"]
    predictions = reconstruct(
        tmp_path / "ar-run/checkpoints/step-00000002.json", dataset, tmp_path / "ar-predictions"
    )
    assert predictions["objective"] == "ar"
    assert len(predictions["predictions"]) == 2
    assert predictions["work"]["inference_tokens"] > 0
    ar.write_text(json.dumps({**common, "objective": "ar", "seed": 99}))
    with pytest.raises(ValueError, match="unmatched"):
        matched_configs(denoiser, ar)


def test_native_ar_attention_cannot_read_future_infill_tokens(tmp_path: Path) -> None:
    initialize_model(tmp_path / "base", width=8, max_positions=128)
    left = inspect_attention(tmp_path / "base", [10, 11], [2, 12, 13], objective="ar")
    right = inspect_attention(tmp_path / "base", [10, 11], [2, 12, 99], objective="ar")
    assert left[0] == right[0]
    assert left[1] == right[1]
    bidirectional = inspect_attention(
        tmp_path / "base", [10, 11], [2, 12, 13], objective="denoising"
    )
    changed = inspect_attention(tmp_path / "base", [10, 11], [2, 12, 99], objective="denoising")
    assert bidirectional[0] != changed[0]


def test_interrupted_optimizer_work_remains_charged_to_the_resumed_budget(
    tmp_path: Path, fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096)
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(tmp_path / "base"),
                "max_steps": 2,
                "max_output_tokens": 128,
                "test_only": True,
            }
        )
    )
    hardlink = Path.hardlink_to

    def interrupted_link(path: Path, target: str | Path) -> None:
        if path.name == "step-00000001.json":
            raise OSError("simulated interruption before checkpoint publication")
        hardlink(path, target)

    monkeypatch.setattr(Path, "hardlink_to", interrupted_link)
    with pytest.raises(OSError, match="interruption"):
        train(config, tmp_path / "run")
    monkeypatch.setattr(Path, "hardlink_to", hardlink)
    resumed = train(config, tmp_path / "run", resume=True)
    continuous = train(config, tmp_path / "continuous")
    assert resumed["parameter_sha256"] == continuous["parameter_sha256"]
    assert resumed["tokens"] > continuous["tokens"]
    assert resumed["reserved_failed_tokens"] == resumed["tokens"] - continuous["tokens"]


def test_cpu_inference_deadline_and_interrupted_allowance_are_preserved(
    tmp_path: Path, fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = training_dataset(
        tmp_path / "data", fake_llm, schedules=[{"kind": "fields", "channel": "goal", "rate": 0.1}]
    )
    initialize_model(tmp_path / "base", width=8, max_positions=4096)
    config = tmp_path / "timed-ar.json"
    config.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "base_model": str(tmp_path / "base"),
                "objective": "ar",
                "max_steps": 1,
                "max_output_tokens": 128,
                "inference_max_seconds": 0.5,
                "test_only": True,
            }
        )
    )
    train(config, tmp_path / "run")
    checkpoint = tmp_path / "run/checkpoints/step-00000001.json"
    ticks = 0.0

    def monotonic() -> float:
        nonlocal ticks
        ticks += 0.02
        return ticks

    monkeypatch.setattr("time.monotonic", monotonic)
    stopped = reconstruct(checkpoint, dataset, tmp_path / "stopped")
    assert stopped["work"]["inference_tokens"] > 0
    assert all(row["status"] == "resource-stop" for row in stopped["predictions"])
    assert stopped["work"]["inference_elapsed_seconds"] >= 0.5
    original_open = Path.open

    def interrupted_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "predictions.json":
            raise OSError("interrupted inference publication")
        return cast(Any, original_open)(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", interrupted_open)
    with pytest.raises(OSError, match="interrupted inference"):
        reconstruct(checkpoint, dataset, tmp_path / "interrupted")
    monkeypatch.setattr(Path, "open", original_open)
    journal = read_sealed(tmp_path / "interrupted/inference-time.json")
    resumed = reconstruct(checkpoint, dataset, tmp_path / "interrupted")
    assert resumed["work"]["inference_tokens"] == journal["tokens"]
    assert resumed["work"]["inference_interrupted_allowance_retained"] is True
    assert all(row["status"] == "resource-stop" for row in resumed["predictions"])
