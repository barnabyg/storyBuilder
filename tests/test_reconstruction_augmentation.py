import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest

from nde.reconstruction import augmentation
from nde.reconstruction.dataset import load_dataset, prepare_dataset
from nde.reconstruction.scoring import score_pair
from nde.reconstruction.training import initialize_model, reconstruct, train
from tests.fake_openai import FakeOpenAI
from tests.reconstruction_support import paired_datasets


def command(*arguments: str) -> int:
    parser = argparse.ArgumentParser()
    augmentation.configure(parser)
    return augmentation.run(parser.parse_args(arguments))


def test_registered_augmentation_preserves_human_population_and_train_only_provenance(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train, evaluation = paired_datasets(tmp_path / "data", fake_llm)
    original = train.read_bytes()
    parent = load_dataset(train)
    config = tmp_path / "augmentation.json"
    config.write_text(json.dumps({"dataset": str(train), "synthetic_per_human": 1, "seed": 13}))
    out = tmp_path / "augmented"
    assert command("create", str(config), "--out", str(out)) == 0
    assert command("inspect", str(out)) == 0
    saved = augmentation.load_augmentation(out)
    assert saved["examples"][:2] == parent["examples"]
    assert len(saved["examples"]) == 4
    assert saved["independent_works"] == 1
    assert saved["independent_authors"] == 1
    assert saved["mixture"]["synthetic_fraction"] == 0.5
    generated = saved["examples"][2:]
    assert all(row["synthetic"] is True and row["author_id"] == "ann" for row in generated)
    assert all(
        row["record"]["view"] != parent["examples"][index]["record"]["view"]
        for index, row in enumerate(generated)
    )
    assert all(
        row["record"]["target_policy"] == parent["examples"][index]["record"]["target_policy"]
        for index, row in enumerate(generated)
    )
    assert all(
        row["record"]["view"]["states"][0]["before"]
        == parent["examples"][index]["record"]["view"]["states"][0]["before"]
        for index, row in enumerate(generated)
    )
    assert "Bea" not in json.dumps(saved["generation"])
    assert "Ann saw the lamp" not in json.dumps(saved["generation"])
    assert saved["generation_costs"]["provider_calls"] == 0
    assert saved["generation_costs"]["output_tokens"] > 0
    assert train.read_bytes() == original
    for index, overrides in enumerate(
        (
            {"max_generated_examples": 0},
            {"max_generation_tokens": 1},
            {"source_example_ids": [load_dataset(evaluation)["examples"][0]["id"]]},
            {"selection_basis": "validation", "selection_validation": str(train)},
        )
    ):
        rejected = tmp_path / f"rejected-config-{index}.json"
        rejected.write_text(json.dumps({"dataset": str(train), **overrides}))
        with pytest.raises(ValueError, match=r"resource limit|eligible training|test labels"):
            augmentation.create_augmentation(rejected, tmp_path / f"rejected-{index}")
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "dataset": str(train),
                "seed": 13,
                "selection_basis": "validation",
                "selection_validation": str(evaluation),
            }
        )
    )
    validation = augmentation.create_augmentation(selected, tmp_path / "validation-selected")
    assert validation["synthetic_example_ids"] == saved["synthetic_example_ids"]
    zero = tmp_path / "zero.json"
    zero.write_text(json.dumps({"dataset": str(train), "synthetic_per_human": 0}))
    unchanged = augmentation.create_augmentation(zero, tmp_path / "zero")
    assert unchanged["examples"] == parent["examples"]
    assert unchanged["generation_costs"]["output_tokens"] == 0
    config.write_text(json.dumps({"dataset": str(evaluation)}))
    assert command("inspect", str(out)) == 2
    with pytest.raises(ValueError, match="evaluator-only"):
        augmentation.create_augmentation(config, tmp_path / "rejected")


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
    reason="optional CPU ML runtime; required in the CPU lane",
)
def test_real_cpu_pairs_report_human_only_and_augmented_runs_separately(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train_path, evaluation = paired_datasets(tmp_path / "data", fake_llm)
    # Keep one eligible goal field hidden so the CPU fixture has a short actual target.
    preparation = load_dataset(train_path)["config"]
    preparation["schedules"] = [{"kind": "fields", "channel": "goal", "rate": 0.1}]
    config = tmp_path / "prepare.json"
    config.write_text(json.dumps(preparation))
    human_dir = tmp_path / "human"
    prepare_dataset(config, human_dir)
    human = human_dir / "dataset.json"
    config = tmp_path / "augmentation.json"
    config.write_text(json.dumps({"dataset": str(human)}))
    aug_dir = tmp_path / "augmented"
    augmentation.create_augmentation(config, aug_dir)
    augmented = aug_dir / "dataset.json"
    base = tmp_path / "base"
    initialize_model(base, width=4, max_positions=8192, seed=19)

    def run_pair(dataset: Path, label: str) -> Path:
        predictions: list[Path] = []
        for objective in ("denoising", "ar"):
            directory = tmp_path / f"{label}-{objective}"
            directory.mkdir()
            configuration = directory / "config.json"
            configuration.write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "base_model": str(base),
                        "objective": objective,
                        "max_steps": 1,
                        "max_output_tokens": 128,
                        "test_only": True,
                    }
                )
            )
            trained = train(configuration, directory / "run")
            assert trained["step"] == 1
            assert trained["parameter_sha256"] != trained["binding"]["initial_parameter_sha256"]
            checkpoint = directory / "run/checkpoints/step-00000001.json"
            reconstruct(checkpoint, evaluation, directory / "predictions")
            predictions.append(directory / "predictions/predictions.json")
        score_pair(evaluation, predictions[0], predictions[1], tmp_path / f"{label}-scores")
        return tmp_path / f"{label}-scores/scores.json"

    human_scores = run_pair(human, "human")
    augmented_scores = run_pair(augmented, "augmented")
    report_config = tmp_path / "ablation.json"
    report_config.write_text(
        json.dumps(
            {
                "augmentation": str(aug_dir),
                "human_scores": str(human_scores),
                "augmented_scores": str(augmented_scores),
            }
        )
    )
    report = augmentation.report_augmentation(report_config, tmp_path / "ablation")
    assert len(report["models"]["human_only"]) == 2
    assert len(report["models"]["augmented"]) == 2
    assert report["g3_authorized"] is False
    assert report["phase7_authorized"] is False
    assert report["summary"]["human_only"]["candidate"]["expected"] == 1
    report_config.write_text(
        json.dumps(
            {
                "augmentation": str(aug_dir),
                "human_scores": str(augmented_scores),
                "augmented_scores": str(augmented_scores),
            }
        )
    )
    with pytest.raises(ValueError, match="registered training population"):
        augmentation.report_augmentation(report_config, tmp_path / "invalid-control")
