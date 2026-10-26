"""Protected datasets are exercised through saved public command artifacts."""

import json
from pathlib import Path

import pytest

from nde.cli import main
from nde.corpus.records_io import identity
from nde.reconstruction.dataset import load_dataset
from tests.fake_openai import FakeOpenAI
from tests.g2_support import approved_export
from tests.reconstruction_support import paired_datasets


def test_training_examples_preserve_policy_parent_and_window_boundaries(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    export = approved_export(tmp_path, fake_llm)
    encoded = tmp_path / "encoded"
    assert main(["reconstruction-codec", "encode", str(export), "--out", str(encoded)]) == 0
    config = tmp_path / "prepare.json"
    config.write_text(json.dumps({"encoding": str(encoded), "granularity": "scene"}))
    out = tmp_path / "prepared"
    assert main(["reconstruction-data", "prepare", str(config), "--out", str(out)]) == 0
    dataset = json.loads((out / "dataset.json").read_text())
    assert dataset["role"] == "optimizer"
    assert dataset["split"] == "train"
    assert len(dataset["examples"]) == 4
    assert {row["work_id"] for row in dataset["examples"]} == {"lamp"}
    assert {row["scene_start"] for row in dataset["examples"]} == {0, 1}
    for example in dataset["examples"]:
        assert example["record"]["provenance"]["boundary_context"] == "out-of-view unknown"
        assert not any(
            slot["eligible"]
            for slot in example["record"]["slots"]
            if slot["channel"] in {"causality", "information"}
        )
        assert example["tasks"]
    assert main(["reconstruction-data", "inspect", str(out)]) == 0
    changed = json.loads((out / "dataset.json").read_text())
    changed["examples"][0]["tasks"] = []
    changed["examples"][0]["id"] = identity(changed["examples"][0])
    changed["id"] = identity(changed)
    (out / "dataset.json").write_text(json.dumps(changed))
    assert main(["reconstruction-data", "inspect", str(out)]) == 2
    (out / "dataset.json").write_text(json.dumps(dataset))
    annotation = Path(json.loads(export.read_text())["records"][0]["annotation_path"])
    annotation.write_text("{}")
    assert main(["reconstruction-data", "inspect", str(out)]) == 2


@pytest.mark.parametrize(("granularity", "window", "stride"), [("window", 0, 1), ("scene", 1, 0)])
def test_invalid_example_boundaries_are_rejected(
    tmp_path: Path, granularity: str, window: int, stride: int
) -> None:
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps(
            {
                "encoding": "missing",
                "granularity": granularity,
                "window_scenes": window,
                "stride": stride,
            }
        )
    )
    assert (
        main(["reconstruction-data", "prepare", str(config), "--out", str(tmp_path / "out")]) == 2
    )


def test_heldout_author_requires_authorization_and_never_enters_optimizer(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train, held = paired_datasets(tmp_path, fake_llm)
    assert load_dataset(train, optimizer=True)["independent_works"] == 1
    evaluation = load_dataset(held)
    assert evaluation["role"] == "evaluator-only"
    assert {row["author_id"] for row in evaluation["examples"]} == {"bea"}
    with pytest.raises(ValueError, match="evaluator-only"):
        load_dataset(held, optimizer=True)
    config = json.loads((tmp_path / "authorize.json").read_text())
    config["annotations"][0]["work_id"] = "lamp"
    bad = tmp_path / "bad-authorize.json"
    bad.write_text(json.dumps(config))
    assert (
        main(
            [
                "reconstruction-data",
                "authorize-evaluation",
                str(bad),
                "--out",
                str(tmp_path / "bad"),
            ]
        )
        == 2
    )
