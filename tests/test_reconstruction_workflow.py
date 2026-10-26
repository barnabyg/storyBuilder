"""CLI acceptance path from protected data through an actual CPU experiment and G3."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from nde.cli import main
from nde.corpus.records_io import read_sealed
from tests.fake_openai import FakeOpenAI
from tests.reconstruction_support import paired_datasets


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
    reason="optional CPU ML runtime; required in the CPU ML lane",
)
def test_cli_cpu_experiment_and_registered_gate_remain_scientifically_pending(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train, evaluation = paired_datasets(tmp_path / "data", fake_llm, split="test")
    base = tmp_path / "base"
    assert (
        main(
            [
                "reconstruction-train",
                "initialize",
                "--out",
                str(base),
                "--width",
                "8",
                "--max-positions",
                "8192",
            ]
        )
        == 0
    )
    config = tmp_path / "experiment-config.json"
    config.write_text(
        json.dumps(
            {
                "base_model": str(base),
                "train_datasets": [str(train)],
                "evaluation_dataset": str(evaluation),
                "seeds": [7],
                "subsets": [100],
                "max_jobs": 2,
                "test_only": True,
                "training": {
                    "max_steps": 1,
                    "max_output_tokens": 2048,
                    "inference_max_tokens": 5000,
                },
            }
        )
    )
    experiment = tmp_path / "experiment"
    assert (
        main(["reconstruction-experiment", "register", str(config), "--out", str(experiment)]) == 0
    )
    matrix = read_sealed(experiment / "experiment.json")
    reference = tmp_path / "reference.json"
    reference.write_text('{"planner":"C","test_only":true}')
    overlap = tmp_path / "overlap"
    config = tmp_path / "g3-config.json"
    config.write_text(
        json.dumps(
            {
                "training": str(train),
                "evaluation": str(evaluation),
                "base_model": str(base),
                "experiment": str(experiment / "experiment.json"),
                "score_reports": matrix["expected_scores"],
                "overlap_report": str(overlap / "overlap.json"),
                "reference": str(reference),
                "test_only": True,
            }
        )
    )
    protocol = tmp_path / "g3-protocol.json"
    assert main(["reconstruction-g3", "readiness", str(config)]) == 0
    assert main(["reconstruction-g3", "register", str(config), "--out", str(protocol)]) == 0
    assert main(["reconstruction-experiment", "run", str(experiment)]) == 0
    assert main(["reconstruction-overlap", str(train), str(evaluation), "--out", str(overlap)]) == 0
    decision = tmp_path / "decision"
    assert main(["reconstruction-g3", "report", str(protocol), "--out", str(decision)]) == 0
    report = read_sealed(decision / "decision.json")
    assert report["outcome"] == "pending"
    assert not any("inconsistent score" in note for note in report["pending"])
    assert not any("differs from frozen paired job" in note for note in report["pending"])
    assert not (tmp_path / "phase7").exists()
