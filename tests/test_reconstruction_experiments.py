import importlib.util
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from nde.reconstruction.experiments import (
    ExperimentConfig,
    register_experiment,
    run_experiment,
)
from nde.reconstruction.training import initialize_model
from nde.reconstruction.training_data import nested_groups
from tests.fake_openai import FakeOpenAI
from tests.reconstruction_support import paired_datasets


def test_experiment_registration_rejects_unbounded_or_non_nested_sweeps() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "base_model": "base",
                "train_datasets": ["train"],
                "evaluation_dataset": "eval",
                "max_jobs": 0,
            }
        )


def test_nested_subsets_keep_all_work_families_under_their_author() -> None:
    examples = [
        {"author_id": f"author-{i}", "work_id": f"work-{i}", "family": family}
        for i in range(8)
        for family in ("first", "second")
    ]
    quarter = nested_groups(examples, 25, seed=7)
    half = nested_groups(examples, 50, seed=7)
    whole = nested_groups(examples, 100, seed=7)
    assert len(quarter) == 2
    assert len(half) == 4
    assert len(whole) == 8
    assert set(quarter) <= set(half) <= set(whole)
    assert quarter == nested_groups(list(reversed(examples)), 25, seed=7)


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
    reason="optional CPU ML runtime",
)
def test_cpu_sweep_resumes_frozen_jobs_and_scores_identical_heldout_tasks(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    train_path, evaluation = paired_datasets(tmp_path / "data", fake_llm)
    initialize_model(tmp_path / "base", width=8, max_positions=8192)
    config = tmp_path / "sweep.json"
    config.write_text(
        json.dumps(
            {
                "base_model": str(tmp_path / "base"),
                "train_datasets": [str(train_path)],
                "evaluation_dataset": str(evaluation),
                "subsets": [25, 100],
                "seeds": [7],
                "max_jobs": 4,
                "test_only": True,
                "training": {
                    "max_steps": 1,
                    "max_output_tokens": 2048,
                    "inference_max_tokens": 5000,
                },
            }
        )
    )
    protocol = register_experiment(config, tmp_path / "sweep")
    assert len(protocol["jobs"]) == 4
    partial = run_experiment(tmp_path / "sweep", max_jobs_this_call=1)
    assert partial["complete_jobs"] == 1
    complete = run_experiment(tmp_path / "sweep")
    assert complete["complete_jobs"] == 4
    assert len(complete["score_reports"]) == 2
    again = run_experiment(tmp_path / "sweep")
    assert again == complete
    assert all(
        row["evaluation_task_set_id"] == protocol["evaluation_task_set_id"]
        for row in complete["jobs"]
    )
    assert all(Path(path).is_file() for path in complete["score_reports"])


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
    reason="optional CPU ML runtime",
)
def test_failed_sweep_jobs_keep_their_resource_reservation_after_training(
    tmp_path: Path, fake_llm: FakeOpenAI
) -> None:
    training, evaluation = paired_datasets(tmp_path / "data", fake_llm)
    initialize_model(tmp_path / "base", width=8, max_positions=8192)
    config = tmp_path / "bounded.json"
    config.write_text(
        json.dumps(
            {
                "base_model": str(tmp_path / "base"),
                "train_datasets": [str(training)],
                "evaluation_dataset": str(evaluation),
                "subsets": [100],
                "max_jobs": 2,
                "max_total_tokens": 15000,
                "test_only": True,
                "training": {
                    "max_steps": 1,
                    "max_tokens": 10000,
                    "max_output_tokens": 2048,
                    "inference_max_tokens": 5000,
                },
            }
        )
    )
    protocol = register_experiment(config, tmp_path / "sweep")
    first = Path(protocol["jobs"][0]["config_path"]).parent
    (first / "predictions").write_text("fixture blocks predictions directory after training")
    result = run_experiment(tmp_path / "sweep")
    assert [job["status"] for job in result["jobs"]] == ["failed", "planned"]
    assert result["jobs"][0]["training_steps"] == 1
    assert result["reserved_tokens"] == 15000
    assert run_experiment(tmp_path / "sweep") == result


@pytest.mark.ml
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None and os.environ.get("NDE_REQUIRE_ML") != "1",
    reason="optional CPU ML runtime",
)
def test_registered_sweep_time_caps_stop_cpu_work_and_survive_resume(
    tmp_path: Path, fake_llm: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, evaluation = paired_datasets(tmp_path / "data", fake_llm)
    initialize_model(tmp_path / "base", width=8, max_positions=8192)
    config = tmp_path / "timed.json"
    config.write_text(
        json.dumps(
            {
                "base_model": str(tmp_path / "base"),
                "train_datasets": [str(training)],
                "evaluation_dataset": str(evaluation),
                "subsets": [100],
                "max_jobs": 2,
                "max_total_seconds": 1,
                "test_only": True,
                "training": {
                    "max_steps": 100,
                    "max_seconds": 3600,
                    "max_output_tokens": 2048,
                    "inference_max_tokens": 5000,
                },
            }
        )
    )
    protocol = register_experiment(config, tmp_path / "sweep")
    budgets = [job["binding"]["config"] for job in protocol["jobs"]]
    assert sum(budget["max_seconds"] + budget["inference_max_seconds"] for budget in budgets) <= 1
    ticks = 0.0

    def monotonic() -> float:
        nonlocal ticks
        ticks += 0.05
        return ticks

    monkeypatch.setattr("time.monotonic", monotonic)
    stopped = run_experiment(tmp_path / "sweep")
    assert any(job["status"] == "resource-stop" for job in stopped["jobs"])
    assert all(job.get("training_steps", 0) < 100 for job in stopped["jobs"])
    assert stopped["reserved_seconds"] <= 1
    resumed = run_experiment(tmp_path / "sweep")
    assert resumed["reserved_seconds"] == stopped["reserved_seconds"]
    assert resumed["jobs"] == stopped["jobs"]
