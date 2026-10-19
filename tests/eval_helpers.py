"""Helpers for evaluation tests: story runs to batch, and a reader client."""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from nde.cli import main
from nde.eval.reader import create_app
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import run, story_config, story_text

# Distinctive names, so a leak of the key into the reader is easy to spot.
INTACT = {"name": "intact", "pipeline": "baseline_a"}
CONTROL = {"name": "control", "pipeline": "baseline_a"}
BRIEFS = ["brief-01", "brief-02", "brief-03", "brief-04"]


def seeded_story(body: Json) -> Reply:
    """A 2,000-word story whose words name the request's seed."""
    return completion(story_text(2000, marker=f"w{body['seed']}"), completion_tokens=2000)


def two_runs(fake: FakeOpenAI, tmp_path: Path) -> tuple[Path, Path]:
    """Two story runs on the same briefs, with different seeds and condition names."""
    fake.default = seeded_story
    configs: list[dict[str, Any]] = []
    for condition, seed in ((INTACT, 100), (CONTROL, 200)):
        config = story_config(fake.base_url, [condition], briefs={"ids": BRIEFS})
        config["seed"] = seed
        config["roles"]["prose"]["model"] = "secret-model-id"
        config["prices"]["fake/secret-model-id"] = {"input_per_mtok": 0, "output_per_mtok": 0}
        configs.append(config)
    return run(tmp_path, configs[0]), run(tmp_path, configs[1])


def build_batch(tmp_path: Path, runs: tuple[Path, Path], name: str = "demo", *extra: str) -> Path:
    first, second = runs
    batches = tmp_path / "batches"
    code = main(
        [
            "eval",
            "batch",
            name,
            "--from",
            str(first),
            "intact",
            "--from",
            str(second),
            "control",
            "--batches-dir",
            str(batches),
            *extra,
        ]
    )
    assert code == 0
    return batches / name


def reader(batch_dir: Path) -> TestClient:
    return TestClient(create_app(batch_dir), base_url="http://127.0.0.1")
