"""Helpers for evaluation tests: story runs to batch, and a reader client."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from nde.cli import main
from nde.eval.reader import create_app
from tests.conftest import REPO
from tests.fake_openai import FakeOpenAI, Json, Reply, completion
from tests.stories import run, story_config, story_text

RUBRIC_DIR = REPO / "benchmark" / "rubric"
DIMENSIONS = [
    "engagement",
    "coherence",
    "emotional_impact",
    "character_believability",
    "pacing",
    "originality",
]

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
            "--rubric-dir",
            str(RUBRIC_DIR),
            *extra,
        ]
    )
    assert code == 0
    return batches / name


def reader(batch_dir: Path) -> TestClient:
    return TestClient(create_app(batch_dir), base_url="http://127.0.0.1")


type Form = dict[str, str | list[str]]


@dataclass(frozen=True)
class Marks:
    """The reader's marks for one story: one score for every dimension, and tags."""

    score: int = 3
    tags: list[str] = field(default_factory=list[str])
    free_tags: str = ""


def complete_form(
    preferred: str = "A",
    strength: int = 2,
    guess: str = "intact",
    a: Marks | None = None,
    b: Marks | None = None,
) -> Form:
    """A complete judgement: preference, strength, guess, every score, and tags."""
    form: Form = {"preferred": preferred, "strength": str(strength), "guess": guess}
    for side, marks in (("a", a or Marks()), ("b", b or Marks())):
        for dimension in DIMENSIONS:
            form[f"score-{side}-{dimension}"] = str(marks.score)
        form[f"tags-{side}"] = marks.tags
        form[f"free-tags-{side}"] = marks.free_tags
    return form
