"""Calibration batches: human-written and generated stories, read one at a time (P1.9).

The owner scores a shuffled mix of public-domain human stories and a few
stories from dev-brief runs, without knowing which is which, to anchor the
rubric scale before any gate reading. The report gives each rubric dimension's
scores by source (human, and each generating condition).

A calibration batch is a single-story batch (`View.kind == "single"`). Its
sealed key (`sealed/key.json`) names each story's source and origin; the reader
never opens it, and `nde eval report` reveals it only once the batch is closed.
"""

import random
import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nde.corpus.provenance import read_texts
from nde.eval.batch import (
    BatchError,
    Source,
    View,
    read_closed,
    read_scores,
    read_sealed,
    write_batch,
)
from nde.eval.rubric import Rubric
from nde.eval.sections import BOOTSTRAP_SEED, mean_ci, warnings_section
from nde.runs.records import STORY_FILE, read_stories

HUMAN = "human"


class CalibrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "human", or the generating condition.
    source: str
    # A human story's title and provenance, or a generated story's run and file.
    origin: str


class CalibrationKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: str
    seed: int
    rubric_version: int
    # The generating conditions, in the order they were given.
    conditions: list[str]
    items: list[CalibrationItem]


def _generated(source: Source, count: int, rng: random.Random) -> list[tuple[CalibrationItem, str]]:
    stories = [
        story
        for story in read_stories(source.run_dir)
        if story.condition == source.condition and story.status == "ok"
    ]
    if len(stories) < count:
        raise BatchError(
            f"run {source.run_dir} has {len(stories)} stories for condition "
            f"{source.condition!r}; {count} are needed"
        )
    chosen = rng.sample(stories, count)
    run = source.run_dir.resolve()
    return [
        (
            CalibrationItem(source=source.condition, origin=f"{run}/{story.path}/{STORY_FILE}"),
            (run / story.path / STORY_FILE).read_text(encoding="utf-8"),
        )
        for story in chosen
    ]


@dataclass(frozen=True)
class CalibrationSpec:
    """What a calibration batch mixes: the human stories and each condition's sample."""

    name: str
    humans: Path
    sources: list[Source]
    per_source: int
    rubric: Rubric
    seed: int | None = None


def build_calibration_batch(batches_dir: Path, spec: CalibrationSpec) -> Path:
    """Mix every fetched human story with `per_source` stories of each condition, shuffled."""
    batch = batches_dir / spec.name
    if batch.exists():
        raise BatchError(f"batch {spec.name!r} already exists at {batch}")
    sources, rubric = spec.sources, spec.rubric
    names = [source.condition for source in sources]
    if len(set(names)) != len(names):
        raise BatchError("calibration sources need distinct condition names")
    if any(source.condition == HUMAN for source in sources):
        raise BatchError(f"{HUMAN!r} is reserved for the human-written stories")
    seed = secrets.randbits(32) if spec.seed is None else spec.seed
    rng = random.Random(seed)  # noqa: S311 - sampling and shuffling for blinding
    items = [
        (
            CalibrationItem(
                source=HUMAN, origin=f"{story.title}, {story.author} ({story.edition})"
            ),
            text,
        )
        for story, text in read_texts(spec.humans).values()
    ]
    for source in sources:
        items += _generated(source, spec.per_source, rng)
    rng.shuffle(items)
    view = View(kind="single", stories=[text for _, text in items], rubric=rubric)
    key = CalibrationKey(
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        rubric_version=rubric.version,
        conditions=[source.condition for source in sources],
        items=[item for item, _ in items],
    )
    write_batch(batch, view, key)
    return batch


def render_calibration_report(batch: Path, view: View) -> str:
    """Scores by source for a closed calibration batch; raises BatchError while it is open."""
    key = read_sealed(batch, CalibrationKey)
    rubric = view.rubric
    if rubric is None:  # pragma: no cover - calibration batches always carry a rubric
        raise BatchError("the calibration batch has no rubric")
    scored = read_scores(batch)
    sources = [HUMAN, *key.conditions]
    scores = {s: {d: list[int]() for d in rubric.dimension_ids} for s in sources}
    tags = {source: Counter[str]() for source in sources}
    for number, item in enumerate(key.items, start=1):
        story = scored[number].story
        for dimension, value in story.scores.items():
            scores[item.source][dimension].append(value)
        tags[item.source].update(story.tags)
    counts = Counter(item.source for item in key.items)
    closed = read_closed(batch)
    lines = [
        f"# Calibration report: {batch.name}",
        "",
        f"- **Stories:** {len(key.items)} ({', '.join(f'{counts[s]} {s}' for s in sources)})",
        f"- **Rubric:** v{key.rubric_version}",
        f"- **Closed:** {closed.closed_at if closed else '-'}",
        "",
    ]
    lines += warnings_section(batch)
    lines += [
        "## Rubric scores by source",
        "",
        f"Mean score on the {rubric.scale.min}-{rubric.scale.max} scale, with a 95% percentile "
        f"bootstrap interval (seed {BOOTSTRAP_SEED}). Few stories per source make the intervals "
        "wide: this anchors the scale, it does not compare sources.",
        "",
        f"| Dimension | {' | '.join(sources)} |",
        f"| --- |{' --- |' * len(sources)}",
        *(
            f"| {d} | {' | '.join(mean_ci(scores[s][d]) for s in sources)} |"
            for d in rubric.dimension_ids
        ),
        "",
        "## Failure tags by source",
        "",
        f"| Tag | {' | '.join(sources)} |",
        f"| --- |{' --- |' * len(sources)}",
        *(
            f"| {tag} | {' | '.join(str(tags[s][tag]) for s in sources)} |"
            for tag in rubric.tag_ids
        ),
        "",
        "## Stories (key revealed)",
        "",
        "| Story | Source | Origin |",
        "| --- | --- | --- |",
        *(
            f"| {number} | {item.source} | {item.origin} |"
            for number, item in enumerate(key.items, start=1)
        ),
        "",
    ]
    return "\n".join(lines)
