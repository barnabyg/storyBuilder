"""Blind batches: story pairs to read blind, a sealed key, judgements and closing.

A batch lives in `<batches dir>/<name>/`:

- `view.json`: what the reader sees, and nothing else: the normalised text of
  both stories of each pair, in the order they are shown.
- `sealed/key.json`: which condition and run each shown story came from. Only
  `nde eval report` reads it, and only once the batch is closed; the reader
  app never opens anything under `sealed/`.
- `judgements.jsonl`: the reader's judgements, appended as they are given.
  The latest judgement of a pair is the one that counts.
- `closed.json`: written when the reader closes the batch. After that,
  judgements cannot change.
"""

import random
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nde.files import write_text_lf
from nde.runs.records import read_stories

DEFAULT_BATCHES_DIR = Path("batches")
VIEW = "view.json"
SEALED = "sealed"
KEY = "key.json"
JUDGEMENTS = "judgements.jsonl"
CLOSED = "closed.json"
STORY_FILE = "story.md"

type Side = Literal["A", "B"]


class BatchError(Exception):
    """A batch cannot be built, read or changed as asked."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShownPair(_Record):
    a: str
    b: str


class View(_Record):
    """Everything the reader may see."""

    pairs: list[ShownPair]


class StorySource(_Record):
    run: str
    condition: str
    # The story file, relative to the run directory.
    story: str


class SealedPair(_Record):
    brief: str
    repetition: int
    a: StorySource
    b: StorySource


class Key(_Record):
    """The sealed key: where every shown story came from."""

    created_at: str
    seed: int
    sources: list[StorySource]
    pairs: list[SealedPair]


class Judgement(_Record):
    pair: int = Field(ge=1)
    preferred: Side
    strength: int = Field(ge=1, le=3)
    at: str


class Closed(_Record):
    closed_at: str
    judged: int


@dataclass(frozen=True)
class Source:
    run_dir: Path
    condition: str


def _stories_by_pair(source: Source) -> dict[tuple[str, int], str]:
    """The run's usable stories for one condition, keyed by (brief, repetition)."""
    stories = {
        (story.brief, story.repetition): f"{story.path}/{STORY_FILE}"
        for story in read_stories(source.run_dir)
        if story.condition == source.condition and story.status == "ok"
    }
    if not stories:
        raise BatchError(f"run {source.run_dir} has no stories for condition {source.condition!r}")
    return stories


def build_batch(
    batches_dir: Path, name: str, first: Source, second: Source, *, seed: int | None = None
) -> Path:
    """Pair two conditions' stories by brief and repetition, shuffle, and seal the key."""
    batch_dir = batches_dir / name
    if batch_dir.exists():
        raise BatchError(f"batch {name!r} already exists at {batch_dir}")
    if (first.run_dir.resolve(), first.condition) == (second.run_dir.resolve(), second.condition):
        raise BatchError("a batch compares two different conditions")
    left, right = _stories_by_pair(first), _stories_by_pair(second)
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise BatchError("the two conditions have no brief and repetition in common")
    seed = secrets.randbits(32) if seed is None else seed
    rng = random.Random(seed)  # noqa: S311 - shuffling for blinding, not cryptography
    rng.shuffle(keys)
    sealed: list[SealedPair] = []
    shown: list[ShownPair] = []
    for brief, repetition in keys:
        sides = [
            StorySource(
                run=str(first.run_dir.resolve()),
                condition=first.condition,
                story=left[brief, repetition],
            ),
            StorySource(
                run=str(second.run_dir.resolve()),
                condition=second.condition,
                story=right[brief, repetition],
            ),
        ]
        rng.shuffle(sides)
        sealed.append(SealedPair(brief=brief, repetition=repetition, a=sides[0], b=sides[1]))
        shown.append(ShownPair(a=_text(sides[0]), b=_text(sides[1])))
    key = Key(
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        sources=[
            StorySource(run=str(source.run_dir.resolve()), condition=source.condition, story="")
            for source in (first, second)
        ],
        pairs=sealed,
    )
    (batch_dir / SEALED).mkdir(parents=True)
    write_text_lf(batch_dir / VIEW, View(pairs=shown).model_dump_json(indent=2) + "\n")
    write_text_lf(batch_dir / SEALED / KEY, key.model_dump_json(indent=2) + "\n")
    return batch_dir


def _text(source: StorySource) -> str:
    return (Path(source.run) / source.story).read_text(encoding="utf-8")


def batch_dir(batches_dir: Path, name: str) -> Path:
    path = batches_dir / name
    if not (path / VIEW).is_file():
        raise BatchError(f"no batch {name!r} in {batches_dir}")
    return path


def read_view(batch: Path) -> View:
    return View.model_validate_json((batch / VIEW).read_text(encoding="utf-8"))


def read_judgements(batch: Path) -> dict[int, Judgement]:
    """The latest judgement of each pair, by pair number."""
    path = batch / JUDGEMENTS
    latest: dict[int, Judgement] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                judgement = Judgement.model_validate_json(line)
                latest[judgement.pair] = judgement
    return latest


def is_closed(batch: Path) -> bool:
    return (batch / CLOSED).is_file()


def record_judgement(batch: Path, pair: int, preferred: Side, strength: int) -> Judgement:
    """Append a judgement, durably, before returning."""
    if is_closed(batch):
        raise BatchError("the batch is closed; judgements cannot change")
    if not 1 <= pair <= len(read_view(batch).pairs):
        raise BatchError(f"no pair {pair}")
    judgement = Judgement(
        pair=pair, preferred=preferred, strength=strength, at=datetime.now(UTC).isoformat()
    )
    with (batch / JUDGEMENTS).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(judgement.model_dump_json() + "\n")
        handle.flush()
    return judgement


def close_batch(batch: Path) -> Closed:
    if is_closed(batch):
        raise BatchError("the batch is already closed")
    pairs = len(read_view(batch).pairs)
    judged = len(read_judgements(batch))
    if judged < pairs:
        raise BatchError(f"{pairs - judged} of {pairs} pairs are not judged yet")
    closed = Closed(closed_at=datetime.now(UTC).isoformat(), judged=judged)
    write_text_lf(batch / CLOSED, closed.model_dump_json(indent=2) + "\n")
    return closed


def read_closed(batch: Path) -> Closed | None:
    if not is_closed(batch):
        return None
    return Closed.model_validate_json((batch / CLOSED).read_text(encoding="utf-8"))


def read_key(batch: Path) -> Key:
    """The sealed key. Refuses while the batch is open, so nothing reveals it early."""
    if not is_closed(batch):
        raise BatchError("the batch is still open; close it in the reader first")
    return Key.model_validate_json((batch / SEALED / KEY).read_text(encoding="utf-8"))
