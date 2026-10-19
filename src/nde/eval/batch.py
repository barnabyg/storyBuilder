"""Blind batches: story pairs to read blind, a sealed key, judgements and closing.

A batch lives in `<batches dir>/<name>/`:

- `view.json`: what the reader sees, and nothing else: the normalised text of
  both stories of each pair, in the order they are shown, the rubric to score
  them with, and the two condition names offered for the condition guess (in a
  fixed order that says nothing about which story is which).
- `sealed/key.json`: which condition and run each shown story came from. Only
  `nde eval report` reads it, and only once the batch is closed; the reader
  app never opens anything under `sealed/`.
- `judgements.jsonl`: the reader's judgements, appended as they are given.
  The latest judgement of a pair is the one that counts.
- `reader-warnings.jsonl`: leakage warnings raised when the reader started.
- `closed.json`: written when the reader closes the batch. After that,
  judgements cannot change.

Batches built before rubric scoring have no rubric and no guess options; they
are read and reported with preferences only.
"""

import random
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nde.eval.rubric import Rubric
from nde.files import append_line, write_text_lf
from nde.runs.records import STORY_FILE, read_manifest, read_stories

DEFAULT_BATCHES_DIR = Path("batches")
VIEW = "view.json"
SEALED = "sealed"
KEY = "key.json"
JUDGEMENTS = "judgements.jsonl"
SCORES = "scores.jsonl"
# Field prefix for the one story of a single-story batch.
SINGLE = "story"
WARNINGS = "reader-warnings.jsonl"
CLOSED = "closed.json"

type Side = Literal["A", "B"]
SIDES: tuple[Side, Side] = ("A", "B")


class BatchError(Exception):
    """A batch cannot be built, read or changed as asked."""


class InvalidJudgementError(BatchError):
    """A submitted judgement is incomplete or names something the batch does not offer."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShownPair(_Record):
    a: str
    b: str


class View(_Record):
    """Everything the reader may see."""

    # pairs: story pairs with a preference; single: one story at a time, scores only.
    kind: Literal["pairs", "single"] = "pairs"
    pairs: list[ShownPair] = Field(default_factory=list[ShownPair])
    stories: list[str] = Field(default_factory=list[str])
    # The rubric to score each story with; None for batches built before rubric scoring.
    rubric: Rubric | None = None
    # The conditions offered for "which condition was A?", sorted by name.
    guess_options: list[str] = Field(default_factory=list[str])
    # When true, only the preference and strength are required: rubric scores, tags
    # and the guess may be left out or given in part (for skim reading).
    optional_scores: bool = False

    @property
    def size(self) -> int:
        """Pairs, or stories in a single-story batch."""
        return len(self.stories) if self.kind == "single" else len(self.pairs)


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


class GeneratorUse(_Record):
    """A model that generated the batch's stories, from its runs' manifests."""

    role: str
    provider: str
    model: str
    family: str | None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


class Key(_Record):
    """The sealed key: where every shown story came from."""

    created_at: str
    seed: int
    sources: list[StorySource]
    pairs: list[SealedPair]
    rubric_version: int | None = None
    # The planner and prose models of the source runs, for the judges' family rule.
    generators: list[GeneratorUse] = Field(default_factory=list[GeneratorUse])


class StoryScores(_Record):
    """The reader's rubric scores and failure tags for one story."""

    scores: dict[str, int]
    tags: list[str] = Field(default_factory=list[str])
    free_tags: list[str] = Field(default_factory=list[str])


class Judgement(_Record):
    pair: int = Field(ge=1)
    preferred: Side
    strength: int = Field(ge=1, le=3)
    at: str
    # Rubric batches only: the guessed condition of story A, and both stories' scores.
    guess: str | None = None
    a: StoryScores | None = None
    b: StoryScores | None = None


class ScoreRecord(_Record):
    """A single-story batch's scores for one story."""

    item: int = Field(ge=1)
    story: StoryScores
    at: str


class ReaderWarning(_Record):
    at: str
    warning: str


class Closed(_Record):
    closed_at: str
    judged: int


@dataclass(frozen=True)
class Scoring:
    """What a pair batch asks for beyond the preference: the rubric, and whether it is optional."""

    rubric: Rubric | None
    # True: rubric scores, tags and the guess may be left out or partial (skim reading).
    optional: bool = False


@dataclass(frozen=True)
class Source:
    run_dir: Path
    condition: str


def labels(sources: Sequence[StorySource]) -> dict[tuple[str, str], str]:
    """A label per source: its condition, numbered when both share a condition name.

    Labels are shown to the reader as the guess options, so they never name a run;
    the report's Sources line maps each label to its run.
    """
    if len({source.condition for source in sources}) == len(sources):
        return {(s.run, s.condition): s.condition for s in sources}
    return {
        (s.run, s.condition): f"{s.condition} (source {number})"
        for number, s in enumerate(sources, start=1)
    }


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


GENERATING_ROLES = ("planner", "prose")


def _generators(sources: Sequence[Source]) -> list[GeneratorUse]:
    found: dict[str, GeneratorUse] = {}
    for source in sources:
        for use in read_manifest(source.run_dir).models:
            if use.role in GENERATING_ROLES:
                generator = GeneratorUse(
                    role=use.role, provider=use.provider, model=use.model, family=use.family
                )
                found.setdefault(generator.label, generator)
    return list(found.values())


def _text(source: StorySource) -> str:
    return (Path(source.run) / source.story).read_text(encoding="utf-8")


def build_batch(
    batches_dir: Path,
    name: str,
    sources: tuple[Source, Source],
    *,
    scoring: Scoring,
    seed: int | None = None,
) -> Path:
    """Pair two conditions' stories by brief and repetition, shuffle, and seal the key."""
    batch_dir = batches_dir / name
    if batch_dir.exists():
        raise BatchError(f"batch {name!r} already exists at {batch_dir}")
    first, second = sources
    if (first.run_dir.resolve(), first.condition) == (second.run_dir.resolve(), second.condition):
        raise BatchError("a batch compares two different conditions")
    left, right = _stories_by_pair(first), _stories_by_pair(second)
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise BatchError("the two conditions have no brief and repetition in common")
    seed = secrets.randbits(32) if seed is None else seed
    rng = random.Random(seed)  # noqa: S311 - shuffling for blinding, not cryptography
    rng.shuffle(keys)
    origins = [
        StorySource(run=str(source.run_dir.resolve()), condition=source.condition, story="")
        for source in sources
    ]
    sealed: list[SealedPair] = []
    shown: list[ShownPair] = []
    for brief, repetition in keys:
        sides = [
            origins[0].model_copy(update={"story": left[brief, repetition]}),
            origins[1].model_copy(update={"story": right[brief, repetition]}),
        ]
        rng.shuffle(sides)
        sealed.append(SealedPair(brief=brief, repetition=repetition, a=sides[0], b=sides[1]))
        shown.append(ShownPair(a=_text(sides[0]), b=_text(sides[1])))
    view = View(
        pairs=shown,
        rubric=scoring.rubric,
        guess_options=sorted(labels(origins).values()) if scoring.rubric is not None else [],
        optional_scores=scoring.optional,
    )
    key = Key(
        created_at=datetime.now(UTC).isoformat(),
        seed=seed,
        sources=origins,
        pairs=sealed,
        rubric_version=None if scoring.rubric is None else scoring.rubric.version,
        generators=_generators(sources),
    )
    write_batch(batch_dir, view, key)
    return batch_dir


def write_batch(batch: Path, view: View, key: BaseModel) -> None:
    """Create a batch: the reader's view, and the key sealed apart from it."""
    (batch / SEALED).mkdir(parents=True)
    write_text_lf(batch / VIEW, view.model_dump_json(indent=2) + "\n")
    write_text_lf(batch / SEALED / KEY, key.model_dump_json(indent=2) + "\n")


def read_sealed[T: BaseModel](batch: Path, model: type[T]) -> T:
    """The sealed key. Refuses while the batch is open, so nothing reveals it early."""
    if not is_closed(batch):
        raise BatchError("the batch is still open; close it in the reader first")
    return model.model_validate_json((batch / SEALED / KEY).read_text(encoding="utf-8"))


def batch_dir(batches_dir: Path, name: str) -> Path:
    path = batches_dir / name
    if not (path / VIEW).is_file():
        raise BatchError(f"no batch {name!r} in {batches_dir}")
    return path


def read_view(batch: Path) -> View:
    return View.model_validate_json((batch / VIEW).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_judgements(batch: Path) -> dict[int, Judgement]:
    """The latest judgement of each pair, by pair number."""
    latest: dict[int, Judgement] = {}
    for line in _jsonl(batch / JUDGEMENTS):
        judgement = Judgement.model_validate_json(line)
        latest[judgement.pair] = judgement
    return latest


def read_scores(batch: Path) -> dict[int, ScoreRecord]:
    """The latest scores of each story of a single-story batch, by story number."""
    latest: dict[int, ScoreRecord] = {}
    for line in _jsonl(batch / SCORES):
        record = ScoreRecord.model_validate_json(line)
        latest[record.item] = record
    return latest


def judged(batch: Path, view: View) -> set[int]:
    """The numbers of the pairs, or stories, judged so far."""
    return set(read_scores(batch) if view.kind == "single" else read_judgements(batch))


def is_closed(batch: Path) -> bool:
    return (batch / CLOSED).is_file()


_TAG_SEPARATORS = re.compile(r"[,;]")


def _one(fields: Mapping[str, Sequence[str]], name: str) -> str | None:
    values = [value for value in fields.get(name, []) if value.strip()]
    return values[0].strip() if len(values) == 1 else None


def _scores(
    fields: Mapping[str, Sequence[str]],
    rubric: Rubric,
    side: str,
    problems: list[str],
    *,
    required: bool = True,
) -> StoryScores:
    scores: dict[str, int] = {}
    scale = rubric.scale
    for dimension in rubric.dimension_ids:
        name = f"score-{side}-{dimension}"
        value = _one(fields, name)
        if value is None and not required:
            continue
        if value is None or not value.isdigit() or not scale.min <= int(value) <= scale.max:
            problems.append(f"{name}: give a score from {scale.min} to {scale.max}")
        else:
            scores[dimension] = int(value)
    tags = [tag for tag in fields.get(f"tags-{side}", []) if tag]
    unknown = [tag for tag in tags if tag not in rubric.tag_ids]
    if unknown:
        problems.append(f"tags-{side}: unknown failure tags {', '.join(unknown)}")
    free = [
        part.strip()
        for value in fields.get(f"free-tags-{side}", [])
        for part in _TAG_SEPARATORS.split(value)
        if part.strip()
    ]
    return StoryScores(scores=scores, tags=sorted(set(tags)), free_tags=free)


def parse_judgement(view: View, pair: int, fields: Mapping[str, Sequence[str]]) -> Judgement:
    """A judgement from submitted form fields; raises InvalidJudgementError naming each gap."""
    problems: list[str] = []
    if not 1 <= pair <= len(view.pairs):
        raise BatchError(f"no pair {pair}")
    preferred = _one(fields, "preferred")
    if preferred not in SIDES:
        problems.append("preferred: choose story A or story B")
    strength = _one(fields, "strength")
    if strength not in {"1", "2", "3"}:
        problems.append("strength: choose 1, 2 or 3")
    judgement = Judgement(
        pair=pair,
        preferred="B" if preferred == "B" else "A",
        strength=int(strength) if strength in {"1", "2", "3"} else 1,
        at=datetime.now(UTC).isoformat(),
    )
    if view.rubric is not None:
        required = not view.optional_scores
        judgement.guess = _one(fields, "guess")
        if judgement.guess not in view.guess_options and (required or judgement.guess is not None):
            problems.append(f"guess: choose one of {', '.join(view.guess_options)}")
        judgement.a = _scores(fields, view.rubric, "a", problems, required=required)
        judgement.b = _scores(fields, view.rubric, "b", problems, required=required)
    if problems:
        raise InvalidJudgementError(problems)
    return judgement


def parse_scores(view: View, item: int, fields: Mapping[str, Sequence[str]]) -> ScoreRecord:
    """One story's scores from submitted form fields; raises InvalidJudgementError."""
    if view.rubric is None or not 1 <= item <= len(view.stories):
        raise BatchError(f"no story {item}")
    problems: list[str] = []
    scores = _scores(fields, view.rubric, SINGLE, problems)
    if problems:
        raise InvalidJudgementError(problems)
    return ScoreRecord(item=item, story=scores, at=datetime.now(UTC).isoformat())


def record_scores(batch: Path, record: ScoreRecord) -> None:
    """Append a story's scores, durably, before returning."""
    if is_closed(batch):
        raise BatchError("the batch is closed; scores cannot change")
    append_line(batch / SCORES, record.model_dump_json())


def record_judgement(batch: Path, judgement: Judgement) -> None:
    """Append a judgement, durably, before returning."""
    if is_closed(batch):
        raise BatchError("the batch is closed; judgements cannot change")
    append_line(batch / JUDGEMENTS, judgement.model_dump_json())


def record_warning(batch: Path, warning: str) -> None:
    entry = ReaderWarning(at=datetime.now(UTC).isoformat(), warning=warning)
    append_line(batch / WARNINGS, entry.model_dump_json())


def read_warnings(batch: Path) -> list[ReaderWarning]:
    return [ReaderWarning.model_validate_json(line) for line in _jsonl(batch / WARNINGS)]


def close_batch(batch: Path) -> Closed:
    if is_closed(batch):
        raise BatchError("the batch is already closed")
    view = read_view(batch)
    done = len(judged(batch, view))
    unit = "stories" if view.kind == "single" else "pairs"
    if done < view.size:
        raise BatchError(f"{view.size - done} of {view.size} {unit} are not judged yet")
    closed = Closed(closed_at=datetime.now(UTC).isoformat(), judged=done)
    write_text_lf(batch / CLOSED, closed.model_dump_json(indent=2) + "\n")
    return closed


def read_closed(batch: Path) -> Closed | None:
    if not is_closed(batch):
        return None
    return Closed.model_validate_json((batch / CLOSED).read_text(encoding="utf-8"))


def read_generators(batch: Path) -> list[GeneratorUse]:
    """The models that generated the batch. Reveals no pairing, so it is readable while open."""
    key = Key.model_validate_json((batch / SEALED / KEY).read_text(encoding="utf-8"))
    return key.generators


def read_key(batch: Path) -> Key:
    """A pair batch's sealed key; refuses while the batch is open."""
    return read_sealed(batch, Key)
