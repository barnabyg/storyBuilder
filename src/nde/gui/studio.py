"""What the GUI works with: story ideas on disk, one writing job at a time, saved stories.

An idea typed into the GUI is saved as a brief in its own folder, never in the benchmark,
so the development and held-out splits stay as they are. The benchmark's dev briefs are
offered as examples; held-out briefs are never opened. Changing or removing an example
is recorded beside the typed ideas, and never touches the benchmark's files.
"""

import json
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nde.briefs import BriefError, load_briefs
from nde.continuity.pipeline import RESULT_FILE
from nde.files import write_text_lf
from nde.runs.config import BriefSelection, ConfigError, load_config
from nde.runs.records import read_manifest, read_stories
from nde.runs.runner import PreparedRun, RunOptions, execute, prepare
from nde.runs.state import RunState
from nde.schema.brief import Brief, BriefContent
from nde.stories.normalise import word_count

DEFAULT_OWN_BRIEFS = Path(".nde/gui-briefs")
# Above every benchmark brief, so a story's brief ID names one idea wherever it is kept.
FIRST_OWN_BRIEF = 1001
SPLIT = "dev"
EXAMPLES_FOLDER = "examples"
REMOVED_EXAMPLES = "removed-benchmark-examples.yaml"


class StudioBusyError(Exception):
    """A story is already being written; carries its job."""

    def __init__(self, job: "Job") -> None:
        self.job = job
        super().__init__("a story is already being written")


def describe_step(call: str | None) -> str:
    """What a story's current call is doing, in plain words."""
    step = "" if call is None else call.rsplit("/", maxsplit=1)[-1]
    attempt = re.search(r" \(attempt \d+\)$", step)
    step = step.removesuffix(attempt.group()) if attempt else step
    if step == "plan":
        return "Planning the characters and scenes"
    if scene := re.fullmatch(r"scene-(\d+)", step):
        again = ", trying again to reach the length" if attempt else ""
        return f"Writing scene {int(scene.group(1))}{again}"
    if step.startswith("audit-"):
        return "Checking the story for problems"
    if step.startswith("repair-"):
        return "Repairing the passages the checks flagged"
    return "Getting started"


@dataclass
class Job:
    """A story being written on a background thread."""

    id: str
    started: float
    cap_usd: float | None
    state: RunState = field(default_factory=RunState)
    thread: threading.Thread | None = None
    run_dir: Path | None = None
    # Set only when the run itself broke, rather than recording a failure.
    crash: str | None = None
    finished: bool = False

    @property
    def spent_usd(self) -> float:
        return sum(call.cost_usd for call in list(self.state.calls))

    @property
    def activity(self) -> str:
        return describe_step(self.state.current_call)

    @property
    def minutes(self) -> int:
        return int((time.monotonic() - self.started) // 60)


@dataclass(frozen=True)
class SavedStory:
    """A story saved in a run directory, as its record describes it."""

    run: str
    brief: str
    folder: Path
    title: str
    finished: bool
    cost_usd: float
    flagged: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]

    @property
    def text(self) -> str:
        return (self.folder / "story.md").read_text(encoding="utf-8")

    @property
    def words(self) -> int:
        return word_count(self.text)

    @property
    def written_on(self) -> str:
        """The day the run started, from its directory name, such as 31 Oct 2026."""
        stamp = time.strptime(self.run[:8], "%Y%m%d")
        return time.strftime("%d %b %Y", stamp).lstrip("0")

    def characters(self) -> list[dict[str, Any]]:
        path = self.folder / "character-sheets.json"
        if not path.is_file():
            return []
        sheets: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return sheets

    def repairs(self) -> list[dict[str, Any]]:
        """The changes that altered the text; a passage given back unchanged is no repair."""
        path = self.folder / "repairs.json"
        if not path.is_file():
            return []
        changes: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return [change for change in changes if change["before"] != change["after"]]


def _saved_story(path: Path) -> SavedStory:
    record = json.loads(path.read_text(encoding="utf-8"))
    return SavedStory(
        run=path.parents[4].name,
        brief=record["brief"],
        folder=path.parent,
        title=record["title"],
        finished=record["finished"],
        cost_usd=record["cost"]["usd"],
        flagged=record["flagged_problems"],
        unresolved=record["unresolved_problems"],
    )


def _failure(run_dir: Path) -> str | None:
    """Why a finished run has no complete story, or None if nothing went wrong."""
    manifest = read_manifest(run_dir)
    if manifest.abort_message is not None:
        return manifest.abort_message
    if manifest.error is not None:
        return manifest.error.message
    return next((story.error for story in read_stories(run_dir) if story.error), None)


def _brief_number(brief: Brief) -> int:
    return int(brief.id.removeprefix("brief-"))


def _brief_yaml(brief: Brief) -> str:
    return yaml.safe_dump(brief.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def _replace_text(path: Path, text: str) -> None:
    """Write a file in one step, so a failure part way cannot leave a broken example."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    write_text_lf(partial, text)
    partial.replace(path)


class Studio:
    """Writes one story at a time and finds the ones already written."""

    def __init__(
        self, config: Path, options: RunOptions, own_briefs: Path = DEFAULT_OWN_BRIEFS
    ) -> None:
        self.config = config
        # Every press of "write" should give a new story, never a stored one.
        self.options = RunOptions(
            runs_dir=options.runs_dir,
            environ=options.environ,
            cache_path=options.cache_path,
            cache_mode="bypass",
            ledger_path=options.ledger_path,
            clock=options.clock,
        )
        self.own_briefs = own_briefs
        self._examples_root = own_briefs / EXAMPLES_FOLDER
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._current: Job | None = None

    @property
    def cap_usd(self) -> float | None:
        return load_config(self.config).config.budget.per_run_usd

    def _benchmark_examples(self) -> dict[str, Brief]:
        folder = load_config(self.config).briefs_dir
        if folder is None:
            raise ConfigError(f"config {self.config} has no `stories` section")
        try:
            return load_briefs(folder, (SPLIT,))
        except BriefError as error:
            raise ConfigError(str(error)) from error

    def _own_examples(self) -> dict[str, Brief]:
        if not (self._examples_root / SPLIT).is_dir():
            return {}
        try:
            return load_briefs(self._examples_root, (SPLIT,))
        except BriefError as error:
            raise ConfigError(str(error)) from error

    def _removed_examples(self) -> set[str]:
        """The benchmark examples taken off the list; their files are never touched."""
        path = self._examples_root / REMOVED_EXAMPLES
        if not path.is_file():
            return set()
        return set(yaml.safe_load(path.read_text(encoding="utf-8")) or [])

    def _set_removed_examples(self, removed: set[str]) -> None:
        self._examples_root.mkdir(parents=True, exist_ok=True)
        _replace_text(
            self._examples_root / REMOVED_EXAMPLES,
            yaml.safe_dump(sorted(removed)) if removed else "",
        )

    def examples(self) -> list[Brief]:
        """The ideas offered as starting points: the benchmark's dev briefs, then the owner's."""
        removed = self._removed_examples()
        kept = [b for b in self._benchmark_examples().values() if b.id not in removed]
        return [
            *sorted(kept, key=lambda brief: brief.id),
            *sorted(self._own_examples().values(), key=_brief_number),
        ]

    def example(self, brief_id: str) -> Brief | None:
        """An example still on the list, by ID."""
        return next((brief for brief in self.examples() if brief.id == brief_id), None)

    def is_benchmark_example(self, brief_id: str) -> bool:
        return brief_id in self._benchmark_examples()

    def removed_example_count(self) -> int:
        return len(self._removed_examples() & self._benchmark_examples().keys())

    def add_example(self, content: BriefContent) -> Brief:
        with self._lock:
            return self._add_example(content)

    def _add_example(self, content: BriefContent) -> Brief:
        brief = Brief(**content.model_dump(), id=self._next_own_id(), split=SPLIT)
        _replace_text(self._own_example_path(brief.id), _brief_yaml(brief))
        return brief

    def update_example(self, brief_id: str, content: BriefContent) -> Brief:
        """Change an example; a benchmark example is replaced by the owner's changed copy."""
        with self._lock:
            if brief_id in self._own_examples():
                brief = Brief(**content.model_dump(), id=brief_id, split=SPLIT)
                _replace_text(self._own_example_path(brief_id), _brief_yaml(brief))
                return brief
            # Copied before the original goes, so a failure cannot lose the example.
            brief = self._add_example(content)
            self._set_removed_examples(self._removed_examples() | {brief_id})
            return brief

    def delete_example(self, brief_id: str) -> None:
        with self._lock:
            if brief_id in self._own_examples():
                self._own_example_path(brief_id).unlink()
            else:
                self._set_removed_examples(self._removed_examples() | {brief_id})

    def restore_examples(self) -> None:
        """Put back every benchmark example taken off the list; the owner's stay as they are."""
        with self._lock:
            self._set_removed_examples(set())

    def _own_ideas(self) -> dict[str, Brief]:
        if not (self.own_briefs / SPLIT).is_dir():
            return {}
        return load_briefs(self.own_briefs, (SPLIT,))

    def find_idea(self, brief_id: str) -> Brief | None:
        """An idea by ID: typed into the GUI, an example, or a benchmark brief taken off the list.

        A story's page needs its idea even after that example has left the list.
        """
        ideas = self._own_ideas() | self._benchmark_examples() | self._own_examples()
        return ideas.get(brief_id)

    def _next_own_id(self) -> str:
        """The next free ID, counted across typed ideas and examples so each names one idea."""
        found = [
            *(self.own_briefs / SPLIT).glob("brief-*.yaml"),
            *(self._examples_root / SPLIT).glob("brief-*.yaml"),
        ]
        numbers = [int(path.stem.removeprefix("brief-")) for path in found]
        return f"brief-{max(numbers, default=FIRST_OWN_BRIEF - 1) + 1}"

    def _own_example_path(self, brief_id: str) -> Path:
        return self._examples_root / SPLIT / f"{brief_id}.yaml"

    def _save_idea(self, content: BriefContent) -> str:
        folder = self.own_briefs / SPLIT
        folder.mkdir(parents=True, exist_ok=True)
        brief_id = self._next_own_id()
        brief = Brief(**content.model_dump(), id=brief_id, split=SPLIT)
        write_text_lf(folder / f"{brief_id}.yaml", _brief_yaml(brief))
        return brief_id

    @property
    def current(self) -> Job | None:
        """The job still writing, if any."""
        job = self._current
        return job if job is not None and not job.finished else None

    def job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def start(self, content: BriefContent) -> Job:
        """Save the idea and start writing it; raises StudioBusyError or ConfigError.

        ConfigError means nothing was spent and no idea was kept.
        """
        with self._lock:
            if (running := self.current) is not None:
                raise StudioBusyError(running)
            brief_id = self._save_idea(content)
            try:
                prepared = prepare(
                    self.config,
                    self.options.environ,
                    briefs=BriefSelection(split=SPLIT, ids=[brief_id]),
                    briefs_dir=self.own_briefs,
                )
            except ConfigError:
                (self.own_briefs / SPLIT / f"{brief_id}.yaml").unlink()
                raise
            job = Job(
                id=secrets.token_hex(8),
                started=time.monotonic(),
                cap_usd=prepared.loaded.config.budget.per_run_usd,
            )
            job.thread = threading.Thread(target=self._write, args=(job, prepared), daemon=True)
            self._jobs[job.id] = job
            self._current = job
            job.thread.start()
            return job

    def _write(self, job: Job, prepared: PreparedRun) -> None:
        try:
            job.run_dir = execute(prepared, self.options, job.state)
        except Exception as error:  # shown on the job's page; the thread has no caller
            job.crash = repr(error)
        finally:
            job.finished = True

    def outcome(self, job: Job) -> tuple[SavedStory | None, str | None]:
        """A finished job's story, if any was saved, and what went wrong, if anything."""
        if job.run_dir is None:
            return None, job.crash
        found = sorted(job.run_dir.glob(f"stories/*/*/r1/{RESULT_FILE}"))
        story = _saved_story(found[0]) if found else None
        return story, job.crash or _failure(job.run_dir)

    def stories(self) -> list[SavedStory]:
        """Every saved continuity story, newest first."""
        runs = self.options.runs_dir
        found = runs.glob(f"*/stories/*/*/r1/{RESULT_FILE}")
        return sorted((_saved_story(path) for path in found), key=lambda s: s.run, reverse=True)

    def story(self, run: str, brief: str) -> SavedStory | None:
        # Matched against what is on disk, so the URL can never name another path.
        return next((s for s in self.stories() if (s.run, s.brief) == (run, brief)), None)

    def delete(self, story: SavedStory) -> None:
        """Permanently remove a story's files; raises StudioBusyError if its run is still going.

        The run's folder goes too unless it holds other stories. The cost ledger and the
        typed idea stay: money spent stays spent, and the idea can be used again.
        """
        run_dir = story.folder.parents[3]
        with self._lock:
            running = self.current
            if running is not None and read_manifest(run_dir).status == "running":
                raise StudioBusyError(running)
            others = [path for path in run_dir.glob("stories/*/*/*") if path != story.folder]
            shutil.rmtree(story.folder if others else run_dir)


def problems_in(error: ValidationError) -> list[tuple[str, str]]:
    """A brief's validation problems, as the field's dotted name and pydantic's message."""
    return [
        (".".join(str(part) for part in item["loc"]), item["msg"])
        for item in error.errors(include_url=False)
    ]
