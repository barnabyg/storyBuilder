"""Frozen, author-disjoint corpus assignments, separate from benchmark briefs."""

import html
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from nde.corpus.acquisition import CorpusError, WorkRecord, json_digest, load_work, sha256
from nde.corpus.artifacts import publish
from nde.files import checksum_lf
from nde.llm.config import Strict

type Split = Literal["train", "validation", "test", "gold", "excluded"]
PERCENT = 100


class CohortConfig(Strict):
    version: Literal["1"] = "1"
    id: str = Field(min_length=1)
    works: list[str] = Field(min_length=1)
    seed: int = 7
    author_aliases: dict[str, str] = Field(default_factory=dict[str, str])
    author_splits: dict[str, Split] = Field(default_factory=dict[str, Split])
    protected_author_ids: list[str] = Field(default_factory=list[str])
    protected_work_ids: list[str] = Field(default_factory=list[str])
    excluded_work_ids: list[str] = Field(default_factory=list[str])
    train_percent: int = Field(default=80, ge=0, le=100)
    validation_percent: int = Field(default=10, ge=0, le=100)
    previous: str | None = None


class Assignment(Strict):
    work_dir: str
    work_id: str
    source_id: str
    author_id: str
    source_author_id: str
    raw_sha256: str
    text_sha256: str
    record_sha256: str
    split: Split
    reason: str
    eligible: bool


class Cohort(Strict):
    version: Literal["1"] = "1"
    id: str
    config_path: str
    config_sha256: str
    config: CohortConfig
    assignments: list[Assignment]
    sha256: str


def _canonical(author_id: str, aliases: dict[str, str]) -> str:
    visited: set[str] = set()
    while author_id in aliases:
        if author_id in visited:
            raise CorpusError("ambiguous author alias cycle")
        visited.add(author_id)
        author_id = aliases[author_id]
    if not author_id.strip() or author_id.casefold() in {"unknown", "anonymous", "unresolved"}:
        raise CorpusError("unresolved author identity cannot enter a frozen cohort")
    return author_id


def _records(config: CohortConfig, base: Path) -> list[tuple[Path, WorkRecord, str]]:
    records: list[tuple[Path, WorkRecord, str]] = []
    seen_work: set[str] = set()
    seen_text: set[str] = set()
    names: dict[str, str] = {}
    sources: dict[str, str] = {}
    for alias in config.author_aliases:
        _canonical(alias, config.author_aliases)
    for location in config.works:
        path = (base / location).resolve()
        work, _ = load_work(path)
        author = _canonical(work.author_id, config.author_aliases)
        if work.work_id in seen_work or work.text_sha256 in seen_text:
            raise CorpusError("duplicate work, edition or canonical content identity")
        name = re.sub(r"\W+", "", work.author.casefold())
        if name in names and names[name] != author:
            raise CorpusError("same author name has conflicting identities; register aliases")
        if work.source_id in sources and sources[work.source_id] != work.raw_sha256:
            raise CorpusError("one source identity names different source bytes")
        if any(
            value.casefold() == "unknown"
            for value in (work.edition, work.licence_basis, work.licence_evidence)
        ):
            raise CorpusError("incomplete edition/licence provenance cannot be frozen")
        names[name] = author
        sources[work.source_id] = work.raw_sha256
        seen_work.add(work.work_id)
        seen_text.add(work.text_sha256)
        records.append((path, work, author))
    return sorted(records, key=lambda row: row[1].work_id)


def _inherited_policy(previous: Cohort | None) -> tuple[set[str], set[str]]:
    excluded: set[str] = set()
    protected: set[str] = set()
    while previous is not None:
        excluded.update(previous.config.excluded_work_ids)
        protected.update(
            _canonical(author, previous.config.author_aliases)
            for author in previous.config.protected_author_ids
        )
        protected.update(
            row.author_id
            for row in previous.assignments
            if row.work_id in previous.config.protected_work_ids
        )
        protected.update(
            _canonical(author, previous.config.author_aliases)
            for author, split in previous.config.author_splits.items()
            if split != "train"
        )
        previous = (
            load_cohort(Path(previous.config_path).parent / previous.config.previous)
            if previous.config.previous is not None
            else None
        )
    return excluded, protected


def _author_assignments(
    config: CohortConfig,
    records: list[tuple[Path, WorkRecord, str]],
    previous: Cohort | None,
) -> dict[str, Split]:
    if config.train_percent + config.validation_percent > PERCENT:
        raise CorpusError("split percentages exceed 100")
    authors = {author for _, _, author in records}
    works = {work.work_id for _, work, _ in records}
    if (set(config.protected_work_ids) | set(config.excluded_work_ids)) - works:
        raise CorpusError("reservation names unknown work assignments")
    explicit: dict[str, Split] = {}
    for key, value in config.author_splits.items():
        author = _canonical(key, config.author_aliases)
        if author not in authors or (author in explicit and explicit[author] != value):
            raise CorpusError("unknown or conflicting explicit author assignment")
        explicit[author] = value
    protected = {_canonical(a, config.author_aliases) for a in config.protected_author_ids}
    if protected - authors:
        raise CorpusError("protected reservation names unknown author")
    protected.update(
        author for _, work, author in records if work.work_id in config.protected_work_ids
    )
    protected.update(_inherited_policy(previous)[1])
    if previous is not None:
        _preserve_author_assignments(previous, records, explicit)
    for author in protected:
        if author in explicit and explicit[author] not in {
            "gold",
            "validation",
            "test",
            "excluded",
        }:
            raise CorpusError("protected author cannot enter training")
        explicit.setdefault(author, "gold")
    for author in authors - explicit.keys():
        bucket = int(sha256(f"{config.seed}:{author}".encode()), 16) % 100
        split: Split = (
            "train"
            if bucket < config.train_percent
            else "validation"
            if bucket < config.train_percent + config.validation_percent
            else "test"
        )
        explicit[author] = split
    return explicit


def _preserve_author_assignments(
    previous: Cohort,
    records: list[tuple[Path, WorkRecord, str]],
    explicit: dict[str, Split],
) -> None:
    work_exclusions = _inherited_policy(previous)[0]
    current = {work.work_id: (work, path, author) for path, work, author in records}
    for assignment in previous.assignments:
        row = current.get(assignment.work_id)
        if (
            row is None
            or json_digest(row[0]) != assignment.record_sha256
            or str(row[1]) != assignment.work_dir
            or row[2] != assignment.author_id
        ):
            raise CorpusError("expansion changed or removed a frozen work")
        if assignment.work_id in work_exclusions:
            continue
        if assignment.author_id in explicit and explicit[assignment.author_id] != assignment.split:
            raise CorpusError("expansion reassigned a frozen author")
        explicit[assignment.author_id] = assignment.split


def _build(config_path: Path) -> Cohort:
    config = CohortConfig.model_validate(yaml.safe_load(config_path.read_text("utf-8")))
    records = _records(config, config_path.parent)
    previous = load_cohort(config_path.parent / config.previous) if config.previous else None
    exclusions = set(config.excluded_work_ids) | _inherited_policy(previous)[0]
    splits = _author_assignments(config, records, previous)
    assignments: list[Assignment] = []
    for path, work, author in records:
        split = "excluded" if work.work_id in exclusions else splits[author]
        assignments.append(
            Assignment(
                work_dir=str(path),
                work_id=work.work_id,
                source_id=work.source_id,
                author_id=author,
                source_author_id=work.author_id,
                raw_sha256=work.raw_sha256,
                text_sha256=work.text_sha256,
                record_sha256=json_digest(work),
                split=split,
                reason="explicit exclusion"
                if split == "excluded"
                else "author-disjoint frozen assignment",
                eligible=split == "train",
            )
        )
    if previous is not None:
        current_splits = {assignment.work_id: assignment.split for assignment in assignments}
        if any(current_splits[old.work_id] != old.split for old in previous.assignments):
            raise CorpusError("expansion reassigned a frozen work")
    value = Cohort(
        id=config.id,
        config_path=str(config_path.resolve()),
        config_sha256=checksum_lf(config_path.read_bytes()),
        config=config,
        assignments=assignments,
        sha256="",
    )
    return value.model_copy(update={"sha256": json_digest(value)})


def load_cohort(path: Path) -> Cohort:
    file = path / "cohort.json" if path.is_dir() else path
    cohort = Cohort.model_validate_json(file.read_text("utf-8"))
    if json_digest(cohort.model_copy(update={"sha256": ""})) != cohort.sha256:
        raise CorpusError("changed frozen cohort manifest")
    if checksum_lf(Path(cohort.config_path).read_bytes()) != cohort.config_sha256:
        raise CorpusError("changed frozen cohort configuration")
    if _build(Path(cohort.config_path)) != cohort:
        raise CorpusError("changed frozen cohort configuration, provenance or assignments")
    return cohort


def freeze_cohort(config_path: Path, out: Path) -> Cohort:
    cohort = _build(config_path)
    if out.exists():
        existing = load_cohort(out)
        if existing != cohort:
            raise CorpusError("frozen assignments cannot be overwritten; use a new cohort version")
        return existing
    publish(
        out,
        {
            "report.html": "<!doctype html><meta charset='utf-8'><h1>Frozen corpus cohort</h1><pre>"
            + html.escape(cohort.model_dump_json(indent=2))
            + "</pre>",
            "cohort.json": cohort.model_dump_json(indent=2) + "\n",
        },
    )
    return cohort
