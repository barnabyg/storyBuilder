"""Strict, inspectable admission of an explicitly registered cohort.

This is independent of the ordinary batch builder's intersection pairing.
All expected pairs are retained in the report, including failed and absent ones.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nde import briefs
from nde.eval.batch import BatchError, Scoring, Source, build_batch
from nde.files import checksum_lf, write_text_lf
from nde.runs.records import (
    STORY_FILE,
    Budget,
    CallRecord,
    Manifest,
    StoryRecord,
    read_calls,
    read_manifest,
    read_stories,
)
from nde.schema.brief import Brief
from nde.stories.budget import story_budget, within
from nde.stories.length import in_range
from nde.stories.normalise import normalise, word_count

SHA256_HEX_LENGTH = 64


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateCohort(Record):
    """Registration data supplied before admission, never inferred from successes."""

    schema_version: Literal["1"] = "1"
    id: str = Field(min_length=1)
    brief_checksums: dict[str, str] = Field(min_length=1)
    split_checksums: dict[str, str] = Field(min_length=1)
    conditions: tuple[str, str]
    repeat: int = Field(default=1, ge=1)
    length_tolerance: float = Field(gt=0, lt=1)
    budget_measure: Literal["generation_tokens"] = "generation_tokens"
    budget_tolerance: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _identities(self) -> Self:
        if self.conditions[0] == self.conditions[1]:
            raise ValueError("two distinct conditions are required")
        for checksum in [*self.brief_checksums.values(), *self.split_checksums.values()]:
            if len(checksum) != SHA256_HEX_LENGTH or any(
                c not in "0123456789abcdef" for c in checksum
            ):
                raise ValueError("checksums must be lowercase SHA-256 values")
        return self


class AdmittedStory(Record):
    condition: str
    run: str
    path: str
    sha256: str
    words: int
    budget: Budget
    new_spend_usd: float


class PairReadiness(Record):
    brief: str
    repetition: int
    stories: list[AdmittedStory] = Field(default_factory=list[AdmittedStory])
    problems: list[str] = Field(default_factory=list[str])


class ReadinessReport(Record):
    schema_version: Literal["1"] = "1"
    status: Literal["malformed", "incomplete", "ready"]
    cohort: GateCohort | None = None
    cohort_sha256: str | None = None
    benchmark: str
    sources: list[dict[str, str]]
    source_checksums: dict[str, dict[str, str]] = Field(default_factory=dict[str, dict[str, str]])
    problems: list[str] = Field(default_factory=list[str])
    pairs: list[PairReadiness] = Field(default_factory=list[PairReadiness])


def cohort_checksum(cohort: GateCohort) -> str:
    return checksum_lf(json.dumps(cohort.model_dump(mode="json"), sort_keys=True).encode())


def _benchmark(cohort: GateCohort, root: Path, problems: list[str]) -> dict[str, Brief]:
    result = briefs.check(root)
    problems.extend(result.problems)
    found = {file.brief.id: file for files in result.briefs.values() for file in files}
    for brief, expected in cohort.brief_checksums.items():
        if brief not in found:
            problems.append(f"{brief}: missing benchmark brief")
        elif found[brief].sha256 != expected:
            problems.append(f"{brief}: benchmark checksum differs from registration")
    for split, expected in cohort.split_checksums.items():
        if split not in {"dev", "test"}:
            problems.append(f"unsupported frozen split {split!r}")
        elif briefs.split_checksum(root, "dev" if split == "dev" else "test") != expected:
            problems.append(f"{split}: frozen split checksum differs from registration")
    frozen = briefs.read_frozen(root)
    for brief in cohort.brief_checksums:
        if brief in found:
            split = found[brief].brief.split
            if (
                split not in cohort.split_checksums
                or frozen.get(split, {}).get(brief) != found[brief].sha256
            ):
                problems.append(f"{brief}: not bound to a matching registered frozen split")
    return {key: file.brief for key, file in found.items()}


def _manifest_problems(
    cohort: GateCohort, source: Source, manifest: Manifest, problems: list[str]
) -> None:
    prefix = f"{source.condition} ({source.run_dir})"
    summary = manifest.stories
    if manifest.status != "succeeded":
        problems.append(f"{prefix}: run is {manifest.status}")
    if summary is None:
        problems.append(f"{prefix}: missing story summary")
        return
    if set(summary.brief_ids) != set(cohort.brief_checksums) or len(summary.brief_ids) != len(
        cohort.brief_checksums
    ):
        problems.append(f"{prefix}: run brief IDs do not match exact cohort")
    if summary.repeat != cohort.repeat:
        problems.append(f"{prefix}: repetition count differs from registration")
    if summary.split_checksums != cohort.split_checksums:
        problems.append(f"{prefix}: frozen benchmark identity differs from registration")
    if summary.length_tolerance != cohort.length_tolerance:
        problems.append(f"{prefix}: length tolerance differs from registration")
    if source.condition not in {condition.name for condition in summary.conditions}:
        problems.append(f"{prefix}: condition not declared by run")
    match = summary.equal_budget
    if match is None:
        problems.append(f"{prefix}: missing equal-budget registration")
    elif (
        set(match.conditions) != set(cohort.conditions)
        or match.tolerance != cohort.budget_tolerance
    ):
        problems.append(f"{prefix}: equal-budget settings differ from registration")


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes run directory: {relative}")
    return path


@dataclass(frozen=True)
class RunEvidence:
    source: Source
    records: list[StoryRecord]
    calls: list[CallRecord]
    manifest: Manifest


def _story(
    evidence: RunEvidence,
    record: StoryRecord,
    brief: Brief,
    cohort: GateCohort,
    problems: list[str],
) -> AdmittedStory | None:
    source, calls = evidence.source, evidence.calls
    prefix = source.condition
    if record.status != "ok":
        problems.append(f"{prefix}: story failed: {record.error or 'no diagnostic'}")
        return None
    relative = f"{record.path}/{STORY_FILE}"
    try:
        data = _safe_file(source.run_dir, relative).read_bytes()
        text = data.decode("utf-8").replace("\r\n", "\n")
        raw = _safe_file(source.run_dir, f"{record.path}/story.raw.md").read_text(encoding="utf-8")
        if normalise(raw) != text:
            problems.append(f"{prefix}: final story differs from selected raw output")
        if record.sha256 is not None and checksum_lf(data) != record.sha256:
            problems.append(f"{prefix}: final story checksum changed")
    except (OSError, ValueError) as error:
        problems.append(f"{prefix}: missing or unreadable final artifact: {error}")
        return None
    words = word_count(text)
    if not in_range(words, brief.target_words, cohort.length_tolerance):
        problems.append(
            f"{prefix}: final length {words} outside registered target {brief.target_words}"
        )
    if words != record.words or record.target_words != brief.target_words or not record.in_range:
        problems.append(f"{prefix}: saved length evidence disagrees with final content or brief")
    budget = story_budget(calls, record.condition, record.brief, record.repetition)
    if not budget.calls or budget != record.budget:
        problems.append(f"{prefix}: attributable call budget missing or differs from saved story")
    if any(call.input_tokens < 0 or call.output_tokens < 0 for call in calls):
        problems.append(f"{prefix}: invalid negative token usage")
    selected = [
        call
        for call in calls
        if (call.condition, call.brief, call.repetition)
        == (record.condition, record.brief, record.repetition)
    ]
    return AdmittedStory(
        condition=record.condition,
        run=str(source.run_dir.resolve()),
        path=relative,
        sha256=checksum_lf(data),
        words=words,
        budget=budget,
        new_spend_usd=sum(call.cost_usd for call in selected),
    )


def _load_runs(
    cohort: GateCohort, sources: tuple[Source, Source], report: ReadinessReport
) -> list[RunEvidence]:
    expected = {
        (brief, repetition)
        for brief in cohort.brief_checksums
        for repetition in range(cohort.repeat)
    }
    loaded: list[RunEvidence] = []
    for source in sources:
        try:
            manifest = read_manifest(source.run_dir)
            records, calls = read_stories(source.run_dir), read_calls(source.run_dir)
            _manifest_problems(cohort, source, manifest, report.problems)
            mine = [record for record in records if record.condition == source.condition]
            identities = Counter((record.brief, record.repetition) for record in mine)
            for identity, count in sorted(identities.items()):
                if identity not in expected or count != 1:
                    report.problems.append(
                        f"{source.condition}: extra or duplicate identity {identity} "
                        f"(count {count})"
                    )
            for call in calls:
                if (
                    call.condition == source.condition
                    and (call.brief, call.repetition) not in expected
                ):
                    report.problems.append(
                        f"{source.condition}: call outside expected cohort {call.call}"
                    )
            report.source_checksums[str(source.run_dir.resolve())] = {
                name: checksum_lf((source.run_dir / name).read_bytes())
                for name in ("manifest.json", "stories.jsonl", "calls.jsonl")
            }
            loaded.append(RunEvidence(source, mine, calls, manifest))
        except (OSError, ValueError) as error:
            report.problems.append(f"{source.condition}: malformed run evidence: {error}")
            report.status = "malformed"
    return loaded


def _pair(
    pair: PairReadiness, loaded: list[RunEvidence], available: dict[str, Brief], cohort: GateCohort
) -> None:
    brief_id, repetition = pair.brief, pair.repetition
    for evidence in loaded:
        source, records, manifest = evidence.source, evidence.records, evidence.manifest
        matches = [
            record
            for record in records
            if (record.brief, record.repetition) == (brief_id, repetition)
        ]
        if len(matches) != 1:
            pair.problems.append(
                f"{source.condition}: expected exactly one story, found {len(matches)}"
            )
            continue
        if brief_id in available:
            story = _story(evidence, matches[0], available[brief_id], cohort, pair.problems)
            if story is not None:
                pair.stories.append(story)
        summary = manifest.stories
        if summary is not None and summary.equal_budget is not None:
            budgets = [
                match
                for match in summary.equal_budget.matches
                if (match.brief, match.repetition) == (brief_id, repetition)
            ]
            if len(budgets) != 1 or not budgets[0].matched:
                pair.problems.append(
                    f"{source.condition}: missing or unmatched recorded budget comparison"
                )
            elif budgets[0].after.get(source.condition) != matches[0].budget.tokens:
                pair.problems.append(
                    f"{source.condition}: recorded budget comparison differs from calls"
                )
    if len(pair.stories) != len(cohort.conditions):
        pair.problems.append("incomplete comparison: both final stories are required")
    elif not within(
        pair.stories[0].budget.tokens, pair.stories[1].budget.tokens, cohort.budget_tolerance
    ):
        pair.problems.append("attributable generation-token budgets exceed registered tolerance")


def check_readiness(
    cohort: GateCohort, benchmark: Path, sources: tuple[Source, Source]
) -> ReadinessReport:
    """Check every registered pair, including artifacts and all attributable calls."""
    report = ReadinessReport(
        status="incomplete",
        cohort=cohort,
        cohort_sha256=cohort_checksum(cohort),
        benchmark=str(benchmark.resolve()),
        sources=[
            {"run": str(source.run_dir.resolve()), "condition": source.condition}
            for source in sources
        ],
    )
    available = _benchmark(cohort, benchmark, report.problems)
    if tuple(source.condition for source in sources) != cohort.conditions:
        report.problems.append("sources must follow registered condition order")
    expected = {
        (brief, repetition)
        for brief in cohort.brief_checksums
        for repetition in range(cohort.repeat)
    }
    loaded = _load_runs(cohort, sources, report)
    for brief_id, repetition in sorted(expected):
        pair = PairReadiness(brief=brief_id, repetition=repetition)
        report.pairs.append(pair)
        _pair(pair, loaded, available, cohort)
    if not report.problems and all(not pair.problems for pair in report.pairs):
        report.status = "ready"
    return report


def write_readiness(report: ReadinessReport, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "readiness.json", report.model_dump_json(indent=2) + "\n")
    lines = [
        f"# Gate readiness: {report.status}",
        f"Cohort: {report.cohort.id if report.cohort else '(malformed)'}",
    ]
    lines.extend(f"- {problem}" for problem in report.problems)
    for pair in report.pairs:
        lines.append(f"## {pair.brief}, repetition {pair.repetition}")
        lines.extend(f"- {problem}" for problem in pair.problems)
        lines.extend(
            f"- {story.condition}: {story.words} words; {story.budget.tokens} generation tokens; "
            f"SHA-256 {story.sha256}"
            for story in pair.stories
        )
    write_text_lf(out / "readiness.md", "\n\n".join(lines) + "\n")


@dataclass(frozen=True)
class GateBatchSettings:
    batches_dir: Path
    name: str
    scoring: Scoring
    seed: int | None = None


def build_gate_batch(
    cohort: GateCohort, benchmark: Path, sources: tuple[Source, Source], settings: GateBatchSettings
) -> Path:
    """Strict admission before invoking the ordinary sealed batch builder."""
    report = check_readiness(cohort, benchmark, sources)
    if report.status != "ready":
        raise BatchError("gate cohort is not ready; inspect a readiness report")
    path = build_batch(
        settings.batches_dir, settings.name, sources, scoring=settings.scoring, seed=settings.seed
    )
    write_readiness(report, path / "sealed" / "readiness")
    return path
