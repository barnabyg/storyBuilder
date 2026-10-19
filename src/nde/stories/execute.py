"""Prepare and execute the `stories` section of a config: one story per brief per condition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nde.briefs import BriefError, load_briefs, split_checksum
from nde.files import write_text_lf
from nde.llm.gateway import Gateway
from nde.runs.config import (
    BriefSelection,
    ConditionSpec,
    ConfigError,
    LengthConfig,
    LoadedConfig,
    RoleName,
)
from nde.runs.prompts import PromptTemplate, load_template
from nde.runs.records import (
    STORY_DIR,
    ConditionUse,
    LengthAttempt,
    StoriesSummary,
    StoryRecord,
)
from nde.runs.state import RunState
from nde.schema.brief import SPLITS, Brief
from nde.stories import PIPELINES
from nde.stories.length import Draft, attempt_seed, choose, measure
from nde.stories.pipeline import Pipeline, StepFailedError, StoryContext

# The normalised story, which blind batches and judges see, and the model's raw text.
STORY_FILE = "story.md"
RAW_FILE = "story.raw.md"


@dataclass(frozen=True)
class Condition:
    spec: ConditionSpec
    pipeline: Pipeline[Any]


@dataclass(frozen=True)
class PreparedStories:
    briefs_dir: Path
    briefs: list[Brief]
    conditions: list[Condition]
    repeat: int
    length: LengthConfig
    split_checksums: dict[str, str]

    @property
    def roles(self) -> list[RoleName]:
        """Every role a condition uses, in condition order."""
        roles: dict[RoleName, None] = {}
        for condition in self.conditions:
            roles.update(dict.fromkeys(condition.pipeline.roles))
        return list(roles)

    def summary(self) -> StoriesSummary:
        return StoriesSummary(
            briefs_dir=str(self.briefs_dir),
            brief_ids=[brief.id for brief in self.briefs],
            split_checksums=self.split_checksums,
            conditions=[
                ConditionUse(name=c.spec.name, pipeline=c.spec.pipeline, params=c.spec.params)
                for c in self.conditions
            ],
            repeat=self.repeat,
            length_tolerance=self.length.tolerance,
            max_attempts=self.length.max_attempts,
        )


def _format(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or '<params>'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )


def _select_briefs(selection: BriefSelection, briefs_dir: Path) -> list[Brief]:
    try:
        available = load_briefs(briefs_dir)
    except BriefError as error:
        raise ConfigError(str(error)) from error
    if selection.split is not None:
        in_split = [brief for brief in available.values() if brief.split == selection.split]
        available = {brief.id: brief for brief in sorted(in_split, key=lambda b: b.id)}
    if selection.ids is None:
        if not available:
            raise ConfigError(f"no briefs found in the {selection.split} split under {briefs_dir}")
        return list(available.values())
    where = f" in the {selection.split} split" if selection.split else ""
    for brief_id in selection.ids:
        if brief_id not in available:
            raise ConfigError(f"unknown brief {brief_id!r}{where} under {briefs_dir}")
    return [available[brief_id] for brief_id in selection.ids]


def _condition(loaded: LoadedConfig, spec: ConditionSpec) -> Condition:
    factory = PIPELINES.get(spec.pipeline)
    if factory is None:
        known = ", ".join(sorted(PIPELINES))
        raise ConfigError(
            f"condition {spec.name!r}: unknown pipeline {spec.pipeline!r}; known pipelines: {known}"
        )
    try:
        pipeline = factory(spec.params)
    except ValidationError as error:
        raise ConfigError(f"condition {spec.name!r}: invalid params: {_format(error)}") from error
    for role in pipeline.roles:
        if role not in loaded.config.roles:
            raise ConfigError(
                f"condition {spec.name!r} uses role {role!r}, which has no model in `roles`"
            )
    return Condition(spec, pipeline)


def prepare_stories(
    loaded: LoadedConfig, templates: dict[str, PromptTemplate]
) -> PreparedStories | None:
    """Resolve briefs, pipelines, roles and templates; raises ConfigError before any call."""
    stories = loaded.config.stories
    briefs_dir = loaded.briefs_dir
    if stories is None or briefs_dir is None:
        return None
    briefs = _select_briefs(stories.briefs, briefs_dir)
    conditions = [_condition(loaded, spec) for spec in stories.conditions]
    for condition in conditions:
        for need in condition.pipeline.templates():
            ref = need.prompt
            if ref.label not in templates:
                templates[ref.label] = load_template(loaded.prompts_dir, ref.template, ref.version)
            templates[ref.label].render(dict.fromkeys(need.variables, "x"))
    checksums: dict[str, str] = {}
    for split in SPLITS:
        checksum = split_checksum(briefs_dir, split)
        if checksum is not None and any(brief.split == split for brief in briefs):
            checksums[split] = checksum
    prepared = PreparedStories(
        briefs_dir, briefs, conditions, stories.repeat, stories.length, checksums
    )
    config = loaded.config
    if not config.budget.allow_unpriced_models and config.unpriced_models(prepared.roles):
        raise ConfigError("; ".join(config.unpriced_warnings(prepared.roles)))
    return prepared


@dataclass(frozen=True)
class StoryRun:
    """Everything needed to write the stories of a prepared config."""

    gateway: Gateway
    loaded: LoadedConfig
    templates: dict[str, PromptTemplate]
    prepared: PreparedStories
    run_dir: Path
    state: RunState


def story_path(condition: str, brief: str, repetition: int) -> Path:
    return Path(STORY_DIR) / condition / brief / f"r{repetition + 1}"


def _write_files(run_dir: Path, folder: Path, files: dict[str, str]) -> list[str]:
    for name, content in files.items():
        path = run_dir / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_lf(path, content)
    return sorted(files)


def _draft(run: StoryRun, ctx: StoryContext, condition: Condition, plan: object) -> list[Draft]:
    """Write the prose until a draft is in the length range or the attempts run out."""
    length = run.prepared.length
    target = ctx.brief.target_words
    drafts: list[Draft] = []
    for attempt in range(length.max_attempts):
        ctx.attempt = attempt
        seed = attempt_seed(ctx.seed, attempt)
        raw = condition.pipeline.write(ctx, plan, seed=seed)
        ctx.keep(RAW_FILE, raw)
        drafts.append(measure(raw, seed=seed, target=target, tolerance=length.tolerance))
        if drafts[-1].in_range:
            break
    ctx.attempt = None
    return drafts


def write_story(run: StoryRun, condition: Condition, brief: Brief, repetition: int) -> StoryRecord:
    config = run.loaded.config
    seed = config.seed + repetition
    folder = story_path(condition.spec.name, brief.id, repetition)
    ctx = StoryContext(
        gateway=run.gateway,
        state=run.state,
        templates=run.templates,
        roles=config.roles,
        structured_config=config.structured,
        brief=brief,
        condition=condition.spec.name,
        repetition=repetition,
        seed=seed,
    )
    record = StoryRecord(
        brief=brief.id,
        condition=condition.spec.name,
        pipeline=condition.spec.pipeline,
        repetition=repetition,
        seed=seed,
        status="ok",
        path=folder.as_posix(),
        target_words=brief.target_words,
    )
    try:
        drafts = _draft(run, ctx, condition, condition.pipeline.plan(ctx))
    except StepFailedError as failure:
        record.status = "failed"
        record.error = str(failure)
    else:
        chosen = choose(drafts, brief.target_words)
        ctx.files[STORY_FILE] = drafts[chosen].text
        ctx.files[RAW_FILE] = drafts[chosen].raw
        record.attempts = [
            LengthAttempt(seed=d.seed, words=d.words, in_range=d.in_range) for d in drafts
        ]
        record.chosen_attempt = chosen
        record.words = drafts[chosen].words
        record.in_range = drafts[chosen].in_range
    record.files = _write_files(run.run_dir, folder, ctx.files)
    return record


def run_stories(run: StoryRun) -> None:
    """Write every story: for each brief and repetition, every condition in turn."""
    for brief in run.prepared.briefs:
        for repetition in range(run.prepared.repeat):
            # Conditions take turns, so a budget stop leaves comparable stories.
            for condition in run.prepared.conditions:
                story = write_story(run, condition, brief, repetition)
                run.state.stories.append(story)
