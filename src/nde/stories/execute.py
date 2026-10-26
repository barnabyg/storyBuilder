"""Prepare and execute the `stories` section of a config: one story per brief per condition."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nde.briefs import BriefError, load_briefs, split_checksum
from nde.files import checksum_lf, write_text_lf
from nde.llm.errors import ProviderError
from nde.llm.gateway import Gateway
from nde.llm.ledger import BudgetExceededError
from nde.prose.revision import REVISION_VARIABLES, revise
from nde.runs.config import (
    BriefSelection,
    ConditionSpec,
    ConfigError,
    EqualBudgetConfig,
    LengthConfig,
    LoadedConfig,
    PromptRef,
    RoleName,
)
from nde.runs.prompts import PromptTemplate, require_template
from nde.runs.records import (
    STORY_DIR,
    STORY_FILE,
    BudgetMatch,
    ConditionUse,
    EqualBudgetSummary,
    LengthAttempt,
    RevisionRecord,
    StoriesSummary,
    StoryRecord,
)
from nde.runs.state import RunState
from nde.schema.brief import SPLITS, Brief
from nde.stories.budget import story_budget, within
from nde.stories.length import Draft, attempt_seed, choose, measure
from nde.stories.pipeline import (
    Pipeline,
    StepFailedError,
    StoryContext,
    TemplateNeed,
    derive_seed,
)
from nde.stories.registry import PIPELINES

# The normalised story, which blind batches and judges see, and the model's raw text.
RAW_FILE = "story.raw.md"
# The story before its first revision pass, and the folder of each pass's files.
ORIGINAL_FILE = "story.original.md"
REVISIONS = "revisions"


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
    equal_budget: EqualBudgetConfig | None
    split_checksums: dict[str, str]

    @property
    def roles(self) -> list[RoleName]:
        """Every role a condition uses, in condition order."""
        roles: dict[RoleName, None] = {}
        for condition in self.conditions:
            roles.update(dict.fromkeys(condition.pipeline.required_roles()))
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
            equal_budget=None
            if self.equal_budget is None
            else EqualBudgetSummary(
                conditions=self.equal_budget.conditions,
                tolerance=self.equal_budget.tolerance,
                max_revision_passes=self.equal_budget.max_revision_passes,
            ),
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
    for role in pipeline.required_roles():
        if role not in loaded.config.roles:
            raise ConfigError(
                f"condition {spec.name!r} uses role {role!r}, which has no model in `roles`"
            )
    pipeline.check_roles(loaded.config.roles)
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
    if any(condition.spec.params.get("exemplars") for condition in conditions) and any(
        brief.split != "dev" for brief in briefs
    ):
        raise ConfigError("corpus exemplars may be used only on dev briefs")
    needs = [need for condition in conditions for need in condition.pipeline.templates()]
    if stories.equal_budget is not None:
        needs.append(TemplateNeed(stories.equal_budget.revision_prompt, REVISION_VARIABLES))
    for need in needs:
        require_template(
            templates, loaded.prompts_dir, need.prompt, dict.fromkeys(need.variables, "x")
        )
    checksums: dict[str, str] = {}
    for split in SPLITS:
        checksum = split_checksum(briefs_dir, split)
        if checksum is not None and any(brief.split == split for brief in briefs):
            checksums[split] = checksum
    return PreparedStories(
        briefs_dir,
        briefs,
        conditions,
        stories.repeat,
        stories.length,
        stories.equal_budget,
        checksums,
    )


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


@dataclass
class StoryJob:
    """A story being written: its context (with its files) and its record."""

    ctx: StoryContext
    record: StoryRecord
    folder: Path

    @property
    def tokens(self) -> int:
        return self.record.budget.tokens


def _generate(
    run: StoryRun, ctx: StoryContext, write: Callable[[int], str], seed: int
) -> list[Draft]:
    """Generate prose until a draft is in the length range or the attempts run out."""
    length = run.prepared.length
    target = ctx.brief.target_words
    drafts: list[Draft] = []
    for attempt in range(length.max_attempts):
        ctx.attempt = attempt
        draft_seed = attempt_seed(seed, attempt)
        raw = write(draft_seed)
        ctx.keep(RAW_FILE, raw)
        drafts.append(measure(raw, seed=draft_seed, target=target, tolerance=length.tolerance))
        if drafts[-1].in_range:
            break
    ctx.attempt = None
    return drafts


def _use(job: StoryJob, drafts: list[Draft]) -> list[LengthAttempt]:
    """Make the chosen draft the story; returns the attempts as recorded."""
    chosen = choose(drafts, job.ctx.brief.target_words)
    job.ctx.files[STORY_FILE] = drafts[chosen].text
    job.ctx.files[RAW_FILE] = drafts[chosen].raw
    job.record.words = drafts[chosen].words
    job.record.in_range = drafts[chosen].in_range
    job.record.chosen_attempt = chosen
    return [LengthAttempt(seed=d.seed, words=d.words, in_range=d.in_range) for d in drafts]


def save(run: StoryRun, job: StoryJob) -> None:
    """Write the story's files and record its generation budget so far."""
    for name, content in job.ctx.files.items():
        path = run.run_dir / job.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_lf(path, content)
    record = job.record
    record.files = sorted(job.ctx.files)
    if STORY_FILE in job.ctx.files:
        record.sha256 = checksum_lf(job.ctx.files[STORY_FILE].encode("utf-8"))
    record.budget = story_budget(run.state.calls, record.condition, record.brief, record.repetition)


def write_story(run: StoryRun, condition: Condition, brief: Brief, repetition: int) -> StoryJob:
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
    job = StoryJob(ctx, record, folder)
    try:
        plan = condition.pipeline.plan(ctx)
        drafts = _generate(
            run, ctx, lambda draft_seed: condition.pipeline.write(ctx, plan, seed=draft_seed), seed
        )
    except StepFailedError as failure:
        record.status = "failed"
        record.error = str(failure)
    except (ProviderError, BudgetExceededError) as failure:
        record.status = "failed"
        record.error = str(failure)
        if ctx.files:
            save(run, job)
            run.state.stories.append(record)
        raise
    else:
        record.attempts = _use(job, drafts)
    save(run, job)
    return job


def revise_story(run: StoryRun, job: StoryJob, prompt: PromptRef) -> None:
    """One revision pass: rewrite the story, under normalisation and length control."""
    ctx, record = job.ctx, job.record
    number = record.revision_passes + 1
    current = ctx.files[STORY_FILE]
    if number == 1:
        ctx.files[ORIGINAL_FILE] = current
    ctx.prefix = f"{REVISIONS}/{number}/"
    try:
        drafts = _generate(
            run,
            ctx,
            lambda draft_seed: revise(ctx, current, number=number, seed=draft_seed, prompt=prompt),
            derive_seed(ctx.seed, "revision", number),
        )
    finally:
        ctx.prefix = ""
    attempts = _use(job, drafts)
    ctx.files[f"{REVISIONS}/{number}/{STORY_FILE}"] = ctx.files[STORY_FILE]
    record.revisions.append(RevisionRecord(attempts=attempts, chosen_attempt=record.chosen_attempt))
    record.revision_passes = number
    save(run, job)


def _passes(count: int) -> str:
    return f"{count} revision pass{'' if count == 1 else 'es'}"


def match_budgets(run: StoryRun, jobs: dict[str, StoryJob], spec: EqualBudgetConfig) -> BudgetMatch:
    """Give the cheaper story revision passes until the two budgets are within tolerance."""
    first, second = (jobs[name] for name in spec.conditions)
    match = BudgetMatch(
        brief=first.record.brief,
        repetition=first.record.repetition,
        before={name: jobs[name].tokens for name in spec.conditions},
        after={},
        matched=False,
    )
    failed = [name for name in spec.conditions if jobs[name].record.status != "ok"]
    cheaper, dearer = sorted((first, second), key=lambda job: job.tokens)
    if failed:
        match.reason = f"no story to match: {', '.join(failed)} failed"
    elif not within(cheaper.tokens, dearer.tokens, spec.tolerance):
        match.revised = cheaper.record.condition
        while match.passes < spec.max_revision_passes:
            try:
                revise_story(run, cheaper, spec.revision_prompt)
            except (ProviderError, BudgetExceededError) as error:
                # Preserve both the previous selected story and partial revision artifacts.
                save(run, cheaper)
                match.reason = f"revision interrupted: {error}"
                match.after = {name: jobs[name].tokens for name in spec.conditions}
                run.state.budget_matches.append(match)
                raise
            match.passes += 1
            if within(cheaper.tokens, dearer.tokens, spec.tolerance):
                break
            if cheaper.tokens > dearer.tokens:
                # Out of tolerance on the other side: another pass only widens the gap.
                match.reason = (
                    f"overshot after {_passes(match.passes)}: {match.revised} costs "
                    f"{cheaper.tokens} tokens against {dearer.tokens}"
                )
                break
    match.after = {name: jobs[name].tokens for name in spec.conditions}
    match.matched = not failed and within(cheaper.tokens, dearer.tokens, spec.tolerance)
    if not match.matched and match.reason is None:
        match.reason = f"not within {spec.tolerance:.0%} after {_passes(match.passes)}"
    return match


def run_stories(run: StoryRun) -> None:
    """Write every story: for each brief and repetition, every condition in turn.

    In equal-budget mode, each brief and repetition's two stories are then
    matched, and the matches are recorded in the manifest's stories summary.
    """
    spec = run.prepared.equal_budget
    for brief in run.prepared.briefs:
        for repetition in range(run.prepared.repeat):
            jobs: dict[str, StoryJob] = {}
            # Conditions take turns, so a budget stop leaves comparable stories.
            for condition in run.prepared.conditions:
                job = write_story(run, condition, brief, repetition)
                run.state.stories.append(job.record)
                jobs[condition.spec.name] = job
            if spec is not None:
                run.state.budget_matches.append(match_budgets(run, jobs, spec))
