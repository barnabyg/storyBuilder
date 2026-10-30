"""The continuity pipeline: sheets first, then prose, checks, repair and an inspection record."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, override

from pydantic import Field, JsonValue

from nde.continuity.audit import AUDIT_VARIABLES, Audit, run_audit
from nde.continuity.clock import ClockAudit, check_clock
from nde.continuity.flags import Flag
from nde.continuity.introductions import IntroductionAudit, check_introductions
from nde.continuity.knowledge import KnowledgeAudit, check_knowledge, knowledge_by_scene
from nde.continuity.plan import (
    PLAIN_STYLE_RULES,
    ContinuityPlan,
    continuity_plan_model,
    scene_specs,
)
from nde.continuity.repair import (
    REPAIR_PROMPT,
    REPAIR_VARIABLES,
    Change,
    group_by_passage,
    repair_passage,
)
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict
from nde.prose.scenes import SCENE_VARIABLES, scene_step, write_scenes
from nde.runs.config import PromptRef, RoleName
from nde.runs.records import STORY_FILE
from nde.stories.baseline_a import BRIEF_VARIABLES, brief_variables
from nde.stories.normalise import word_count
from nde.stories.pipeline import ATTEMPTS, Pipeline, Step, StoryContext, TemplateNeed

PLAN_STEP = "plan"
SCENE_PROMPT = PromptRef(template="continuity_scene", version=1)
AUDITS: tuple[Audit[Any], ...] = (
    Audit(
        "introductions",
        PromptRef(template="continuity_introductions", version=1),
        IntroductionAudit,
        check_introductions,
    ),
    Audit("clock", PromptRef(template="continuity_clock", version=1), ClockAudit, check_clock),
    Audit(
        "knowledge",
        PromptRef(template="continuity_knowledge", version=1),
        KnowledgeAudit,
        check_knowledge,
        record_file="knowledge.json",
        record=knowledge_by_scene,
    ),
)
# The record of the whole run for a person to inspect, and its readable summary.
RESULT_FILE = "continuity.json"
SUMMARY_FILE = "summary.md"
REPAIRS_FILE = "repairs.json"
# The story as first written, before any repair.
UNREPAIRED_FILE = "story.unrepaired.md"
QUALITY_NOTE = (
    "Automated checks are diagnostics that guide repair. Only the owner's reading "
    "establishes story quality."
)


class ContinuityParams(Strict):
    plan_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="continuity_plan", version=1)
    )
    # Rounds of repair after the first check; each repairs every flagged passage, then rechecks.
    repair_tries: int = Field(default=2, ge=0)


def _dumps(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _cost(ctx: StoryContext) -> dict[str, float]:
    """What this story has cost so far, every check and retry included."""
    mine = [
        call
        for call in ctx.state.calls
        if (call.condition, call.brief, call.repetition)
        == (ctx.condition, ctx.brief.id, ctx.repetition)
    ]
    return {
        "calls": len(mine),
        "input_tokens": sum(call.input_tokens for call in mine),
        "output_tokens": sum(call.output_tokens for call in mine),
        "usd": sum(call.cost_usd for call in mine),
        "list_price_usd": sum(call.token_cost_usd for call in mine),
    }


def _flag_lines(flags: list[Flag]) -> list[str]:
    if not flags:
        return ["None."]
    lines: list[str] = []
    for flag in flags:
        lines += [f"- **{flag.problem_type}:** {flag.reason}", f"  > {flag.passage}"]
    return lines


def _change_lines(changes: list[Change]) -> list[str]:
    if not changes:
        return ["None."]
    lines: list[str] = []
    for change in changes:
        why = "; ".join(flag.reason for flag in change.problems)
        lines += [
            f"- **Try {change.attempt}, paragraph {change.paragraph}:** {why}",
            f"  - Before: {change.before}",
            f"  - After: {change.after}",
        ]
    return lines


def _dump_flags(flags: list[Flag]) -> list[dict[str, Any]]:
    return [flag.model_dump(mode="json") for flag in flags]


@dataclass
class _Progress:
    """Where the checks and repairs have got to, kept on file after every step."""

    plan: ContinuityPlan
    story: NumberedStory
    # The flags each check round found, the first round first.
    rounds: list[list[Flag]] = field(default_factory=list[list[Flag]])
    changes: list[Change] = field(default_factory=list[Change])
    finished: bool = False

    def keep(self, ctx: StoryContext) -> None:
        """Save everything, so that a budget stop part-way still shows what happened."""
        ctx.files[STORY_FILE] = self.story.text
        cost = _cost(ctx)
        found = self.rounds[0] if self.rounds else []
        unresolved = self.rounds[-1] if self.rounds else []
        ctx.keep(REPAIRS_FILE, _dumps([change.model_dump(mode="json") for change in self.changes]))
        ctx.keep(
            RESULT_FILE,
            _dumps(
                {
                    "brief": ctx.brief.id,
                    "title": self.plan.title,
                    "story": STORY_FILE,
                    "character_sheets": "character-sheets.json",
                    "finished": self.finished,
                    "flagged_problems": _dump_flags(found),
                    "flags_by_round": [_dump_flags(flags) for flags in self.rounds],
                    "repairs": REPAIRS_FILE,
                    "unresolved_problems": _dump_flags(unresolved),
                    "cost": cost,
                    "note": QUALITY_NOTE,
                }
            ),
        )
        ctx.keep(SUMMARY_FILE, self._summary(cost["usd"], found, unresolved))

    def _summary(self, cost: float, found: list[Flag], unresolved: list[Flag]) -> str:
        lines = [
            f"# {self.plan.title}",
            "",
            f"Words: {word_count(self.story.text)}. Cost: US${cost:.4f}.",
            "",
            QUALITY_NOTE,
            "",
        ]
        if not self.finished:
            lines += ["**This run stopped before the checks and repairs finished.**", ""]
        lines += [
            "## Characters",
            "",
            *(
                f"- {sheet.name}, enters in scene {sheet.enters_scene}"
                for sheet in self.plan.characters
            ),
            "",
            "## Flagged problems, first check",
            "",
            *_flag_lines(found),
            "",
            "## Repairs",
            "",
            *_change_lines(self.changes),
            "",
            "## Unresolved problems, last check",
            "",
            *_flag_lines(unresolved),
        ]
        return "\n".join(lines) + "\n"


class Continuity(Pipeline[ContinuityPlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose", "critic")
    dev_only: ClassVar[bool] = True

    def __init__(self, params: ContinuityParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(ContinuityParams.model_validate(raw))

    @override
    def templates(self) -> list[TemplateNeed]:
        return [
            TemplateNeed(self.params.plan_prompt, BRIEF_VARIABLES),
            TemplateNeed(SCENE_PROMPT, SCENE_VARIABLES),
            *(TemplateNeed(audit.prompt, AUDIT_VARIABLES) for audit in AUDITS),
            TemplateNeed(REPAIR_PROMPT, REPAIR_VARIABLES),
        ]

    @override
    def plan(self, ctx: StoryContext) -> ContinuityPlan:
        brief = ctx.brief
        step = Step(PLAN_STEP, "planner", self.params.plan_prompt, brief_variables(brief))
        plan = ctx.structured(
            step, continuity_plan_model(brief.min_scenes, brief.max_scenes), seed=ctx.seed
        )
        dumped = plan.model_dump(mode="json")
        ctx.keep("plan.json", _dumps(dumped))
        ctx.keep("character-sheets.json", _dumps(dumped["characters"]))
        ctx.keep("style-rules.md", PLAIN_STYLE_RULES)
        return plan

    @override
    def write(self, ctx: StoryContext, plan: ContinuityPlan, *, seed: int) -> str:
        return write_scenes(ctx, scene_specs(plan), seed=seed, prompt=SCENE_PROMPT)

    @override
    def finish(self, ctx: StoryContext, plan: ContinuityPlan, story: str, *, attempt: int) -> str:
        scene_texts = [
            ctx.files[f"{ATTEMPTS}/{attempt + 1}/{scene_step(index)}.md"]
            for index in range(len(plan.scenes))
        ]
        ctx.keep(UNREPAIRED_FILE, story)
        progress = _Progress(plan, NumberedStory.from_scenes(story, scene_texts))
        progress.keep(ctx)
        self._check(ctx, progress, 0)
        for number in range(1, self.params.repair_tries + 1):
            if not progress.rounds[-1] or not self._repair(ctx, progress, number):
                break
            self._check(ctx, progress, number)
        progress.finished = True
        progress.keep(ctx)
        return progress.story.text

    def _check(self, ctx: StoryContext, progress: _Progress, round_number: int) -> None:
        progress.rounds.append(
            [
                flag
                for audit in AUDITS
                for flag in run_audit(
                    ctx, audit, progress.plan, progress.story, round_number=round_number
                )
            ]
        )
        progress.keep(ctx)

    def _repair(self, ctx: StoryContext, progress: _Progress, attempt: int) -> bool:
        """One try at every flagged passage; returns whether any text changed."""
        changed = False
        for flags in group_by_passage(progress.rounds[-1]):
            repaired = repair_passage(ctx, progress.plan, progress.story, flags, attempt=attempt)
            if repaired is None:
                continue
            progress.story, change = repaired
            progress.changes.append(change)
            changed = changed or change.before != change.after
            progress.keep(ctx)
        return changed
