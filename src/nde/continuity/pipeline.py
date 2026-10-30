"""The continuity pipeline: sheets first, then prose, checks and a saved inspection record."""

import json
from collections.abc import Mapping, Sequence
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
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict
from nde.prose.scenes import SCENE_VARIABLES, scene_step, write_scenes
from nde.runs.config import PromptRef, RoleName
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
QUALITY_NOTE = (
    "Automated checks are diagnostics that guide repair. Only the owner's reading "
    "establishes story quality."
)


class ContinuityParams(Strict):
    plan_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="continuity_plan", version=1)
    )


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


def _summary(plan: ContinuityPlan, story: NumberedStory, flags: Sequence[Flag], cost: float) -> str:
    lines = [
        f"# {plan.title}",
        "",
        f"Words: {word_count(story.text)}. Cost: US${cost:.4f}.",
        "",
        QUALITY_NOTE,
        "",
        "## Characters",
        "",
        *(f"- {sheet.name}, enters in scene {sheet.enters_scene}" for sheet in plan.characters),
        "",
        "## Flagged problems",
        "",
    ]
    if not flags:
        lines.append("None.")
    for flag in flags:
        lines += [f"- **{flag.problem_type}:** {flag.reason}", f"  > {flag.passage}"]
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
        numbered = NumberedStory.from_scenes(story, scene_texts)
        flags = [
            flag
            for audit in AUDITS
            for flag in run_audit(ctx, audit, plan, numbered, round_number=0)
        ]
        cost = _cost(ctx)
        ctx.keep(
            RESULT_FILE,
            _dumps(
                {
                    "brief": ctx.brief.id,
                    "title": plan.title,
                    "story": "story.md",
                    "character_sheets": "character-sheets.json",
                    "flagged_problems": [flag.model_dump(mode="json") for flag in flags],
                    "cost": cost,
                    "note": QUALITY_NOTE,
                }
            ),
        )
        ctx.keep(SUMMARY_FILE, _summary(plan, numbered, flags, cost["usd"]))
        return story
