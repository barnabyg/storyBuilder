"""Whole-story planning followed by scene-channel completion on the released schema."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from nde.gitinfo import git
from nde.llm import structured
from nde.llm.gateway import Prompt
from nde.narrative.interpretation import iter_annotations
from nde.narrative.models import NarrativePlan, Strict
from nde.narrative.state import Constraint
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import rendered_files, schema_json
from nde.runs.config import PromptRef
from nde.runs.records import StructuredMetrics
from nde.runs.state import Attribution, record_calls_since
from nde.stories.pipeline import Step, StepFailedError, StoryContext, TemplateNeed, derive_seed

SKELETON = PromptRef(template="plan_skeleton", version=1)
CHANNELS = PromptRef(template="plan_channels", version=1)


class NarrativePlanningParams(Strict):
    skeleton_prompt: PromptRef = Field(default_factory=lambda: SKELETON)
    channels_prompt: PromptRef = Field(default_factory=lambda: CHANNELS)
    constraints: list[Constraint] = Field(default_factory=list[Constraint])


def planning_templates(params: NarrativePlanningParams) -> list[TemplateNeed]:
    return [
        TemplateNeed(params.skeleton_prompt, frozenset({"brief", "constraints"})),
        TemplateNeed(params.channels_prompt, frozenset({"brief", "plan", "scene_id"})),
    ]


def generated_errors(plan: NarrativePlan, ctx: StoryContext) -> list[str]:
    errors = [
        f"{d.code} [{d.location}]: {d.message}"
        for d in validate_plan(plan).diagnostics
        if d.severity == "error"
    ]
    if plan.tier != "L":
        errors.append("generated plan must use tier L")
    if not ctx.brief.min_scenes <= len(plan.scenes) <= ctx.brief.max_scenes:
        errors.append(f"scene_count: expected {ctx.brief.min_scenes} to {ctx.brief.max_scenes}")
    evidence = [span for scene in plan.scenes for span in scene.evidence]
    evidence += [span for scene in plan.scenes for beat in scene.beats for span in beat.evidence]
    evidence += [span for _, item in iter_annotations(plan) for span in item.evidence]
    evidence += [span for item in plan.motif_echoes for span in item.evidence]
    if plan.sources or evidence:
        errors.append("generated_evidence: generated plans cannot assert source text or evidence")
    if any(not scene.summary.strip() or not scene.beats for scene in plan.scenes):
        errors.append("scene_function: every scene requires a summary and concrete beats")
    return errors


def request_plan[T: BaseModel](
    ctx: StoryContext,
    step: Step,
    model: type[T],
    check: Callable[[T], list[str]],
) -> T:
    template = ctx.templates[step.prompt.label]
    messages = template.render(step.variables)
    binding = ctx.roles[step.role]
    response_format = structured.response_format(model)
    if not ctx.gateway.supports_json_schema(binding):
        messages = structured.with_schema_instruction(messages, model)
        response_format = None
    attribution = Attribution(
        call=f"{ctx.condition}/{ctx.brief.id}/{step.name}",
        repetition=ctx.repetition,
        brief=ctx.brief.id,
        condition=ctx.condition,
        step=step.name,
    )
    metrics = ctx.state.structured.setdefault(binding.label, StructuredMetrics())
    first = len(ctx.gateway.history)
    attempts = 0
    valid = False
    value: T | None = None
    errors: list[str] = []
    try:
        for attempt in range(ctx.structured_config.max_repairs + 1):
            ctx.state.current_call = attribution.call
            response = ctx.gateway.complete(
                binding,
                Prompt(template.name, template.version, messages),
                seed=derive_seed(ctx.seed, "plan", step.name),
                response_format=response_format,
            )
            attempts += 1
            try:
                value = structured.parse(response.text, model)
            except ValidationError as error:
                errors = structured.describe_errors(error)
            else:
                errors = check(value)
            ctx.keep(f"planning/{step.name}/{attempt + 1}.txt", response.text)
            ctx.keep(f"planning/{step.name}/{attempt + 1}.diagnostics.json", json.dumps(errors))
            if not errors and value is not None:
                valid = True
                return value
            messages = (*messages, *structured.repair_messages(response.text, errors))
        raise StepFailedError(step.name, "; ".join(errors))
    finally:
        record_calls_since(ctx.gateway, ctx.state, first, attribution)
        if attempts:
            metrics.record(attempts=attempts, valid=valid)


def stable_errors(previous: NarrativePlan, current: NarrativePlan, scene_id: str) -> list[str]:
    errors: list[str] = []
    for name in ("id", "tier", "title", "characters", "sources", "constraints", "arcs"):
        if getattr(previous, name) != getattr(current, name):
            errors.append(f"stable_identity: cannot change skeleton {name}")
    if [scene.id for scene in previous.scenes] != [scene.id for scene in current.scenes]:
        errors.append("stable_identity: cannot change scene IDs or order")
    for before, after in zip(previous.scenes, current.scenes, strict=False):
        if before.id != scene_id and before != after:
            errors.append(f"stable_identity: cannot change scene {before.id}")
        if before.id == scene_id and (
            before.character_ids != after.character_ids
            or [beat.id for beat in before.beats] != [beat.id for beat in after.beats]
        ):
            errors.append(f"stable_identity: cannot change character/beat IDs in {scene_id}")
    excluded = {"id", "schema_version", "tier", "title", "characters", "sources", "scenes"}
    for name in NarrativePlan.model_fields.keys() - excluded:
        earlier: list[Any] = getattr(previous, name)
        later: list[Any] = getattr(current, name)
        if later[: len(earlier)] != earlier:
            errors.append(f"accepted_channel: cannot rewrite accepted {name}")
    return errors


def _keep_plan(ctx: StoryContext, plan: NarrativePlan) -> None:
    _, files = rendered_files(plan)
    for name, content in files.items():
        ctx.keep(name, content)


def generate_plan(ctx: StoryContext, params: NarrativePlanningParams) -> NarrativePlan:
    """Generate a validated L plan; retain every attempt, including failed calls."""
    first = len(ctx.state.calls)
    status = "failed"
    selected: NarrativePlan | None = None
    ending = Constraint(
        id="brief-ending",
        kind="literary",
        key="ending",
        note=ctx.brief.ending_constraint.detail,
    )
    constraints = [ending, *params.constraints]

    def check_skeleton(plan: NarrativePlan) -> list[str]:
        # Caller requirements are authoritative, including when the provider omits them.
        plan.constraints = constraints
        errors = generated_errors(plan, ctx)
        if not plan.arcs:
            errors.append("skeleton_arc: include the whole-story character arc")
        # Natural-language ending requirements are literary checks, never inferred hard facts.
        return errors

    try:
        selected = request_plan(
            ctx,
            Step(
                "skeleton",
                "planner",
                params.skeleton_prompt,
                {
                    "brief": ctx.brief.model_dump_json(),
                    "constraints": json.dumps([c.model_dump(mode="json") for c in constraints]),
                },
            ),
            NarrativePlan,
            check_skeleton,
        )
        ctx.keep("skeleton.json", selected.model_dump_json(indent=2) + "\n")
        for scene in selected.scenes:
            previous = selected
            selected = request_plan(
                ctx,
                Step(
                    f"channels-{scene.id}",
                    "planner",
                    params.channels_prompt,
                    {
                        "brief": ctx.brief.model_dump_json(),
                        "plan": previous.model_dump_json(),
                        "scene_id": scene.id,
                    },
                ),
                NarrativePlan,
                lambda plan, previous=previous, scene_id=scene.id: (
                    generated_errors(plan, ctx) + stable_errors(previous, plan, scene_id)
                ),
            )
        _keep_plan(ctx, selected)
        status = "succeeded"
        return selected
    finally:
        calls = ctx.state.calls[first:]
        provenance = {
            "version": "1",
            "status": status,
            "brief": ctx.brief.model_dump(mode="json"),
            "git_sha": git(["rev-parse", "HEAD"], Path(__file__).parent) or "unknown",
            "brief_sha256": hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
            "params": params.model_dump(mode="json"),
            "prompts": {
                ref.label: ctx.templates[ref.label].sha256
                for ref in (
                    params.skeleton_prompt,
                    params.channels_prompt,
                )
            },
            "model": ctx.roles["planner"].model_dump(mode="json"),
            "seed": ctx.seed,
            "max_repairs": ctx.structured_config.max_repairs,
            "plan_sha256": None
            if status != "succeeded"
            else hashlib.sha256(ctx.files["plan.json"].encode()).hexdigest(),
            "calls": [call.model_dump(mode="json") for call in calls],
        }
        ctx.keep("planning-provenance.json", json.dumps(provenance, indent=2) + "\n")
