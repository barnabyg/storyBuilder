"""Candidate D's coarse intentions, conditioned skeleton and scene completion."""

import hashlib
import json
from pathlib import Path

from pydantic import Field

from nde.corpus.exemplar_records import check_exemplars
from nde.gitinfo import git
from nde.narrative.interpretation import Arc, ReaderAffect
from nde.narrative.models import Character, NarrativePlan, Scene, Strict
from nde.narrative.state import Constraint
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import rendered_files, schema_json
from nde.plan.narrative import (
    NarrativePlanningParams,
    generated_errors,
    prepare_exemplars,
    request_plan,
    stable_errors,
)
from nde.runs.config import PromptRef
from nde.stories.pipeline import Step, StoryContext, TemplateNeed

INTENTIONS = PromptRef(template="plan_intentions", version=1)
D_SKELETON = PromptRef(template="plan_d_skeleton", version=1)


class CoarseParams(NarrativePlanningParams):
    intentions_prompt: PromptRef = Field(default_factory=lambda: INTENTIONS)
    skeleton_prompt: PromptRef = Field(default_factory=lambda: D_SKELETON)


class GlobalIntentions(Strict):
    id: str
    title: str
    characters: list[Character] = Field(min_length=1)
    scene_ids: list[str] = Field(min_length=3)
    arcs: list[Arc] = Field(min_length=1)
    reader_affect: list[ReaderAffect] = Field(min_length=1)

    def envelope(self, constraints: list[Constraint]) -> NarrativePlan:
        return NarrativePlan(
            id=self.id,
            title=self.title,
            characters=self.characters,
            scenes=[Scene(id=scene_id, title=scene_id) for scene_id in self.scene_ids],
            arcs=self.arcs,
            reader_affect=self.reader_affect,
            constraints=constraints,
        )


def coarse_templates(params: CoarseParams) -> list[TemplateNeed]:
    check_exemplars(params.exemplars)
    return [
        TemplateNeed(params.intentions_prompt, frozenset({"brief", "constraints"})),
        TemplateNeed(params.skeleton_prompt, frozenset({"brief", "constraints", "intentions"})),
        TemplateNeed(params.channels_prompt, frozenset({"brief", "plan", "scene_id"})),
    ]


def intention_errors(global_plan: GlobalIntentions, plan: NarrativePlan) -> list[str]:
    errors: list[str] = []
    expected = {
        "id": global_plan.id,
        "title": global_plan.title,
        "characters": global_plan.characters,
        "arcs": global_plan.arcs,
        "reader_affect": global_plan.reader_affect,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            errors.append(f"global_intention: cannot rewrite accepted {field}")
    if [scene.id for scene in plan.scenes] != global_plan.scene_ids:
        errors.append("global_intention: cannot change accepted scene IDs, order or count")
    return errors


def generate_coarse_plan(ctx: StoryContext, params: CoarseParams) -> NarrativePlan:
    """All requests use the same gateway/repair/accounting path as C."""
    prepare_exemplars(ctx, params.exemplars)
    first = len(ctx.state.calls)
    status = "failed"
    constraints = [
        Constraint(
            id="brief-ending",
            kind="literary",
            key="ending",
            note=ctx.brief.ending_constraint.detail,
        ),
        *params.constraints,
    ]
    serialized = json.dumps([item.model_dump(mode="json") for item in constraints])

    def check_global(value: GlobalIntentions) -> list[str]:
        plan = value.envelope(constraints)
        errors = [
            f"{d.code}: {d.message}"
            for d in validate_plan(plan).diagnostics
            if d.severity == "error"
        ]
        if not ctx.brief.min_scenes <= len(value.scene_ids) <= ctx.brief.max_scenes:
            errors.append("scene_count: global scene IDs must satisfy brief scene bounds")
        for item in [*value.arcs, *value.reader_affect]:
            if item.evidence:
                errors.append("generated_evidence: intentions cannot assert source evidence")
        if any(item.trajectory is None for item in value.reader_affect):
            errors.append("global_affect: every affect intention requires a coarse trajectory")
        return errors

    try:
        intentions = request_plan(
            ctx,
            Step(
                "intentions",
                "planner",
                params.intentions_prompt,
                {"brief": ctx.brief.model_dump_json(), "constraints": serialized},
            ),
            GlobalIntentions,
            check_global,
        )
        ctx.keep("intentions.json", intentions.model_dump_json(indent=2) + "\n")

        def check_skeleton(plan: NarrativePlan) -> list[str]:
            if plan.constraints != constraints:
                return ["hard_requirements: preserve all supplied constraints exactly"]
            return generated_errors(plan, ctx) + intention_errors(intentions, plan)

        selected = request_plan(
            ctx,
            Step(
                "skeleton",
                "planner",
                params.skeleton_prompt,
                {
                    "brief": ctx.brief.model_dump_json(),
                    "constraints": serialized,
                    "intentions": intentions.model_dump_json(),
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
                    generated_errors(plan, ctx)
                    + stable_errors(previous, plan, scene_id)
                    + intention_errors(intentions, plan)
                ),
            )
        _, files = rendered_files(selected)
        for name, content in files.items():
            ctx.keep(name, content)
        status = "succeeded"
        return selected
    finally:
        provenance = {
            "version": "1",
            "pipeline": "baseline_d",
            "status": status,
            "brief": ctx.brief.model_dump(mode="json"),
            "brief_sha256": hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
            "git_sha": git(["rev-parse", "HEAD"], Path(__file__).parent) or "unknown",
            "params": params.model_dump(mode="json"),
            "prompts": {
                need.prompt.label: ctx.templates[need.prompt.label].sha256
                for need in coarse_templates(params)
            },
            "model": ctx.roles["planner"].model_dump(mode="json"),
            "seed": ctx.seed,
            "max_repairs": ctx.structured_config.max_repairs,
            "plan_sha256": None
            if status != "succeeded"
            else hashlib.sha256(ctx.files["plan.json"].encode()).hexdigest(),
            "calls": [call.model_dump(mode="json") for call in ctx.state.calls[first:]],
        }
        ctx.keep("planning-provenance.json", json.dumps(provenance, indent=2) + "\n")
