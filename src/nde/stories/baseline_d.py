"""Candidate D uses coarse planning and the established C/B prose contract."""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Self, override

from pydantic import JsonValue, model_validator

from nde.llm.config import ModelBinding
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError, rendered_files
from nde.plan.coarse import coarse_templates, generate_coarse_plan
from nde.plan.imported_d import import_d_plan
from nde.plan.refinement import (
    RefinementParams,
    check_critic_families,
    refine_plan,
    refinement_templates,
)
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES
from nde.runs.config import RoleName
from nde.stories.baseline_c import BaselineC, BaselineCParams
from nde.stories.pipeline import Pipeline, StepFailedError, StoryContext, TemplateNeed


class BaselineDParams(RefinementParams):
    saved_plan: str | None = None
    provenance: str | None = None

    @model_validator(mode="after")
    def _attributed_import(self) -> Self:
        if (self.saved_plan is None) != (self.provenance is None):
            raise ValueError("saved_plan and provenance must be supplied together")
        if self.saved_plan is not None and self.exemplars is not None:
            raise ValueError("imported plans cannot acquire new exemplar conditioning")
        return self

    def refinement(self) -> RefinementParams:
        return RefinementParams.model_validate(
            self.model_dump(exclude={"saved_plan", "provenance"})
        )


class BaselineD(Pipeline[NarrativePlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")

    def __init__(self, params: BaselineDParams) -> None:
        self.params = params
        self.prose = BaselineC(BaselineCParams())

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(BaselineDParams.model_validate(raw))

    @override
    def required_roles(self) -> tuple[RoleName, ...]:
        return (*self.roles, "critic") if self.params.critic_iterations else self.roles

    @override
    def check_roles(self, roles: Mapping[RoleName, ModelBinding]) -> None:
        super().check_roles(roles)
        check_critic_families(self.params, roles)

    @override
    def templates(self) -> list[TemplateNeed]:
        return [
            *coarse_templates(self.params),
            *refinement_templates(self.params),
            TemplateNeed(SCENE_PROSE, SCENE_VARIABLES),
        ]

    @override
    def plan(self, ctx: StoryContext) -> NarrativePlan:
        if self.params.saved_plan is None or self.params.provenance is None:
            initial = generate_coarse_plan(ctx, self.params.coarse())
            return refine_plan(ctx, initial, self.params.refinement())
        try:
            selected = import_d_plan(
                ctx,
                Path(self.params.saved_plan),
                Path(self.params.provenance),
                self.params.refinement(),
            )
        except (PlanError, OSError, ValueError) as failure:
            raise StepFailedError("saved-D-plan", str(failure)) from failure
        _, files = rendered_files(selected)
        for name, content in files.items():
            ctx.keep(name, content)
        return selected

    @override
    def write(self, ctx: StoryContext, plan: NarrativePlan, *, seed: int) -> str:
        return self.prose.write(ctx, plan, seed=seed)
