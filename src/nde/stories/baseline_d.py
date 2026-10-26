"""Candidate D uses coarse planning and the established C/B prose contract."""

from collections.abc import Mapping
from typing import ClassVar, Self, override

from pydantic import JsonValue

from nde.llm.config import ModelBinding
from nde.narrative.models import NarrativePlan
from nde.plan.coarse import coarse_templates, generate_coarse_plan
from nde.plan.refinement import (
    RefinementParams,
    check_critic_families,
    refine_plan,
    refinement_templates,
)
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES
from nde.runs.config import RoleName
from nde.stories.baseline_c import BaselineC, BaselineCParams
from nde.stories.pipeline import Pipeline, StoryContext, TemplateNeed


class BaselineD(Pipeline[NarrativePlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")

    def __init__(self, params: RefinementParams) -> None:
        self.params = params
        self.prose = BaselineC(BaselineCParams())

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(RefinementParams.model_validate(raw))

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
        initial = generate_coarse_plan(ctx, self.params.coarse())
        return refine_plan(ctx, initial, self.params)

    @override
    def write(self, ctx: StoryContext, plan: NarrativePlan, *, seed: int) -> str:
        return self.prose.write(ctx, plan, seed=seed)
