"""Candidate D uses coarse planning and the established C/B prose contract."""

from collections.abc import Mapping
from typing import ClassVar, Self, override

from pydantic import JsonValue

from nde.narrative.models import NarrativePlan
from nde.plan.coarse import CoarseParams, coarse_templates, generate_coarse_plan
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES
from nde.runs.config import RoleName
from nde.stories.baseline_c import BaselineC, BaselineCParams
from nde.stories.pipeline import Pipeline, StoryContext, TemplateNeed


class BaselineD(Pipeline[NarrativePlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")

    def __init__(self, params: CoarseParams) -> None:
        self.params = params
        self.prose = BaselineC(BaselineCParams())

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(CoarseParams.model_validate(raw))

    @override
    def templates(self) -> list[TemplateNeed]:
        return [*coarse_templates(self.params), TemplateNeed(SCENE_PROSE, SCENE_VARIABLES)]

    @override
    def plan(self, ctx: StoryContext) -> NarrativePlan:
        return generate_coarse_plan(ctx, self.params)

    @override
    def write(self, ctx: StoryContext, plan: NarrativePlan, *, seed: int) -> str:
        return self.prose.write(ctx, plan, seed=seed)
