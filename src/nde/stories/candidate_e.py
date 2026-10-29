"""Candidate E: a learned conditional plan through the established C/D prose contract.

E's structure comes from a local conditional checkpoint and its free text from
the hybrid completion workflow. Prose, the planned-state ledger, scene notes,
length control, equal-budget revisions and final-story selection are C's,
unchanged, so E differs from C and D only in how its plan is made.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Self, override

from pydantic import JsonValue, ValidationError, model_validator

from nde.conditional.completion import MAX_REPAIRS as COMPLETION_MAX_REPAIRS
from nde.llm.config import ModelBinding
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError
from nde.plan.conditional import ConditionalPlanningParams, checked_checkpoint, generate_e_plan
from nde.plan.imported_e import import_e_plan
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES
from nde.runs.config import ConfigError, RoleName, StructuredConfig
from nde.stories.baseline_c import BaselineC, BaselineCParams
from nde.stories.pipeline import Pipeline, StepFailedError, StoryContext, TemplateNeed


class CandidateEParams(ConditionalPlanningParams):
    saved_plan: str | None = None
    provenance: str | None = None

    @model_validator(mode="after")
    def _attributed_import(self) -> Self:
        if (self.saved_plan is None) != (self.provenance is None):
            raise ValueError("saved_plan and provenance must be supplied together")
        return self

    def planning(self) -> ConditionalPlanningParams:
        return ConditionalPlanningParams.model_validate(
            self.model_dump(exclude={"saved_plan", "provenance"})
        )


class CandidateE(Pipeline[NarrativePlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")
    local_model: ClassVar[bool] = True

    def __init__(self, params: CandidateEParams) -> None:
        self.params = params
        self.prose = BaselineC(BaselineCParams())

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(CandidateEParams.model_validate(raw))

    @override
    def check_roles(self, roles: Mapping[RoleName, ModelBinding]) -> None:
        super().check_roles(roles)
        params = roles["planner"].params
        if params.get("stream") or params.get("n", 1) != 1:
            raise ConfigError("E's text completion needs a single non-streaming planner response")

    @override
    def check_structured(self, structured: StructuredConfig) -> None:
        if structured.max_repairs > COMPLETION_MAX_REPAIRS:
            raise ConfigError(
                f"structured.max_repairs is {structured.max_repairs}, but E's text completion "
                f"allows at most {COMPLETION_MAX_REPAIRS} repairs"
            )

    @override
    def check_inputs(self) -> None:
        try:
            checked_checkpoint(self.params.planning())
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as error:
            raise ConfigError(f"conditional checkpoint cannot serve E: {error}") from error

    @override
    def templates(self) -> list[TemplateNeed]:
        # Planning uses the conditional workflows' own versioned contracts, not templates.
        return [TemplateNeed(SCENE_PROSE, SCENE_VARIABLES)]

    @override
    def plan(self, ctx: StoryContext) -> NarrativePlan:
        if self.params.saved_plan is None or self.params.provenance is None:
            return generate_e_plan(ctx, self.params.planning())
        try:
            return import_e_plan(
                ctx,
                Path(self.params.saved_plan),
                Path(self.params.provenance),
                self.params.planning(),
            )
        except (PlanError, OSError, ValueError, KeyError, TypeError) as failure:
            raise StepFailedError("saved-E-plan", str(failure)) from failure

    @override
    def write(self, ctx: StoryContext, plan: NarrativePlan, *, seed: int) -> str:
        return self.prose.write(ctx, plan, seed=seed)
