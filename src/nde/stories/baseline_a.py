"""Baseline A: one prompt from brief to complete story, on the `prose` role."""

from collections.abc import Mapping
from typing import ClassVar, Self, override

from pydantic import Field, JsonValue

from nde.llm.config import Strict
from nde.runs.config import PromptRef, RoleName
from nde.schema.brief import Brief
from nde.stories.pipeline import Pipeline, Step, StoryContext, TemplateNeed

STEP = "story"


class BaselineAParams(Strict):
    prompt: PromptRef = Field(default_factory=lambda: PromptRef(template="story_a", version=1))


def brief_variables(brief: Brief) -> dict[str, str]:
    """The brief's fields as prompt variables: `brief`, `target_words` and the scene range."""
    return {
        "brief": brief.describe(),
        "target_words": str(brief.target_words),
        "min_scenes": str(brief.min_scenes),
        "max_scenes": str(brief.max_scenes),
    }


BRIEF_VARIABLES = frozenset({"brief", "target_words", "min_scenes", "max_scenes"})


class BaselineA(Pipeline[None]):
    roles: ClassVar[tuple[RoleName, ...]] = ("prose",)

    def __init__(self, params: BaselineAParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(BaselineAParams.model_validate(raw))

    @override
    def templates(self) -> list[TemplateNeed]:
        return [TemplateNeed(self.params.prompt, BRIEF_VARIABLES)]

    @override
    def plan(self, ctx: StoryContext) -> None:
        return None

    @override
    def write(self, ctx: StoryContext, plan: None, *, seed: int) -> str:
        step = Step(STEP, "prose", self.params.prompt, brief_variables(ctx.brief))
        return ctx.complete(step, seed=seed)
