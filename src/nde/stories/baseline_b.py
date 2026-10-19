"""Baseline B: a beat outline on the `planner` role, then prose scene by scene.

The prose goes through the shared prose contract (`nde.prose.scenes`), which
every later condition reuses unchanged. `plan` makes the outline and `write`
feeds its scenes, in order, to the shared prose steps. A condition that only
changes the scene list, such as B-scrambled, overrides `plan`.
"""

import json
from collections.abc import Mapping
from typing import ClassVar, Self, override

from pydantic import Field, JsonValue

from nde.llm.config import Strict
from nde.plan.beats import BeatOutline, outline_model
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES, write_scenes
from nde.runs.config import PromptRef, RoleName
from nde.stories.baseline_a import BRIEF_VARIABLES, brief_variables
from nde.stories.pipeline import Pipeline, Step, StoryContext, TemplateNeed

OUTLINE_STEP = "outline"
OUTLINE_FILE = "outline.json"


class BaselineBParams(Strict):
    outline_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="beat_outline", version=1)
    )


class BaselineB(Pipeline[BeatOutline]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")

    def __init__(self, params: BaselineBParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(BaselineBParams.model_validate(raw))

    @override
    def templates(self) -> list[TemplateNeed]:
        return [
            TemplateNeed(self.params.outline_prompt, BRIEF_VARIABLES),
            TemplateNeed(SCENE_PROSE, SCENE_VARIABLES),
        ]

    @override
    def plan(self, ctx: StoryContext) -> BeatOutline:
        """The beat outline. Its seed is the story's, so conditions sharing it share the outline."""
        brief = ctx.brief
        step = Step(OUTLINE_STEP, "planner", self.params.outline_prompt, brief_variables(brief))
        model = outline_model(brief.min_scenes, brief.max_scenes)
        outline = ctx.structured(step, model, seed=ctx.seed)
        ctx.keep(OUTLINE_FILE, json.dumps(outline.model_dump(mode="json"), indent=2) + "\n")
        return outline

    @override
    def write(self, ctx: StoryContext, plan: BeatOutline, *, seed: int) -> str:
        return write_scenes(ctx, plan.scene_specs(), seed=seed)
