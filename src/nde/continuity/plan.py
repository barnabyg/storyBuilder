"""The continuity plan: a sheet for each character and the scenes, made before any prose."""

from functools import cache
from typing import Self

from pydantic import Field, create_model, model_validator

from nde.llm.config import Strict
from nde.prose.scenes import SceneSpec

# Given to every prose and repair step, and kept beside the story.
PLAIN_STYLE_RULES = """\
Plain-style rules:
- Use at most one simile or metaphor in a scene. Most scenes need none.
- Never give objects, weather, light, silence or buildings feelings or intentions.
- Name what a person can see, hear, touch or do. Do not explain what it means.
- Keep most sentences under 20 words, and use no more than one adjective per noun.
- Use common words. Choose "said" or "asked" for speech, or no speech tag.
- Do not end a scene or paragraph on a reflective or poetic line. Stop on an event.
- Avoid "seemed to", "as if", "as though", "somehow", "something like" and "a kind of".
"""


class CharacterSheet(Strict):
    name: str = Field(min_length=1, description="the character's full name")
    appearance: str = Field(min_length=1, description="what a stranger would notice about them")
    back_story: str = Field(
        min_length=1, description="who they are and what happened to them before the story"
    )
    enters_scene: int = Field(ge=1, description="the number of the scene they first appear in")
    introduction: str = Field(
        min_length=1,
        description="how they enter, and what the reader is told about them when they do",
    )


class PlannedScene(Strict):
    summary: str = Field(min_length=1, description="what happens in the scene, in one sentence")
    beats: list[str] = Field(
        min_length=1, max_length=6, description="the scene's concrete events, in order"
    )


class ContinuityPlan(Strict):
    title: str = Field(min_length=1)
    characters: list[CharacterSheet] = Field(
        min_length=1, description="one sheet for every named person in the story"
    )
    scenes: list[PlannedScene] = Field(min_length=1)

    @model_validator(mode="after")
    def _sheets_fit_the_scenes(self) -> Self:
        names = [sheet.name.casefold() for sheet in self.characters]
        if len(set(names)) != len(names):
            raise ValueError("each character needs exactly one sheet")
        for sheet in self.characters:
            if sheet.enters_scene > len(self.scenes):
                raise ValueError(
                    f"{sheet.name} enters in scene {sheet.enters_scene}, but the plan has "
                    f"{len(self.scenes)} scenes"
                )
        return self

    def sheet_for(self, name: str) -> CharacterSheet | None:
        """The sheet a name in the prose refers to, such as "Priya" or "Ms Raman"."""
        for sheet in self.characters:
            if name.casefold() == sheet.name.casefold():
                return sheet
        words = set(name.casefold().split())
        for sheet in self.characters:
            if words & set(sheet.name.casefold().split()):
                return sheet
        return None


@cache
def continuity_plan_model(min_scenes: int, max_scenes: int) -> type[ContinuityPlan]:
    """A `ContinuityPlan` whose scene count is bounded by a brief's scene range."""
    return create_model(
        "ContinuityPlan",
        __base__=ContinuityPlan,
        scenes=(list[PlannedScene], Field(min_length=min_scenes, max_length=max_scenes)),
    )


def render_sheets(plan: ContinuityPlan) -> str:
    return "\n\n".join(
        f"{sheet.name}\n"
        f"Appearance: {sheet.appearance}\n"
        f"Back-story: {sheet.back_story}\n"
        f"Enters in scene {sheet.enters_scene}: {sheet.introduction}"
        for sheet in plan.characters
    )


def render_plan(plan: ContinuityPlan) -> str:
    """The sheets and scenes, for the steps that work on a finished story."""
    scenes = "\n".join(
        f"Scene {number}: {scene.summary}" for number, scene in enumerate(plan.scenes, 1)
    )
    return f"Character sheets:\n\n{render_sheets(plan)}\n\nScenes:\n{scenes}"


def scene_specs(plan: ContinuityPlan) -> list[SceneSpec]:
    """The planned scenes for the shared prose steps, with the sheets and rules as notes."""
    notes = f"Character sheets:\n\n{render_sheets(plan)}\n\n{PLAIN_STYLE_RULES}"
    return [
        SceneSpec(summary=scene.summary, beats=scene.beats, notes=notes) for scene in plan.scenes
    ]
