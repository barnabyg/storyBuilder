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


KNOWLEDGE_RULE = (
    "A character may act only on what their sheet says they know at the start, or on "
    "what they have seen, heard or worked out earlier on the page."
)


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
    knows_at_start: list[str] = Field(
        description="the facts that matter to the plot which they already know when the "
        "story starts, one sentence each. Leave out anything they learn during the story."
    )


class KeyEvent(Strict):
    event: str = Field(min_length=1, description="an event in the scene that takes real time")
    expected_min_minutes: float = Field(ge=0, description="the least time it could really take")
    expected_max_minutes: float = Field(ge=0, description="the most time it could really take")

    @model_validator(mode="after")
    def _range_is_ordered(self) -> Self:
        if self.expected_min_minutes > self.expected_max_minutes:
            raise ValueError(f"{self.event}: the least time is more than the most time")
        return self


class PlannedScene(Strict):
    summary: str = Field(min_length=1, description="what happens in the scene, in one sentence")
    beats: list[str] = Field(
        min_length=1, max_length=6, description="the scene's concrete events, in order"
    )
    elapsed_minutes: float = Field(
        ge=0,
        description="the story time, in minutes, from the end of the previous scene to the "
        "end of this one",
    )
    key_events: list[KeyEvent] = Field(
        description="the events whose realistic duration matters: spoiling, healing, travel, "
        "weather, work. Empty if none."
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

    def key_events(self) -> dict[str, KeyEvent]:
        """Every key event by its label, K1 onwards, in story order."""
        events = [event for scene in self.scenes for event in scene.key_events]
        return {f"K{number}": event for number, event in enumerate(events, 1)}

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


def format_minutes(minutes: float) -> str:
    """A duration a person can read: "4 minutes", "3 hours", "2 days"."""
    for size, unit in ((1440, "day"), (60, "hour")):
        if minutes >= size:
            amount = minutes / size
            shown = f"{amount:.1f}".removesuffix(".0")
            return f"{shown} {unit}{'' if shown == '1' else 's'}"
    shown = f"{minutes:.1f}".removesuffix(".0")
    return f"{shown} minute{'' if shown == '1' else 's'}"


def _render_key_event(label: str, event: KeyEvent) -> str:
    low, high = (
        format_minutes(event.expected_min_minutes),
        format_minutes(event.expected_max_minutes),
    )
    return f"{label}: {event.event} ({low} to {high})"


def render_sheets(plan: ContinuityPlan) -> str:
    return "\n\n".join(_render_sheet(sheet) for sheet in plan.characters)


def _render_sheet(sheet: CharacterSheet) -> str:
    known = [f"{number}. {fact}" for number, fact in enumerate(sheet.knows_at_start, 1)]
    return "\n".join(
        [
            sheet.name,
            f"Appearance: {sheet.appearance}",
            f"Back-story: {sheet.back_story}",
            f"Enters in scene {sheet.enters_scene}: {sheet.introduction}",
            "Knows at the start:",
            *(known or ["Nothing that matters to the plot."]),
        ]
    )


def render_plan(plan: ContinuityPlan) -> str:
    """The sheets and scenes, for the steps that work on a finished story."""
    labels = iter(plan.key_events().items())
    lines: list[str] = []
    for number, scene in enumerate(plan.scenes, 1):
        elapsed = format_minutes(scene.elapsed_minutes)
        lines.append(f"Scene {number} ({elapsed} since the scene before): {scene.summary}")
        lines += [_render_key_event(*next(labels)) for _ in scene.key_events]
    scenes = "\n".join(lines)
    return f"Character sheets:\n\n{render_sheets(plan)}\n\nScenes:\n{scenes}"


def _clock_note(scene: PlannedScene) -> str:
    lines = [f"Scene clock: {format_minutes(scene.elapsed_minutes)} pass since the scene before."]
    lines += [
        f"{event.event} takes {format_minutes(event.expected_min_minutes)} to "
        f"{format_minutes(event.expected_max_minutes)}. Do not show it faster or slower."
        for event in scene.key_events
    ]
    return "\n".join(lines)


def scene_specs(plan: ContinuityPlan) -> list[SceneSpec]:
    """The planned scenes for the shared prose steps, with the sheets and rules as notes."""
    notes = f"Character sheets:\n\n{render_sheets(plan)}\n\n{KNOWLEDGE_RULE}\n\n{PLAIN_STYLE_RULES}"
    return [
        SceneSpec(
            summary=scene.summary, beats=scene.beats, notes=f"{_clock_note(scene)}\n\n{notes}"
        )
        for scene in plan.scenes
    ]
