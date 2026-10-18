"""The G0 benchmark schema: a story outline shaped like the future narrative schema.

It deliberately combines what low-cost models find hard: nested objects, lists
with length bounds, enums, optional fields, and ID references that must
resolve across the document (scenes point at characters, locations and
earlier scenes).
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CharacterRole(StrEnum):
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    ALLY = "ally"
    MENTOR = "mentor"
    OTHER = "other"


class BeatKind(StrEnum):
    SETUP = "setup"
    COMPLICATION = "complication"
    TURN = "turn"
    CLIMAX = "climax"
    RESOLUTION = "resolution"


class Character(_Strict):
    id: str = Field(pattern=r"^c\d+$", description="c1, c2, ...")
    name: str = Field(min_length=1)
    role: CharacterRole
    want: str = Field(min_length=1, description="what the character is trying to get")
    flaw: str | None = None


class Location(_Strict):
    id: str = Field(pattern=r"^l\d+$", description="l1, l2, ...")
    name: str = Field(min_length=1)
    description: str | None = None


class Beat(_Strict):
    kind: BeatKind
    summary: str = Field(min_length=1)


class Scene(_Strict):
    id: str = Field(pattern=r"^s\d+$", description="s1, s2, ... in story order")
    title: str = Field(min_length=1)
    location_id: str
    pov_character_id: str
    character_ids: list[str] = Field(min_length=1)
    beats: list[Beat] = Field(min_length=1, max_length=4)
    tension: int = Field(ge=1, le=5)
    # Earlier scenes whose setups this scene pays off.
    pays_off_scene_ids: list[str] = Field(default_factory=list[str])


class StoryOutline(_Strict):
    title: str = Field(min_length=1)
    logline: str = Field(min_length=1)
    characters: list[Character] = Field(min_length=2, max_length=6)
    locations: list[Location] = Field(min_length=1, max_length=5)
    scenes: list[Scene] = Field(min_length=5, max_length=8)

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        problems: list[str] = []
        character_ids = [character.id for character in self.characters]
        location_ids = [location.id for location in self.locations]
        scene_ids = [scene.id for scene in self.scenes]
        for kind, ids in (
            ("character", character_ids),
            ("location", location_ids),
            ("scene", scene_ids),
        ):
            if len(set(ids)) != len(ids):
                problems.append(f"{kind} ids must be unique")
        for index, scene in enumerate(self.scenes):
            if scene.location_id not in location_ids:
                problems.append(f"scene {scene.id}: unknown location_id {scene.location_id!r}")
            if scene.pov_character_id not in character_ids:
                problems.append(
                    f"scene {scene.id}: unknown pov_character_id {scene.pov_character_id!r}"
                )
            elif scene.pov_character_id not in scene.character_ids:
                problems.append(f"scene {scene.id}: the pov character must be in character_ids")
            problems += [
                f"scene {scene.id}: unknown character id {ref!r}"
                for ref in scene.character_ids
                if ref not in character_ids
            ]
            earlier = scene_ids[:index]
            problems += [
                f"scene {scene.id}: pays_off_scene_ids must name earlier scenes, not {ref!r}"
                for ref in scene.pays_off_scene_ids
                if ref not in earlier
            ]
        if problems:
            raise ValueError("; ".join(problems))
        return self
