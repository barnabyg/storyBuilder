"""Explicit character changes and represented hard facts; no prose inference."""

from typing import Literal

from pydantic import Field

from nde.narrative.base import Identified
from nde.narrative.uncertainty import Reading

type StateKind = Literal["goal", "belief", "value", "canon"]


class StateTransition(Identified):
    scene_id: str
    # None identifies a story-wide canon fact, rather than a character.
    character_id: str | None = None
    kind: StateKind
    key: str = Field(min_length=1)
    before: Reading
    after: Reading


class RelationshipTransition(Identified):
    scene_id: str
    participant_ids: list[str] = Field(min_length=2)
    key: str = Field(min_length=1)
    before: Reading
    after: Reading


class Constraint(Identified):
    kind: Literal["locked", "required_ending", "literary"]
    character_id: str | None = None
    state_kind: StateKind = "canon"
    key: str = Field(min_length=1)
    expected: str | None = None
    note: str = ""
