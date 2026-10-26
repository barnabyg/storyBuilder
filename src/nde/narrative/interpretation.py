"""Interpretive channels retain uncertainty; trajectories describe coarse ordinals."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from nde.narrative.base import Identified, Span, Strict
from nde.narrative.uncertainty import Reading

type Ordinal = Annotated[int, Field(strict=True, ge=0, le=4)]


class Shape(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"
    PEAK = "peak"
    VALLEY = "valley"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class OrdinalPoint(Strict):
    scene_id: str
    value: Ordinal | None = None


class Trajectory(Strict):
    shape: Shape
    points: list[OrdinalPoint] = Field(min_length=3, max_length=5)


class Annotation(Identified):
    kind: Literal["observation", "interpretation"] = "interpretation"
    scene_ids: list[str] = Field(default_factory=list[str])
    reading: Reading
    evidence: list[Span] = Field(default_factory=list[Span])


class Arc(Annotation):
    character_id: str | None = None
    trajectory: Trajectory | None = None


class ReaderAffect(Annotation):
    trajectory: Trajectory | None = None


class Tension(Annotation):
    trajectory: Trajectory | None = None


class Theme(Annotation):
    """A story-level theme or a scene-linked reading."""


class Motif(Annotation):
    """An occurrence of a recurring textual or interpreted motif."""


class Discourse(Annotation):
    """An observation or interpretation of textual discourse."""


class Subtext(Annotation):
    """A reading of implied meaning, with alternatives retained."""


class MotifEcho(Identified):
    from_motif_id: str
    to_motif_id: str
    kind: Literal["repetition", "transformation", "contrast"]
    evidence: list[Span] = Field(default_factory=list[Span])
