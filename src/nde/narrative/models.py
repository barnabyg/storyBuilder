"""The narrative plan envelope. List position defines chronology, never ID spelling."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identified(Strict):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Source(Identified):
    title: str
    text: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: str
    span_basis: Literal["unicode-codepoints-lf"] = "unicode-codepoints-lf"


class Span(Strict):
    source_id: str
    source_sha256: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Character(Identified):
    name: str


class Beat(Identified):
    summary: str
    evidence: list[Span] = Field(default_factory=list[Span])


class Scene(Identified):
    title: str
    summary: str = ""
    character_ids: list[str] = Field(default_factory=list[str])
    beats: list[Beat] = Field(default_factory=list[Beat])
    evidence: list[Span] = Field(default_factory=list[Span])


class NarrativePlan(Identified):
    schema_version: Literal["1"] = "1"
    tier: Literal["S", "M", "L"] = "L"
    title: str
    characters: list[Character] = Field(default_factory=list[Character])
    sources: list[Source] = Field(default_factory=list[Source])
    scenes: list[Scene] = Field(min_length=1)
