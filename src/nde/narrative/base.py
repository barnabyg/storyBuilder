"""Shared strict records and source spans for narrative channels."""

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identified(Strict):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Span(Strict):
    source_id: str
    source_sha256: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
