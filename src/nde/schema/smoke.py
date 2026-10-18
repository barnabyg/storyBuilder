"""A small nested schema for exercising structured output end to end."""

from pydantic import BaseModel, ConfigDict, Field


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)


class StoryPremise(BaseModel):
    """A one-paragraph story premise with a protagonist and three to five beats."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    protagonist: Character
    beats: list[str] = Field(min_length=3, max_length=5)
