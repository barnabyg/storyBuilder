"""Briefs: the benchmark's unit of input.

A brief says what story to write: genre, premise, protagonist, central
conflict, themes, an ending constraint and a target length. Every condition
writes one story per brief, so briefs are what stories are paired by.

`BriefContent` is what a drafting model produces; a `Brief` adds the stable ID
and the split (dev or test) that the owner assigns when curating it.
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Split = Literal["dev", "test"]

SPLITS: tuple[Split, ...] = ("dev", "test")

DEFAULT_TARGET_WORDS = 2000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndingKind(StrEnum):
    HAPPY = "happy"
    TRAGIC = "tragic"
    BITTERSWEET = "bittersweet"
    AMBIGUOUS = "ambiguous"
    TWIST = "twist"
    OPEN = "open"


class Protagonist(_Strict):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1, description="who they are, in one sentence")


class EndingConstraint(_Strict):
    kind: EndingKind
    detail: str = Field(min_length=1, description="what the ending must do, in one sentence")


class BriefContent(_Strict):
    """A brief's content, without its ID and split."""

    genre: str = Field(min_length=1)
    premise: str = Field(min_length=1, description="the situation the story starts from")
    protagonist: Protagonist
    central_conflict: str = Field(min_length=1)
    themes: list[str] = Field(min_length=1, max_length=4)
    ending_constraint: EndingConstraint
    target_words: int = Field(default=DEFAULT_TARGET_WORDS, ge=1000, le=8000)
    min_scenes: int = Field(default=5, ge=2, le=12)
    max_scenes: int = Field(default=8, ge=2, le=12)

    @model_validator(mode="after")
    def _scene_range_is_ordered(self) -> Self:
        if self.min_scenes > self.max_scenes:
            raise ValueError("min_scenes must not exceed max_scenes")
        return self

    def describe(self) -> str:
        """The brief as plain text, for prompts."""
        return "\n".join(
            [
                f"Genre: {self.genre}",
                f"Premise: {self.premise}",
                f"Protagonist: {self.protagonist.name}, {self.protagonist.description}",
                f"Central conflict: {self.central_conflict}",
                f"Themes: {', '.join(self.themes)}",
                f"Ending ({self.ending_constraint.kind.value}): {self.ending_constraint.detail}",
            ]
        )


class Brief(BriefContent):
    id: str = Field(pattern=r"^brief-\d{2,}$", description="brief-01, brief-02, ...")
    split: Split


class BriefDrafts(_Strict):
    """A batch of candidate briefs from the drafting prompt, for the owner to curate."""

    briefs: list[BriefContent] = Field(min_length=1, max_length=20)
