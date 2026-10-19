"""The rubric and failure taxonomy that the blind reader and the LLM judges score against.

The rubric is a versioned data file, `benchmark/rubric/v<n>.yaml`, released
like a prompt template: `benchmark/rubric/checksums.json` records each
version's checksum, and changing a released version's text is rejected. To
change the wording, save it as a new version and run `nde rubric lock`.
Batches and judge runs record the rubric version they used.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nde import versioned

DEFAULT_RUBRIC_DIR = Path("benchmark") / "rubric"
RUBRIC = "rubric"


class RubricError(Exception):
    """The rubric version cannot be loaded."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Scale(_Strict):
    min: int = Field(ge=0)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.min >= self.max:
            raise ValueError("scale min must be below max")
        return self

    @property
    def middle(self) -> float:
        return (self.min + self.max) / 2


class Anchors(_Strict):
    """What the low end, the middle and the high end of the scale look like."""

    low: str = Field(min_length=1)
    middle: str = Field(min_length=1)
    high: str = Field(min_length=1)


class Dimension(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z_]*$")
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    anchors: Anchors


class FailureTag(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z-]*$")
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    example: str = Field(min_length=1)


class Rubric(_Strict):
    version: int = Field(ge=1)
    scale: Scale
    dimensions: list[Dimension] = Field(min_length=1)
    failure_tags: list[FailureTag] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        for kind, ids in (
            ("dimension", [dimension.id for dimension in self.dimensions]),
            ("failure tag", [tag.id for tag in self.failure_tags]),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{kind} ids must be unique")
        return self

    @property
    def dimension_ids(self) -> list[str]:
        return [dimension.id for dimension in self.dimensions]

    @property
    def tag_ids(self) -> list[str]:
        return [tag.id for tag in self.failure_tags]

    def describe(self) -> str:
        """The rubric as plain text, for judge prompts."""
        scale = self.scale
        lines = [f"Score each dimension from {scale.min} to {scale.max}.", ""]
        for dimension in self.dimensions:
            anchors = dimension.anchors
            lines += [
                f"{dimension.id} ({dimension.name}): {dimension.definition}",
                f"  {scale.min}: {anchors.low}",
                f"  middle: {anchors.middle}",
                f"  {scale.max}: {anchors.high}",
            ]
        lines += ["", "Failure tags (list every one that applies):"]
        lines += [f"- {tag.id}: {tag.definition}" for tag in self.failure_tags]
        return "\n".join(lines)


def rubric_path(directory: Path, version: int) -> Path:
    return directory / f"v{version}.yaml"


def load_rubric(directory: Path, version: int) -> Rubric:
    label = f"{RUBRIC}@{version}"
    path = rubric_path(directory, version)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RubricError(f"rubric v{version} not found at {path}") from error
    released = versioned.status(directory, label, data)
    if released == "unregistered":
        raise RubricError(f"rubric v{version} is not registered; run `nde rubric lock`")
    if released == "changed":
        raise RubricError(
            f"rubric v{version} changed after release; save the new text as a new version"
        )
    try:
        rubric = Rubric.model_validate(yaml.safe_load(data))
    except (yaml.YAMLError, ValidationError) as error:
        raise RubricError(f"rubric v{version} is malformed: {error}") from error
    if rubric.version != version:
        raise RubricError(f"rubric file {path.name} declares version {rubric.version}")
    return rubric


def check(directory: Path) -> tuple[list[str], list[str]]:
    """Return (unregistered labels, changed labels) for rubric versions."""
    return versioned.check(directory, RUBRIC)


def lock(directory: Path) -> tuple[list[str], list[str]]:
    """Record checksums of new rubric versions; returns (added, changed) labels."""
    return versioned.lock(directory, RUBRIC)
