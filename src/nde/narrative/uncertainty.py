"""A reading retains uncertainty instead of turning absence into a confident label."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    status: Literal["known", "unknown", "ambiguous"] = "known"
    confidence: float | None = Field(default=None, ge=0, le=1)
    alternatives: list[str] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _value_matches_status(self) -> Self:
        if self.status == "known" and not self.value:
            raise ValueError("known reading requires a nonempty value")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("unknown reading must not assert a value")
        return self


def describe_reading(reading: Reading) -> str:
    value = reading.value if reading.value is not None else "(no asserted value)"
    confidence = "unrecorded" if reading.confidence is None else str(reading.confidence)
    alternatives = "; ".join(reading.alternatives) or "(none recorded)"
    return f"{reading.status}: {value}; confidence {confidence}; alternatives: {alternatives}"
