"""Flagged problems: the one record format every continuity check reports in."""

from pydantic import Field

from nde.llm.config import Strict


class Flag(Strict):
    """One problem a check found. Repair treats `problem_type` as an opaque label."""

    problem_type: str = Field(min_length=1)
    # Verbatim text from the story; repair finds the problem by it.
    passage: str = Field(min_length=1)
    reason: str = Field(min_length=1)
