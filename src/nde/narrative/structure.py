"""Typed structural links and explicit character knowledge, without prose inference."""

from typing import Literal

from pydantic import Field

from nde.llm.config import Strict


class StructuralRecord(Strict):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CausalLink(StructuralRecord):
    kind: Literal["causal"] = "causal"
    cause_scene_id: str
    effect_scene_id: str


class Setup(StructuralRecord):
    scene_id: str
    description: str


class PayoffLink(StructuralRecord):
    kind: Literal["setup_payoff"] = "setup_payoff"
    setup_id: str
    payoff_scene_id: str


class KnowledgeFact(StructuralRecord):
    description: str
    initial_holder_ids: list[str] = Field(default_factory=list[str])


class RevealLink(StructuralRecord):
    kind: Literal["reveal"] = "reveal"
    fact_id: str
    scene_id: str
    holder_ids: list[str] = Field(min_length=1)


class KnowledgeUse(StructuralRecord):
    fact_id: str
    scene_id: str
    holder_id: str
