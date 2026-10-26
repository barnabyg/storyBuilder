"""The narrative plan envelope. List position defines chronology, never ID spelling."""

from typing import Literal

from pydantic import Field

from nde.narrative.base import Identified, Span, Strict
from nde.narrative.state import Constraint, RelationshipTransition, StateTransition
from nde.narrative.structure import (
    CausalLink,
    KnowledgeFact,
    KnowledgeUse,
    PayoffLink,
    RevealLink,
    Setup,
)

__all__ = ["Identified", "Span", "Strict"]


class Source(Identified):
    title: str
    text: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: str
    span_basis: Literal["unicode-codepoints-lf"] = "unicode-codepoints-lf"


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
    causal_links: list[CausalLink] = Field(default_factory=list[CausalLink])
    setups: list[Setup] = Field(default_factory=list[Setup])
    payoff_links: list[PayoffLink] = Field(default_factory=list[PayoffLink])
    knowledge_facts: list[KnowledgeFact] = Field(default_factory=list[KnowledgeFact])
    reveal_links: list[RevealLink] = Field(default_factory=list[RevealLink])
    knowledge_uses: list[KnowledgeUse] = Field(default_factory=list[KnowledgeUse])
    states: list[StateTransition] = Field(default_factory=list[StateTransition])
    relationships: list[RelationshipTransition] = Field(
        default_factory=list[RelationshipTransition]
    )
    constraints: list[Constraint] = Field(default_factory=list[Constraint])
