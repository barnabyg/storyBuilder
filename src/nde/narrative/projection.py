"""Exact tier membership, pure projection and envelope-preserving serialization."""

from typing import Any, Literal

from nde.narrative.diagnostics import Diagnostic
from nde.narrative.models import NarrativePlan

type Tier = Literal["S", "M", "L"]

STRUCTURE = frozenset(
    {"causal_links", "setups", "payoff_links", "knowledge_facts", "reveal_links", "knowledge_uses"}
)
INTERPRETATION = frozenset(
    {"arcs", "reader_affect", "tension", "themes", "motifs", "motif_echoes", "discourse", "subtext"}
)
CHANNELS = STRUCTURE | INTERPRETATION | {"states", "relationships", "constraints"}
MEMBERS: dict[Tier, frozenset[str]] = {
    "S": frozenset(),
    "M": STRUCTURE | {"states"},
    "L": CHANNELS,
}


def tier_diagnostics(plan: NarrativePlan) -> list[Diagnostic]:
    data = plan.model_dump()
    excluded = CHANNELS - MEMBERS[plan.tier]
    leaks = [channel for channel in sorted(excluded) if data[channel]]
    if plan.tier == "M" and any(state.kind not in {"goal", "belief"} for state in plan.states):
        leaks.append("states outside goals/beliefs")
    return [
        Diagnostic(
            code="tier_membership", location=plan.id, message=f"tier {plan.tier} excludes {channel}"
        )
        for channel in leaks
    ]


def project(plan: NarrativePlan, tier: Tier) -> NarrativePlan:
    """Return a deep independent projection; IDs and all retained lists keep their order."""
    data = plan.model_dump(mode="json")
    data["tier"] = tier
    for channel in CHANNELS - MEMBERS[tier]:
        data[channel] = []
    if tier == "M":
        data["states"] = [state for state in data["states"] if state["kind"] in {"goal", "belief"}]
    return NarrativePlan.model_validate(data)


def serialized_plan(plan: NarrativePlan) -> dict[str, Any]:
    """Omit excluded channel keys while keeping explicit version and structural metadata."""
    return plan.model_dump(mode="json", exclude=set(CHANNELS - MEMBERS[plan.tier]))
