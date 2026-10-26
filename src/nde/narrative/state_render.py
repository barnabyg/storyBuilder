"""Readable state and relationship transitions, including unresolved constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nde.narrative.state_validation import constraint_status
from nde.narrative.uncertainty import describe_reading

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan


def state_card_lines(plan: NarrativePlan) -> list[str]:
    lines: list[str] = []
    for scene in plan.scenes:
        for state in plan.states:
            if state.scene_id == scene.id:
                lines.append(
                    f"State {state.id} [{scene.id}] {state.character_id or 'story'} "
                    f"{state.kind} {state.key}: {describe_reading(state.before)} "
                    f"→ {describe_reading(state.after)}"
                )
        for relationship in plan.relationships:
            if relationship.scene_id == scene.id:
                participants = ", ".join(relationship.participant_ids)
                lines.append(
                    f"Relationship {relationship.id} [{scene.id}] {participants} "
                    f"{relationship.key}: {describe_reading(relationship.before)} "
                    f"→ {describe_reading(relationship.after)}"
                )
    for constraint in plan.constraints:
        lines.append(
            f"Constraint {constraint.id}: {constraint_status(plan, constraint)}; "
            f"{constraint.kind} {constraint.character_id or 'story'} "
            f"{constraint.state_kind} {constraint.key}; "
            f"expected {constraint.expected!r}; {constraint.note}"
        )
    return lines
