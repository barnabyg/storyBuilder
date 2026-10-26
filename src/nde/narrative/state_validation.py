"""Chronological validation of explicit transitions and constraints."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from nde.narrative.diagnostics import Diagnostic
from nde.narrative.state import Constraint, StateTransition
from nde.narrative.uncertainty import Reading

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan

type StateKey = tuple[str | None, str, str]


def _key(state: StateTransition | Constraint) -> StateKey:
    kind = state.kind if isinstance(state, StateTransition) else state.state_kind
    return state.character_id, kind, state.key


def _contradicts(previous: Reading, following: Reading) -> bool:
    return previous.status == following.status == "known" and previous.value != following.value


def _diagnostic(code: str, location: str, message: str, *, human: bool = False) -> Diagnostic:
    return Diagnostic(
        code=code, location=location, message=message, severity="human" if human else "error"
    )


def _state_transitions(plan: NarrativePlan) -> list[Diagnostic]:
    scenes = {scene.id: scene for scene in plan.scenes}
    order = {scene.id: index for index, scene in enumerate(plan.scenes)}
    characters = {character.id for character in plan.characters}
    previous: dict[StateKey, Reading] = {}
    seen: set[tuple[str, StateKey]] = set()
    diagnostics: list[Diagnostic] = []
    for state in sorted(plan.states, key=lambda state: order.get(state.scene_id, -1)):
        location = f"{state.id} scene {state.scene_id} character {state.character_id or 'story'}"
        scene = scenes.get(state.scene_id)
        if scene is None:
            diagnostics.append(
                _diagnostic("state_scene_reference", location, f"unknown scene {state.scene_id}")
            )
        if state.character_id is None:
            if state.kind != "canon":
                diagnostics.append(
                    _diagnostic(
                        "state_character_required", location, "only canon can have story-wide scope"
                    )
                )
        elif state.character_id not in characters:
            diagnostics.append(
                _diagnostic(
                    "state_character_reference", location, f"unknown character {state.character_id}"
                )
            )
        elif scene is not None and state.character_id not in scene.character_ids:
            diagnostics.append(
                _diagnostic("state_presence", location, "character is absent from transition scene")
            )
        key = _key(state)
        identity = state.scene_id, key
        if identity in seen:
            diagnostics.append(
                _diagnostic(
                    "state_duplicate",
                    location,
                    f"multiple transitions for {state.kind} {state.key}",
                )
            )
        seen.add(identity)
        if key in previous and _contradicts(previous[key], state.before):
            diagnostics.append(
                _diagnostic(
                    "state_transition",
                    location,
                    "before value contradicts previous explicit after value",
                )
            )
        previous[key] = state.after
    return diagnostics


def _relationships(plan: NarrativePlan) -> list[Diagnostic]:
    scenes = {scene.id: scene for scene in plan.scenes}
    order = {scene.id: index for index, scene in enumerate(plan.scenes)}
    characters = {character.id for character in plan.characters}
    previous: dict[tuple[tuple[str, ...], str], Reading] = {}
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    diagnostics: list[Diagnostic] = []
    for relation in sorted(plan.relationships, key=lambda item: order.get(item.scene_id, -1)):
        location = f"{relation.id} scene {relation.scene_id}"
        scene = scenes.get(relation.scene_id)
        if scene is None:
            diagnostics.append(
                _diagnostic(
                    "relationship_scene_reference", location, f"unknown scene {relation.scene_id}"
                )
            )
        if len(set(relation.participant_ids)) != len(relation.participant_ids):
            diagnostics.append(
                _diagnostic(
                    "relationship_participant_duplicate",
                    location,
                    "relationship participants must be distinct",
                )
            )
        for participant in relation.participant_ids:
            if participant not in characters:
                diagnostics.append(
                    _diagnostic(
                        "relationship_participant", location, f"unknown participant {participant}"
                    )
                )
            elif scene is not None and participant not in scene.character_ids:
                diagnostics.append(
                    _diagnostic(
                        "relationship_presence", location, f"participant {participant} is absent"
                    )
                )
        participants = tuple(sorted(relation.participant_ids))
        key = participants, relation.key
        identity = relation.scene_id, participants, relation.key
        if identity in seen:
            diagnostics.append(
                _diagnostic(
                    "relationship_duplicate",
                    location,
                    "multiple transitions for the same relationship",
                )
            )
        seen.add(identity)
        if key in previous and _contradicts(previous[key], relation.before):
            diagnostics.append(
                _diagnostic(
                    "relationship_transition",
                    location,
                    "before value contradicts previous explicit after value",
                )
            )
        previous[key] = relation.after
    return diagnostics


def constraint_status(plan: NarrativePlan, constraint: Constraint) -> str:
    if constraint.kind == "literary":
        return "human assessment required"
    matching = [state for state in plan.states if _key(state) == _key(constraint)]
    if not matching or not constraint.expected:
        return "unresolved: explicit facts missing"
    order = {scene.id: index for index, scene in enumerate(plan.scenes)}
    matching.sort(key=lambda item: order.get(item.scene_id, -1))
    readings = (
        [reading for state in matching for reading in (state.before, state.after)]
        if constraint.kind == "locked"
        else [matching[-1].after]
    )
    if any(
        reading.status == "known" and reading.value != constraint.expected for reading in readings
    ):
        return "violated"
    if any(reading.status != "known" for reading in readings):
        return "unresolved: human assessment required"
    return "satisfied"


def _constraints(plan: NarrativePlan) -> list[Diagnostic]:
    characters = {character.id for character in plan.characters}
    states: dict[StateKey, list[StateTransition]] = defaultdict(list)
    for state in plan.states:
        states[_key(state)].append(state)
    diagnostics: list[Diagnostic] = []
    for constraint in plan.constraints:
        location = f"{constraint.id} character {constraint.character_id or 'story'}"
        if constraint.character_id is not None and constraint.character_id not in characters:
            diagnostics.append(
                _diagnostic(
                    "constraint_character_reference",
                    location,
                    f"unknown character {constraint.character_id}",
                )
            )
        if constraint.kind == "literary":
            diagnostics.append(
                _diagnostic(
                    "constraint_human_assessment",
                    location,
                    constraint.note or constraint.key,
                    human=True,
                )
            )
            continue
        if constraint.character_id is None and constraint.state_kind != "canon":
            diagnostics.append(
                _diagnostic(
                    "constraint_character_required",
                    location,
                    "only canon can have story-wide scope",
                )
            )
        if not constraint.expected:
            diagnostics.append(
                _diagnostic(
                    "constraint_expected",
                    location,
                    "mechanical constraint needs an explicit expected value",
                )
            )
        status = constraint_status(plan, constraint)
        if status == "violated":
            code = "locked_value" if constraint.kind == "locked" else "required_ending"
            diagnostics.append(
                _diagnostic(code, location, f"{constraint.key}: expected {constraint.expected!r}")
            )
        elif status.startswith("unresolved"):
            code = (
                "constraint_unresolved" if states[_key(constraint)] else "constraint_fact_missing"
            )
            diagnostics.append(
                _diagnostic(code, location, f"{constraint.key}: {status}", human=True)
            )
    return diagnostics


def state_diagnostics(plan: NarrativePlan) -> list[Diagnostic]:
    return [*_state_transitions(plan), *_relationships(plan), *_constraints(plan)]
