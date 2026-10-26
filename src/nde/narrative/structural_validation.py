"""Mechanical invariants for chronology, setups and explicit character knowledge."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING

from nde.narrative.diagnostics import Diagnostic
from nde.narrative.structure import KnowledgeUse, RevealLink

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan


def _causality(plan: NarrativePlan) -> list[Diagnostic]:
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    graph: dict[str, set[str]] = {scene.id: set() for scene in plan.scenes}
    result: list[Diagnostic] = []
    for link in plan.causal_links:
        cause, effect = link.cause_scene_id, link.effect_scene_id
        if cause not in positions or effect not in positions:
            code = "causal_endpoint"
        elif cause == effect:
            code = "causal_self_link"
        elif positions[cause] >= positions[effect]:
            code = "causal_order"
        else:
            code = ""
        if code:
            result.append(
                Diagnostic(
                    code=code,
                    location=link.id,
                    message=f"cause {cause} must be an earlier scene than {effect}",
                )
            )
        if cause in positions and effect in positions:
            graph[effect].add(cause)
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as error:
        result.append(
            Diagnostic(
                code="causal_cycle",
                location=plan.id,
                message=f"causal links contain a cycle: {error.args[1]}",
            )
        )
    return result


def _setups(plan: NarrativePlan) -> list[Diagnostic]:
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    setups = {setup.id: setup for setup in plan.setups}
    paid: set[str] = set()
    result: list[Diagnostic] = []
    for setup in plan.setups:
        if setup.scene_id not in positions:
            result.append(
                Diagnostic(
                    code="setup_scene_reference",
                    location=setup.id,
                    message=f"unknown setup scene {setup.scene_id}",
                )
            )
    for link in plan.payoff_links:
        setup = setups.get(link.setup_id)
        if setup is None:
            code = "payoff_setup_reference"
        elif link.payoff_scene_id not in positions:
            code = "payoff_scene_reference"
        elif setup.scene_id not in positions:
            continue
        elif positions[setup.scene_id] >= positions[link.payoff_scene_id]:
            code = "payoff_order"
        else:
            paid.add(setup.id)
            continue
        result.append(
            Diagnostic(
                code=code,
                location=link.id,
                message=f"setup {link.setup_id} must precede payoff scene {link.payoff_scene_id}",
            )
        )
    result.extend(
        Diagnostic(
            code="unpaid_setup",
            location=setup.id,
            message=f"setup {setup.id} at {setup.scene_id} has no payoff",
            severity="advisory",
        )
        for setup in plan.setups
        if setup.id not in paid
    )
    return result


def validate_structure(plan: NarrativePlan) -> list[Diagnostic]:
    return [*_causality(plan), *_setups(plan), *_knowledge(plan)]


def _initial_knowledge(plan: NarrativePlan) -> list[Diagnostic]:
    characters = {character.id for character in plan.characters}
    seen: set[str] = set()
    result: list[Diagnostic] = []
    for fact in plan.knowledge_facts:
        if fact.id in seen:
            result.append(
                Diagnostic(
                    code="duplicate_fact_id",
                    location=fact.id,
                    message=f"duplicate knowledge fact {fact.id}",
                )
            )
        seen.add(fact.id)
        if len(fact.initial_holder_ids) != len(set(fact.initial_holder_ids)):
            result.append(
                Diagnostic(
                    code="duplicate_fact_reference",
                    location=fact.id,
                    message="initial holders must be unique",
                )
            )
        result.extend(
            Diagnostic(
                code="knowledge_holder",
                location=fact.id,
                message=f"unknown initial holder {holder}",
            )
            for holder in fact.initial_holder_ids
            if holder not in characters
        )
    return result


def _knowledge_event(plan: NarrativePlan, event: RevealLink | KnowledgeUse) -> list[Diagnostic]:
    scenes = {scene.id: scene for scene in plan.scenes}
    facts = {fact.id for fact in plan.knowledge_facts}
    characters = {character.id for character in plan.characters}
    holders = event.holder_ids if isinstance(event, RevealLink) else [event.holder_id]
    scene = scenes.get(event.scene_id)
    result: list[Diagnostic] = []
    if scene is None:
        result.append(
            Diagnostic(
                code="knowledge_scene",
                location=event.id,
                message=f"unknown knowledge scene {event.scene_id}",
            )
        )
    if event.fact_id not in facts:
        result.append(
            Diagnostic(
                code="knowledge_fact",
                location=event.id,
                message=f"unknown fact {event.fact_id} at {event.scene_id}",
            )
        )
    if len(holders) != len(set(holders)):
        result.append(
            Diagnostic(
                code="duplicate_fact_reference",
                location=event.id,
                message=f"fact holders must be unique at {event.scene_id}",
            )
        )
    for holder in holders:
        if holder not in characters:
            result.append(
                Diagnostic(
                    code="knowledge_holder",
                    location=event.id,
                    message=f"unknown holder {holder} at {event.scene_id}",
                )
            )
        elif scene is not None and holder not in scene.character_ids:
            result.append(
                Diagnostic(
                    code="knowledge_presence",
                    location=event.id,
                    message=f"holder {holder} is absent at {event.scene_id}",
                )
            )
    return result


def _acquisitions(plan: NarrativePlan) -> tuple[dict[tuple[str, str], int], list[Diagnostic]]:
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    acquired = {
        (fact.id, holder): -1 for fact in plan.knowledge_facts for holder in fact.initial_holder_ids
    }
    result: list[Diagnostic] = []
    # Process in story order, independent of reveal list order.
    for reveal in sorted(plan.reveal_links, key=lambda item: positions.get(item.scene_id, -1)):
        diagnostics = _knowledge_event(plan, reveal)
        result.extend(diagnostics)
        if diagnostics:
            continue
        for holder in reveal.holder_ids:
            key = (reveal.fact_id, holder)
            if key in acquired:
                result.append(
                    Diagnostic(
                        code="duplicate_acquisition",
                        location=reveal.id,
                        message=f"{holder} already holds fact {reveal.fact_id} "
                        f"before reveal at {reveal.scene_id}",
                    )
                )
            else:
                acquired[key] = positions[reveal.scene_id]
    return acquired, result


def _knowledge(plan: NarrativePlan) -> list[Diagnostic]:
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    acquired, result = _acquisitions(plan)
    result.extend(_initial_knowledge(plan))
    seen: set[tuple[str, str, str]] = set()
    for use in plan.knowledge_uses:
        diagnostics = _knowledge_event(plan, use)
        result.extend(diagnostics)
        key = (use.fact_id, use.holder_id, use.scene_id)
        if key in seen:
            result.append(
                Diagnostic(
                    code="duplicate_knowledge_use",
                    location=use.id,
                    message=f"duplicate use of {use.fact_id} by {use.holder_id} at {use.scene_id}",
                )
            )
        seen.add(key)
        if diagnostics:
            continue
        when = acquired.get((use.fact_id, use.holder_id))
        if when is None or when > positions[use.scene_id]:
            result.append(
                Diagnostic(
                    code="knowledge_before_acquisition",
                    location=use.id,
                    message=f"{use.holder_id} uses fact {use.fact_id} "
                    f"before acquiring it at {use.scene_id}",
                )
            )
    return result
