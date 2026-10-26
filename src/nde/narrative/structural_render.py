"""Scene-card presentation for causal and knowledge records."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan


def structural_lines(plan: NarrativePlan, scene_id: str) -> list[str]:
    lines: list[str] = []
    if scene_id == plan.scenes[0].id:
        lines.extend(
            f"Fact {fact.id}: {fact.description}; initial holders: "
            f"{', '.join(fact.initial_holder_ids) or '(none)'}"
            for fact in plan.knowledge_facts
        )
    lines.extend(
        f"Causal {link.id}: earlier cause {link.cause_scene_id} → effect {link.effect_scene_id}"
        for link in plan.causal_links
        if scene_id in (link.cause_scene_id, link.effect_scene_id)
    )
    lines.extend(
        f"Setup {setup.id} at {setup.scene_id}: {setup.description}"
        for setup in plan.setups
        if setup.scene_id == scene_id
    )
    lines.extend(
        f"Payoff {link.id}: setup {link.setup_id} → payoff at {link.payoff_scene_id}"
        for link in plan.payoff_links
        if link.payoff_scene_id == scene_id
    )
    lines.extend(
        f"Reveal {link.id}: fact {link.fact_id} at {link.scene_id}; holders: "
        f"{', '.join(link.holder_ids)} (acquired at scene start)"
        for link in plan.reveal_links
        if link.scene_id == scene_id
    )
    lines.extend(
        f"Knowledge use {use.id}: {use.holder_id} uses fact {use.fact_id} at {use.scene_id}"
        for use in plan.knowledge_uses
        if use.scene_id == scene_id
    )
    return lines
