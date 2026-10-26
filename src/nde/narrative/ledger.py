"""Deterministic replay of represented planned state, never inferred prose state."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from nde.files import write_text_lf
from nde.narrative.base import Strict
from nde.narrative.interpretation import Motif
from nde.narrative.models import NarrativePlan
from nde.narrative.projection import Tier, serialized_plan
from nde.narrative.state import Constraint, StateKind
from nde.narrative.structure import KnowledgeUse, Setup
from nde.narrative.uncertainty import Reading, describe_reading
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError


class StateValue(Strict):
    character_id: str | None
    kind: StateKind
    key: str
    reading: Reading


class RelationshipValue(Strict):
    participant_ids: list[str]
    key: str
    reading: Reading


class KnowledgeValue(Strict):
    fact_id: str
    description: str
    holder_ids: list[str]


class Snapshot(Strict):
    states: list[StateValue] = Field(default_factory=list[StateValue])
    relationships: list[RelationshipValue] = Field(default_factory=list[RelationshipValue])
    knowledge: list[KnowledgeValue] = Field(default_factory=list[KnowledgeValue])
    open_setups: list[Setup] = Field(default_factory=list[Setup])
    motifs: list[Motif] = Field(default_factory=list[Motif])


class SceneLedger(Strict):
    scene_id: str
    before: Snapshot
    after: Snapshot
    summary: str
    after_summary: str
    knowledge_uses: list[KnowledgeUse] = Field(default_factory=list[KnowledgeUse])


class Ledger(Strict):
    schema_version: Literal["1"] = "1"
    status: Literal["planned-only"] = "planned-only"
    plan_id: str
    plan_sha256: str
    tier: Tier
    constraints: list[Constraint]
    scenes: list[SceneLedger]


def plan_identity(plan: NarrativePlan) -> str:
    """Canonical content identity, independent of JSON whitespace or checkout newlines."""
    canonical = json.dumps(serialized_plan(plan), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _summary(snapshot: Snapshot, plan: NarrativePlan, present: list[str]) -> str:
    lines = ["Planned state only; prose realisation is unassessed."]
    lines.extend(
        f"Requirement {item.id} ({item.kind}; {item.character_id or 'story'} {item.state_kind}): "
        f"{item.key} = {item.expected or item.note}"
        for item in plan.constraints
    )
    lines.extend(
        f"{item.character_id or 'story'} {item.kind} {item.key}: {describe_reading(item.reading)}"
        for item in snapshot.states
        if item.kind in {"canon", "value"} or item.character_id in present
    )
    lines.extend(
        f"Relationship {','.join(item.participant_ids)} {item.key}: "
        f"{describe_reading(item.reading)}"
        for item in snapshot.relationships
        if set(item.participant_ids) & set(present)
    )
    lines.extend(
        f"Knowledge {item.fact_id}: {item.description}; holders: {','.join(item.holder_ids)}"
        for item in snapshot.knowledge
        if set(item.holder_ids) & set(present)
    )
    lines.extend(f"Open setup {item.id}: {item.description}" for item in snapshot.open_setups)
    lines.extend(f"Motif {item.id}: {describe_reading(item.reading)}" for item in snapshot.motifs)
    return "\n".join(lines)


def _scene_events(current: Snapshot, plan: NarrativePlan, scene_id: str) -> None:
    facts = {fact.id: fact for fact in plan.knowledge_facts}
    for reveal in plan.reveal_links:
        if reveal.scene_id != scene_id:
            continue
        knowledge = next(
            (item for item in current.knowledge if item.fact_id == reveal.fact_id), None
        )
        if knowledge is None:
            knowledge = KnowledgeValue(
                fact_id=reveal.fact_id, description=facts[reveal.fact_id].description, holder_ids=[]
            )
            current.knowledge.append(knowledge)
        knowledge.holder_ids = sorted(set(knowledge.holder_ids) | set(reveal.holder_ids))
    current.open_setups.extend(
        item.model_copy(deep=True) for item in plan.setups if item.scene_id == scene_id
    )
    paid = {item.setup_id for item in plan.payoff_links if item.payoff_scene_id == scene_id}
    current.open_setups = [item for item in current.open_setups if item.id not in paid]
    # A multi-scene motif is recorded at its first occurrence, with only completed
    # scene references exposed. Story-level interpretations have no timed occurrence.
    seen = {item.id: item for item in current.motifs}
    for motif in plan.motifs:
        if scene_id not in motif.scene_ids:
            continue
        if motif.id not in seen:
            occurrence = motif.model_copy(deep=True)
            occurrence.scene_ids = [scene_id]
            current.motifs.append(occurrence)
        else:
            seen[motif.id].scene_ids.append(scene_id)


def _boundary(
    current: Snapshot, plan: NarrativePlan, scene_id: str, boundary: Literal["before", "after"]
) -> None:
    states = {(item.character_id, item.kind, item.key): item for item in current.states}
    for item in plan.states:
        if item.scene_id == scene_id:
            states[item.character_id, item.kind, item.key] = StateValue(
                character_id=item.character_id,
                kind=item.kind,
                key=item.key,
                reading=getattr(item, boundary).model_copy(deep=True),
            )
    current.states = list(states.values())
    relationships = {
        (tuple(sorted(item.participant_ids)), item.key): item for item in current.relationships
    }
    for relation in plan.relationships:
        if relation.scene_id == scene_id:
            participants = tuple(sorted(relation.participant_ids))
            relationships[participants, relation.key] = RelationshipValue(
                participant_ids=list(participants),
                key=relation.key,
                reading=getattr(relation, boundary).model_copy(deep=True),
            )
    current.relationships = list(relationships.values())


def replay(plan: NarrativePlan) -> Ledger:
    """Replay in list order; only the current scene's before readings introduce new state."""
    result = validate_plan(plan)
    if not result.valid:
        details = "; ".join(
            f"{item.code} [{item.location}]: {item.message}"
            for item in result.diagnostics
            if item.severity == "error"
        )
        raise PlanError(f"cannot replay invalid plan: {details}")
    current = Snapshot(
        knowledge=[
            KnowledgeValue(
                fact_id=fact.id,
                description=fact.description,
                holder_ids=sorted(fact.initial_holder_ids),
            )
            for fact in plan.knowledge_facts
            if fact.initial_holder_ids
        ]
    )
    scenes: list[SceneLedger] = []
    for scene in plan.scenes:
        _boundary(current, plan, scene.id, "before")
        before = current.model_copy(deep=True)
        _boundary(current, plan, scene.id, "after")
        _scene_events(current, plan, scene.id)
        after = current.model_copy(deep=True)
        scenes.append(
            SceneLedger(
                scene_id=scene.id,
                before=before,
                after=after,
                summary=_summary(before, plan, scene.character_ids),
                after_summary=_summary(after, plan, scene.character_ids),
                knowledge_uses=[
                    item.model_copy(deep=True)
                    for item in plan.knowledge_uses
                    if item.scene_id == scene.id
                ],
            )
        )
    return Ledger(
        plan_id=plan.id,
        plan_sha256=plan_identity(plan),
        tier=plan.tier,
        constraints=[item.model_copy(deep=True) for item in plan.constraints],
        scenes=scenes,
    )


def render_ledger(plan: NarrativePlan, out: Path) -> Ledger:
    ledger = replay(plan)
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "ledger.json", ledger.model_dump_json(indent=2) + "\n")
    lines = [f"# Planned state ledger: {plan.title}", f"Plan {plan.id}; {ledger.plan_sha256}"]
    for scene in ledger.scenes:
        lines.extend(
            [f"## {scene.scene_id}", "### Before", scene.summary, "### After", scene.after_summary]
        )
    write_text_lf(out / "ledger.md", "\n\n".join(lines) + "\n")
    write_text_lf(out / "validation.json", validate_plan(plan).model_dump_json(indent=2) + "\n")
    return ledger
