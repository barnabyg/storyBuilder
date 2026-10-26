"""Baseline C: validated narrative intentions through B's unchanged prose contract."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self, override

from pydantic import JsonValue, model_validator

from nde.narrative.interpretation import iter_annotations
from nde.narrative.ledger import plan_identity, replay
from nde.narrative.models import NarrativePlan
from nde.narrative.projection import MEMBERS, Tier, project
from nde.narrative.workflow import PlanError, rendered_files
from nde.plan.imported import import_plan
from nde.plan.narrative import NarrativePlanningParams, generate_plan, planning_templates
from nde.prose.scenes import SCENE_PROSE, SCENE_VARIABLES, SceneSpec, scene_step, write_scenes
from nde.runs.config import RoleName
from nde.stories.pipeline import Pipeline, StepFailedError, StoryContext, TemplateNeed


class BaselineCParams(NarrativePlanningParams):
    tier: Tier = "L"
    saved_plan: str | None = None
    provenance: str | None = None

    @model_validator(mode="after")
    def _import_is_attributed(self) -> Self:
        if (self.saved_plan is None) != (self.provenance is None):
            raise ValueError("saved_plan and provenance must be supplied together")
        return self

    def planning(self) -> NarrativePlanningParams:
        return NarrativePlanningParams.model_validate(
            self.model_dump(exclude={"tier", "saved_plan", "provenance"})
        )


def _scene_record(plan: NarrativePlan, scene_id: str) -> dict[str, object]:
    scene = next(scene for scene in plan.scenes if scene.id == scene_id)
    record: dict[str, object] = {"scene": scene.model_dump(mode="json")}
    for channel in ("states", "relationships", "setups", "reveal_links", "knowledge_uses"):
        if channel not in MEMBERS[plan.tier]:
            continue
        record[channel] = [
            item.model_dump(mode="json")
            for item in getattr(plan, channel)
            if item.scene_id == scene_id
        ]
    if "causal_links" in MEMBERS[plan.tier]:
        record["causal_links"] = [
            item.model_dump(mode="json")
            for item in plan.causal_links
            if scene_id in {item.cause_scene_id, item.effect_scene_id}
        ]
        record["payoff_links"] = [
            item.model_dump(mode="json")
            for item in plan.payoff_links
            if item.payoff_scene_id == scene_id
        ]
        relevant_facts = {item.fact_id for item in plan.reveal_links if item.scene_id == scene_id}
        relevant_facts.update(
            item.fact_id for item in plan.knowledge_uses if item.scene_id == scene_id
        )
        record["knowledge_facts"] = [
            item.model_dump(mode="json")
            for item in plan.knowledge_facts
            if item.id in relevant_facts
        ]
    annotations: dict[str, list[dict[str, Any]]] = {}
    for channel, item in iter_annotations(plan):
        if not item.scene_ids or scene_id in item.scene_ids:
            annotations.setdefault(channel, []).append(item.model_dump(mode="json"))
    record.update(annotations)
    present_motifs = {item.id for item in plan.motifs if scene_id in item.scene_ids}
    if "motif_echoes" in MEMBERS[plan.tier]:
        record["motif_echoes"] = [
            item.model_dump(mode="json")
            for item in plan.motif_echoes
            if item.to_motif_id in present_motifs
        ]
    return record


class BaselineC(Pipeline[NarrativePlan]):
    roles: ClassVar[tuple[RoleName, ...]] = ("planner", "prose")

    def __init__(self, params: BaselineCParams) -> None:
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, JsonValue]) -> Self:
        return cls(BaselineCParams.model_validate(raw))

    @override
    def templates(self) -> list[TemplateNeed]:
        return [
            *planning_templates(self.params.planning()),
            TemplateNeed(SCENE_PROSE, SCENE_VARIABLES),
        ]

    @override
    def plan(self, ctx: StoryContext) -> NarrativePlan:
        try:
            original = (
                generate_plan(ctx, self.params.planning())
                if self.params.saved_plan is None or self.params.provenance is None
                else import_plan(
                    ctx,
                    Path(self.params.saved_plan),
                    Path(self.params.provenance),
                    self.params.planning(),
                )
            )
        except PlanError as error:
            raise StepFailedError("saved-plan", str(error)) from error
        _, original_files = rendered_files(original)
        ctx.keep("source-plan.json", original_files["plan.json"])
        selected = project(original, self.params.tier)
        _, files = rendered_files(selected)
        for name, content in files.items():
            ctx.keep(name, content)
        ctx.keep(
            "plan-selection.json",
            json.dumps(
                {
                    "tier": selected.tier,
                    "source_plan_sha256": plan_identity(original),
                    "plan_sha256": plan_identity(selected),
                },
                indent=2,
            )
            + "\n",
        )
        return selected

    @override
    def write(self, ctx: StoryContext, plan: NarrativePlan, *, seed: int) -> str:
        try:
            ledger = replay(plan)
        except PlanError as error:
            raise StepFailedError("ledger", str(error)) from error
        ledger_text = ledger.model_dump_json(indent=2) + "\n"
        ctx.files["ledger.json"] = ledger_text
        scenes: list[SceneSpec] = []
        for index, (scene, snapshot) in enumerate(zip(plan.scenes, ledger.scenes, strict=True)):
            inputs = {
                "scene_id": scene.id,
                "plan_sha256": ledger.plan_sha256,
                "ledger_sha256": hashlib.sha256(ledger_text.encode()).hexdigest(),
                "planned_state_before": snapshot.summary,
                "scene_record": _scene_record(plan, scene.id),
            }
            notes = json.dumps(inputs, indent=2, ensure_ascii=False)
            ctx.keep(f"{scene_step(index)}.inputs.json", notes + "\n")
            scenes.append(
                SceneSpec(
                    summary=scene.summary or scene.title,
                    beats=[beat.summary for beat in scene.beats] or [scene.summary or scene.title],
                    notes=notes,
                )
            )
        return write_scenes(ctx, scenes, seed=seed)
