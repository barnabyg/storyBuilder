"""Validated imports retain original planning attribution without charging it twice."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from nde.llm.config import ModelBinding
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan, schema_json
from nde.plan.narrative import NarrativePlanningParams
from nde.runs.records import CallRecord
from nde.schema.brief import Brief
from nde.stories.pipeline import StoryContext, derive_seed


class PlanningProvenance(Strict):
    version: Literal["1"]
    status: Literal["succeeded", "failed"]
    brief: Brief
    git_sha: str
    brief_sha256: str
    schema_sha256: str
    params: NarrativePlanningParams
    prompts: dict[str, str]
    model: ModelBinding
    seed: int
    max_repairs: int
    plan_sha256: str | None
    calls: list[CallRecord]


def import_plan(
    ctx: StoryContext,
    plan_path: Path,
    provenance_path: Path,
    params: NarrativePlanningParams,
) -> NarrativePlan:
    try:
        plan = load_plan(plan_path)
        provenance_text = provenance_path.read_text(encoding="utf-8")
        provenance = PlanningProvenance.model_validate_json(provenance_text)
        checksum = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    except (OSError, ValidationError) as error:
        raise PlanError(f"saved plan provenance: {error}") from error
    expected_prompts = {
        ref.label: ctx.templates[ref.label].sha256
        for ref in (params.skeleton_prompt, params.channels_prompt)
    }
    if provenance.status != "succeeded" or provenance.plan_sha256 != checksum:
        raise PlanError("saved plan failed or checksum does not match its provenance")
    if (
        provenance.brief != ctx.brief
        or provenance.brief_sha256
        != hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest()
        or provenance.schema_sha256 != hashlib.sha256(schema_json().encode()).hexdigest()
        or provenance.params != params
        or provenance.prompts != expected_prompts
        or provenance.model != ctx.roles["planner"]
        or provenance.seed != ctx.seed
        or provenance.max_repairs != ctx.structured_config.max_repairs
    ):
        raise PlanError("saved plan brief/schema/planning settings differ from this run")
    validation = validate_plan(plan)
    if plan.tier != "L" or not validation.valid:
        raise PlanError("saved plan must be a validated full tier-L plan")
    expected_steps = {"skeleton", *(f"channels-{scene.id}" for scene in plan.scenes)}
    if {call.step for call in provenance.calls} != expected_steps:
        raise PlanError("saved plan lacks complete attributable planning calls")
    for step in expected_steps:
        calls = [call for call in provenance.calls if call.step == step]
        if [call.attempt for call in calls] != list(range(len(calls))) or any(
            call.brief != ctx.brief.id
            or call.provider != provenance.model.provider
            or call.model != provenance.model.model
            or call.seed != derive_seed(provenance.seed, "plan", step)
            for call in calls
        ):
            raise PlanError("saved planning calls have inconsistent identities or repair sequence")
    for call in provenance.calls:
        ctx.state.calls.append(
            call.model_copy(
                update={
                    "call": f"{ctx.condition}/{ctx.brief.id}/imported-{call.step}",
                    "condition": ctx.condition,
                    "brief": ctx.brief.id,
                    "repetition": ctx.repetition,
                    "step": f"imported-{call.step}",
                    "cost_usd": 0.0,
                    "cached": True,
                    "latency_s": 0.0,
                    "imported_from": checksum,
                }
            )
        )
    ctx.keep("planning-provenance.json", provenance_text)
    ctx.keep("source-provenance.json", provenance_text)
    ctx.keep(
        "plan-import.json",
        json.dumps(
            {
                "source_plan_sha256": checksum,
                "source_plan_path": str(plan_path.resolve()),
                "source_provenance_path": str(provenance_path.resolve()),
                "source_provenance_sha256": hashlib.sha256(provenance_text.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
    )
    return plan
