"""Validate E's complete conditional and hybrid provenance before reusing its plan.

An import binds the same brief, schema, conditions, guidance, limits, planner
binding and an identical conditional checkpoint. The learned proposal, every
generation attempt, the hybrid completion and its calls must form one
consistent chain to the saved plan. The original local work and provider calls
are attributed again, once, so reuse never makes prior work or costs disappear.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from nde.conditional.completion import (
    PROMPT_ID,
    PROMPT_NAME,
    PROMPT_VERSION,
    CompletionRequest,
    apply_fields,
)
from nde.conditional.generation import PROPOSAL_KIND, Generation, completion_fields
from nde.corpus.records_io import identity as record_identity
from nde.narrative.ledger import plan_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan, schema_json
from nde.plan.conditional import (
    PIPELINE,
    ConditionalPlanningParams,
    EPlanningProvenance,
    brief_conditions,
    checked_checkpoint,
    completion_seed,
    generation_seed,
    keep_plan,
    local_work,
)
from nde.plan.revision import identity
from nde.stories.pipeline import StoryContext
from nde.stories.resources import COMPLETION_STEP, IMPORTED


def _sealed(record: dict[str, Any] | None, kind: str) -> dict[str, Any]:
    if (
        record is None
        or record.get("version") != "1"
        or record.get("id") != record_identity(record)
        or record.get("kind") != kind
    ):
        raise PlanError(f"saved E {kind} record is missing or changed")
    return record


def _check_generation(
    ctx: StoryContext,
    provenance: EPlanningProvenance,
    checkpoint: dict[str, Any],
    params: ConditionalPlanningParams,
) -> dict[str, Any]:
    """The learned proposal and every attempt that led to it; returns the proposal."""
    conditions = brief_conditions(ctx.brief, params.condition_fields).model_dump(mode="json")
    if provenance.generation_request is None or provenance.generation is None:
        raise PlanError("saved E plan lacks its generation request or report")
    request = Generation.model_validate(provenance.generation_request)
    expected = Generation(
        checkpoint=request.checkpoint,
        conditions=brief_conditions(ctx.brief, params.condition_fields),
        locks=params.locks,
        seed=generation_seed(ctx.seed),
        guidance_scale=params.guidance_scale,
        max_attempts=params.max_attempts,
        max_tokens=params.max_tokens,
        max_attention_cells=params.max_attention_cells,
        max_seconds=params.max_seconds,
    )
    if request != expected:
        raise PlanError("saved E generation conditions/guidance/locks/limits differ from this run")
    proposal = _sealed(provenance.proposal, PROPOSAL_KIND)
    plan = NarrativePlan.model_validate(proposal["plan"])
    if (
        proposal.get("status") != "structural-proposal"
        or proposal.get("checkpoint_id") != checkpoint["id"]
        or proposal.get("conditions") != conditions
        or proposal.get("locks") != params.locks
        or sorted(proposal.get("completion_fields", [])) != completion_fields(plan, params.locks)
        or not validate_plan(plan).valid
    ):
        raise PlanError("saved E proposal does not match this checkpoint, conditions or locks")
    report = provenance.generation
    rows: list[dict[str, Any]] = report.get("attempts", [])
    if (
        report.get("status") != "structural-proposal"
        or report.get("proposal_id") != proposal["id"]
        or not 1 <= len(rows) <= params.max_attempts
        or proposal.get("selected_attempt") != len(rows)
        or [row.get("status") for row in rows] != ["rejected"] * (len(rows) - 1) + ["selected"]
        or [row.get("seed") for row in rows] != [request.seed + i for i in range(len(rows))]
        or any(
            row["candidate"].get("checkpoint_id") != checkpoint["id"]
            or row["candidate"].get("guidance_scale") != params.guidance_scale
            or row["candidate"].get("conditions") != conditions
            for row in rows
        )
    ):
        raise PlanError("saved E generation attempts do not lead to the selected proposal")
    if report["work"]["tokens"] != sum(
        row["candidate"]["work"]["tokens"] for row in rows
    ) or report["work"]["attention_cells"] != sum(
        row["candidate"]["work"]["attention_cells"] for row in rows
    ):
        raise PlanError("saved E generation work differs from its attempts")
    rows_json = [record.model_dump(mode="json") for record in provenance.local_work]
    attributed = local_work(ctx, checkpoint, request, report).model_dump(
        exclude={"call", "condition", "repetition"}
    )
    if (
        provenance.local_work_sha256 != identity(rows_json)
        or len(provenance.local_work) != 1
        or provenance.local_work[0].model_dump(exclude={"call", "condition", "repetition"})
        != attributed
    ):
        raise PlanError("saved E local model work is missing, partial or changed")
    return proposal


def _check_completion(
    ctx: StoryContext,
    provenance: EPlanningProvenance,
    proposal: dict[str, Any],
    plan: NarrativePlan,
    params: ConditionalPlanningParams,
) -> dict[str, Any]:
    """The hybrid completion filled exactly the proposal's manifest to give this plan."""
    completion = _sealed(provenance.completion, "conditional-text-completion")
    request = CompletionRequest.model_validate(completion.get("config"))
    attempts: list[dict[str, Any]] = completion.get("attempts", [])
    manifest = sorted(proposal["completion_fields"])
    selected: dict[str, str] = completion.get("selected_fields", {})
    if (
        completion.get("status") != "complete"
        or completion.get("proposal_id") != proposal["id"]
        or completion.get("prompt_id") != PROMPT_ID
        or sorted(request.fields) != manifest
        or sorted(selected) != manifest
        or request.model != ctx.roles["planner"]
        or request.seed != completion_seed(ctx.seed)
        or request.max_repairs != ctx.structured_config.max_repairs
        or request.max_tokens != params.completion_max_tokens
        or not 1 <= len(attempts) <= request.max_repairs + 1
        or [row.get("status") for row in attempts]
        != ["rejected"] * (len(attempts) - 1) + ["accepted"]
    ):
        raise PlanError("saved E hybrid completion is incomplete or differs from this run")
    try:
        completed = apply_fields(NarrativePlan.model_validate(proposal["plan"]), selected)
    except ValueError as error:
        raise PlanError(f"saved E completion is not authorised: {error}") from error
    if plan_identity(completed) != plan_identity(plan):
        raise PlanError("saved E plan is not the completed learned proposal")
    calls = provenance.calls
    if len(completion.get("calls", [])) != len(calls) or completion.get("tokens") != sum(
        call.input_tokens + call.output_tokens for call in calls
    ):
        raise PlanError("saved E completion calls differ from its attributed calls")
    return completion


def _check_calls(ctx: StoryContext, provenance: EPlanningProvenance) -> None:
    rows = [call.model_dump(mode="json") for call in provenance.calls]
    if provenance.calls_sha256 != identity(rows):
        raise PlanError("saved E completion call evidence changed")
    binding = provenance.model
    seed = completion_seed(provenance.seed)
    for index, call in enumerate(provenance.calls):
        price = ctx.gateway.pricing.cost(binding.label, call.input_tokens, call.output_tokens)
        if (
            call.step != COMPLETION_STEP
            or call.attempt != index
            or call.brief != provenance.brief.id
            or call.provider != binding.provider
            or call.model != binding.model
            or call.template != PROMPT_NAME
            or call.template_version != PROMPT_VERSION
            or call.seed != seed
            or call.input_tokens < 0
            or call.output_tokens < 0
            or not math.isfinite(call.cost_usd)
            or not 0 <= call.cost_usd <= call.token_cost_usd
            or not math.isclose(call.token_cost_usd, price, abs_tol=1e-12, rel_tol=1e-9)
        ):
            raise PlanError("saved E completion calls have inconsistent identities or accounting")


def import_e_plan(
    ctx: StoryContext, plan_path: Path, provenance_path: Path, params: ConditionalPlanningParams
) -> NarrativePlan:
    plan = load_plan(plan_path)
    text = provenance_path.read_text(encoding="utf-8")
    provenance = EPlanningProvenance.model_validate_json(text)
    checksum = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if provenance.status != "succeeded" or provenance.plan_sha256 != checksum:
        raise PlanError("saved E plan failed or checksum differs from its provenance")
    checkpoint = checked_checkpoint(params)
    conditions = brief_conditions(ctx.brief, params.condition_fields)
    if (
        provenance.brief != ctx.brief
        or provenance.brief_sha256
        != hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest()
        or provenance.schema_sha256 != hashlib.sha256(schema_json().encode()).hexdigest()
        or provenance.params.model_dump(exclude={"checkpoint"})
        != params.model_dump(exclude={"checkpoint"})
        or provenance.model != ctx.roles["planner"]
        or provenance.seed != ctx.seed
        or provenance.max_repairs != ctx.structured_config.max_repairs
        or provenance.completion_prompt_id != PROMPT_ID
        or provenance.conditions != conditions.model_dump(mode="json")
    ):
        raise PlanError("saved E brief/schema/conditions/planning settings differ from this run")
    if provenance.checkpoint is None or provenance.checkpoint.get("id") != checkpoint["id"]:
        raise PlanError("saved E plan used a different conditional checkpoint")
    proposal = _check_generation(ctx, provenance, checkpoint, params)
    completion = _check_completion(ctx, provenance, proposal, plan, params)
    _check_calls(ctx, provenance)
    for call in provenance.calls:
        ctx.state.calls.append(
            call.model_copy(
                update={
                    "call": f"{ctx.condition}/{ctx.brief.id}/{IMPORTED}{call.step}",
                    "condition": ctx.condition,
                    "brief": ctx.brief.id,
                    "repetition": ctx.repetition,
                    "step": f"{IMPORTED}{call.step}",
                    "cost_usd": 0.0,
                    "cached": True,
                    "latency_s": 0.0,
                    "imported_from": checksum,
                }
            )
        )
    for record in provenance.local_work:
        ctx.state.local_work.append(
            record.model_copy(
                update={
                    "call": f"{ctx.condition}/{ctx.brief.id}/{IMPORTED}{record.step}",
                    "condition": ctx.condition,
                    "brief": ctx.brief.id,
                    "repetition": ctx.repetition,
                    "step": f"{IMPORTED}{record.step}",
                    "imported_from": checksum,
                }
            )
        )
    ctx.keep("planning-provenance.json", text)
    ctx.keep("source-provenance.json", text)
    ctx.keep(
        "plan-import.json",
        json.dumps(
            {
                "source_plan_sha256": checksum,
                "source_plan_path": str(plan_path.resolve()),
                "source_provenance_path": str(provenance_path.resolve()),
                "source_provenance_sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
    )
    keep_plan(
        ctx,
        plan,
        {
            "pipeline": PIPELINE,
            "source": "imported",
            "checkpoint_id": checkpoint["id"],
            "proposal_id": proposal["id"],
            "completion_id": completion["id"],
        },
    )
    return plan
