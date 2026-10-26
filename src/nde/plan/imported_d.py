"""Validate D's full planning/refinement provenance before importing its token budget."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nde.llm.config import ModelBinding
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError, load_plan, schema_json
from nde.plan.coarse import coarse_templates
from nde.plan.narrative import generated_errors
from nde.plan.refinement import CONSECUTIVE_FAILURE_LIMIT, RefinementParams, refinement_templates
from nde.plan.revision import RevisionSpec, identity
from nde.runs.config import PromptRef
from nde.runs.records import CallRecord
from nde.schema.brief import Brief
from nde.stories.pipeline import StoryContext, derive_seed


class SavedIteration(Strict):
    iteration: int = Field(ge=1)
    before_sha256: str
    status: Literal["critic_failed", "no_actionable_issues", "rejected", "noop", "accepted"]
    revision_steps: list[str]
    diagnostics: list[str] = Field(default_factory=list[str])
    proposed: dict[str, Any] | None = None
    mask: RevisionSpec | None = None
    after_sha256: str | None = None


class SavedRefinement(Strict):
    status: Literal["succeeded", "failed"]
    stopping_reason: str
    iterations: list[SavedIteration]
    initial_plan_sha256: str | None = None
    selected_plan_sha256: str
    allow_critic_generator_family: bool


class DPlanningProvenance(Strict):
    version: Literal["1"]
    pipeline: Literal["baseline_d"]
    status: Literal["succeeded", "failed"]
    brief: Brief
    brief_sha256: str
    schema_sha256: str
    git_sha: str
    params: RefinementParams
    prompts: dict[str, str]
    model: ModelBinding
    seed: int
    max_repairs: int
    plan_sha256: str | None
    calls: list[CallRecord]
    calls_sha256: str | None = None
    critic_model: ModelBinding | None = None
    refinement: SavedRefinement | None = None


def _history_chain(history: SavedRefinement, params: RefinementParams, plan: NarrativePlan) -> None:
    iterations = history.iterations
    last = iterations[-1].status if iterations else None
    terminal = {
        "disabled": not iterations and not params.critic_iterations,
        "iteration_limit": len(iterations) == params.critic_iterations and last == "accepted",
        "revision_rejected": len(iterations) == params.critic_iterations and last == "rejected",
        "repeated_failure": len(iterations) >= CONSECUTIVE_FAILURE_LIMIT
        and all(record.status == "rejected" for record in iterations[-CONSECUTIVE_FAILURE_LIMIT:]),
        "critic_failed": last == "critic_failed",
        "no_actionable_issues": last == "no_actionable_issues",
        "noop": last == "noop",
    }
    if not terminal.get(history.stopping_reason, False):
        raise PlanError("saved D terminal stopping reason contradicts its complete history")
    current = history.initial_plan_sha256
    if current is None:
        if iterations:
            raise PlanError("saved D refinement lacks its initial plan identity")
        current = identity(plan)
    for record in iterations:
        if record.before_sha256 != current:
            raise PlanError("saved D refinement plan identity chain is inconsistent")
        if record.status in {"accepted", "noop"}:
            if record.after_sha256 is None or record.mask is None:
                raise PlanError("saved D accepted revision lacks its mask or output identity")
            if (record.after_sha256 == current) != (record.status == "noop"):
                raise PlanError("saved D revision status contradicts its before/after identities")
            current = record.after_sha256
        elif record.after_sha256 is not None:
            raise PlanError("saved D rejected revision claims a selected output")
        if record is not iterations[-1] and record.status in {
            "noop",
            "critic_failed",
            "no_actionable_issues",
        }:
            raise PlanError("saved D history continues after a terminal iteration")
    if current != identity(plan):
        raise PlanError("saved D refinement history does not lead to the selected plan")


def _expected_steps(plan: NarrativePlan, provenance: DPlanningProvenance) -> dict[str, PromptRef]:
    params = provenance.params
    expected = {
        "intentions": params.intentions_prompt,
        "skeleton": params.skeleton_prompt,
        **{f"channels-{scene.id}": params.channels_prompt for scene in plan.scenes},
    }
    history = provenance.refinement
    if history is None:
        if params.critic_iterations:
            raise PlanError("saved D plan lacks refinement history")
        return expected
    if (
        history.status != "succeeded"
        or history.selected_plan_sha256 != identity(plan)
        or history.allow_critic_generator_family != params.allow_critic_generator_family
    ):
        raise PlanError("saved D refinement did not select this successful plan")
    iterations = history.iterations
    if len(iterations) > params.critic_iterations or (params.critic_iterations and not iterations):
        raise PlanError("saved D critic history violates configured iteration bounds")
    _history_chain(history, params, plan)
    for number, record in enumerate(iterations, 1):
        if record.iteration != number:
            raise PlanError("saved D iteration sequence is incomplete")
        expected[f"critic-{number}"] = params.critic_prompt
        revisions = record.revision_steps
        needs_revision = record.status in {"accepted", "noop", "rejected"}
        if revisions != ([f"revision-critic-{number}"] if needs_revision else []):
            raise PlanError("saved D revision attribution is incomplete")
        if needs_revision:
            expected[f"revision-critic-{number}"] = params.revision_prompt
    return expected


def _check_calls(ctx: StoryContext, plan: NarrativePlan, provenance: DPlanningProvenance) -> None:
    if provenance.calls_sha256 != identity(
        [call.model_dump(mode="json") for call in provenance.calls]
    ):
        raise PlanError("saved D planning/refinement call evidence changed")
    expected = _expected_steps(plan, provenance)
    if {call.step for call in provenance.calls} != set(expected):
        raise PlanError("saved D plan lacks complete planning/critic/regeneration calls")
    for step, prompt in expected.items():
        calls = [call for call in provenance.calls if call.step == step]
        binding = provenance.critic_model if step.startswith("critic-") else provenance.model
        if binding is None:
            raise PlanError("saved D critic model is missing")
        for call in calls:
            price = ctx.gateway.pricing.cost(binding.label, call.input_tokens, call.output_tokens)
            if (
                call.input_tokens < 0
                or call.output_tokens < 0
                or not math.isfinite(call.cost_usd)
                or not math.isfinite(call.token_cost_usd)
                or not 0 <= call.cost_usd <= call.token_cost_usd
                or not math.isclose(call.token_cost_usd, price, abs_tol=1e-12, rel_tol=1e-9)
            ):
                raise PlanError("saved D token usage/list-price accounting is invalid or stale")
        if not 1 <= len(calls) <= provenance.max_repairs + 1:
            raise PlanError("saved D repair calls violate configured bounds")
        if [call.attempt for call in calls] != list(range(len(calls))) or any(
            call.brief != provenance.brief.id
            or call.provider != binding.provider
            or call.model != binding.model
            or call.template != prompt.template
            or call.template_version != prompt.version
            or call.seed != derive_seed(provenance.seed, "plan", step)
            for call in calls
        ):
            raise PlanError("saved D calls have inconsistent identities or repair sequence")


def import_d_plan(
    ctx: StoryContext, plan_path: Path, provenance_path: Path, params: RefinementParams
) -> NarrativePlan:
    plan = load_plan(plan_path)
    text = provenance_path.read_text(encoding="utf-8")
    provenance = DPlanningProvenance.model_validate_json(text)
    checksum = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    expected_prompts = {
        need.prompt.label: ctx.templates[need.prompt.label].sha256
        for need in [*coarse_templates(params), *refinement_templates(params)]
    }
    if provenance.status != "succeeded" or provenance.plan_sha256 != checksum:
        raise PlanError("saved D plan failed or checksum differs from its provenance")
    if (
        provenance.brief != ctx.brief
        or provenance.params != params
        or provenance.brief_sha256
        != hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest()
        or provenance.schema_sha256 != hashlib.sha256(schema_json().encode()).hexdigest()
        or provenance.prompts != expected_prompts
        or provenance.model != ctx.roles["planner"]
        or provenance.seed != ctx.seed
        or provenance.max_repairs != ctx.structured_config.max_repairs
        or provenance.critic_model != (ctx.roles["critic"] if params.critic_iterations else None)
    ):
        raise PlanError("saved D brief/schema/planning/refinement settings differ from this run")
    if generated_errors(plan, ctx):
        raise PlanError("saved D plan does not pass generated-plan validation")
    _check_calls(ctx, plan, provenance)
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
    ctx.keep("planning-provenance.json", text)
    ctx.keep("source-provenance.json", text)
    if provenance.refinement is not None:
        ctx.keep("refinement.json", provenance.refinement.model_dump_json(indent=2) + "\n")
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
    return plan
