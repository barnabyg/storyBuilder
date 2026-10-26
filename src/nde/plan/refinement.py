"""Bounded critic suggestions become explicit revision authority, never gate scores."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from nde.llm.config import ModelBinding
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, rendered_files
from nde.plan.coarse import CoarseParams
from nde.plan.narrative import request_plan
from nde.plan.revision import (
    REVISION,
    RevisionMask,
    RevisionSpec,
    identity,
    revise_plan,
    revision_templates,
    selected_units,
    validate_selection,
)
from nde.runs.config import ConfigError, PromptRef, RoleName
from nde.stories.pipeline import Step, StepFailedError, StoryContext, TemplateNeed

CRITIC = PromptRef(template="plan_critic", version=1)


class RefinementParams(CoarseParams):
    critic_iterations: int = Field(default=0, ge=0, le=10)
    critic_prompt: PromptRef = Field(default_factory=lambda: CRITIC)
    revision_prompt: PromptRef = Field(default_factory=lambda: REVISION)
    revision_locks: list[str] = Field(default_factory=list[str])
    allow_critic_generator_family: bool = False

    def coarse(self) -> CoarseParams:
        return CoarseParams.model_validate(self.model_dump(include=set(CoarseParams.model_fields)))


class CriticIssue(Strict):
    plan_sha256: str
    mask: RevisionMask
    proposed_change: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class CriticReview(Strict):
    issues: list[CriticIssue] = Field(default_factory=list[CriticIssue], max_length=20)


def check_critic_families(params: RefinementParams, roles: Mapping[RoleName, ModelBinding]) -> None:
    if not params.critic_iterations:
        return
    if any(roles[name].family is None for name in ("critic", "planner", "prose")):
        raise ConfigError(
            "critic family separation requires family metadata on critic/planner/prose"
        )
    if not params.allow_critic_generator_family and roles["critic"].family in {
        roles["planner"].family,
        roles["prose"].family,
    }:
        raise ConfigError("critic family must differ from planner and prose families")


def refinement_templates(params: RefinementParams) -> list[TemplateNeed]:
    if not params.critic_iterations:
        return []
    return [
        TemplateNeed(
            params.critic_prompt, frozenset({"brief", "plan", "plan_sha256", "diagnostics"})
        ),
        *revision_templates(
            RevisionSpec(mask=RevisionMask(), instruction="prepare", prompt=params.revision_prompt)
        ),
    ]


def _overlap(first: str, second: str) -> bool:
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def _accepted_issues(
    review: CriticReview, plan: NarrativePlan, params: RefinementParams
) -> tuple[list[tuple[CriticIssue, RevisionSpec]], list[str]]:
    accepted: list[tuple[CriticIssue, RevisionSpec]] = []
    errors: list[str] = []
    for index, issue in enumerate(review.issues):
        if issue.plan_sha256 != identity(plan):
            errors.append(f"issue {index}: stale plan identity")
            continue
        spec = RevisionSpec(
            mask=issue.mask,
            locks=["constraints", *params.revision_locks],
            instruction=issue.proposed_change + "\nRationale: " + issue.rationale,
            prompt=params.revision_prompt,
        )
        try:
            validate_selection(plan, spec)
        except PlanError as failure:
            errors.append(f"issue {index}: unsupported or locked target: {failure}")
            continue
        conflicts = [
            prior
            for _, prior in accepted
            if any(_overlap(a, b) for a in selected_units(spec) for b in selected_units(prior))
        ]
        if conflicts:
            errors.append(
                f"issue {index}: contradictory/overlapping target; reject overlapping batch"
            )
            return [], errors
        accepted.append((issue, spec))
    return accepted, errors


def refine_plan(
    ctx: StoryContext, initial: NarrativePlan, params: RefinementParams
) -> NarrativePlan:
    selected = initial
    iterations: list[dict[str, Any]] = []
    first = len(ctx.state.calls)
    stopping = "iteration_limit" if params.critic_iterations else "disabled"
    status = "failed"
    try:
        for iteration in range(params.critic_iterations):
            name = f"critic-{iteration + 1}"
            record: dict[str, Any] = {
                "iteration": iteration + 1,
                "before_sha256": identity(selected),
                "status": "interrupted",
                "revision_steps": [],
            }
            iterations.append(record)
            findings = validate_plan(selected)
            try:
                review = request_plan(
                    ctx,
                    Step(
                        name,
                        "critic",
                        params.critic_prompt,
                        {
                            "brief": ctx.brief.model_dump_json(),
                            "plan": selected.model_dump_json(),
                            "plan_sha256": identity(selected),
                            "diagnostics": findings.model_dump_json(),
                        },
                    ),
                    CriticReview,
                    lambda _: [],
                )
            except StepFailedError as failure:
                record.update(status="critic_failed", diagnostics=[str(failure)])
                stopping = "critic_failed"
                break
            ctx.keep(
                f"refinement/{iteration + 1}/issues.json", review.model_dump_json(indent=2) + "\n"
            )
            actionable, rejected = _accepted_issues(review, selected, params)
            record.update(proposed=review.model_dump(mode="json"), diagnostics=rejected)
            if not actionable:
                record["status"] = "no_actionable_issues"
                stopping = "no_actionable_issues"
                break
            # One atomic candidate covers the disjoint targets from this accepted review.
            combined = RevisionSpec(
                mask=RevisionMask(
                    channels=[c for _, spec in actionable for c in spec.mask.channels],
                    units=[u for _, spec in actionable for u in spec.mask.units],
                    scene_ids=[s for _, spec in actionable for s in spec.mask.scene_ids],
                ),
                locks=["constraints", *params.revision_locks],
                prompt=params.revision_prompt,
                instruction="\n\n".join(spec.instruction for _, spec in actionable),
            )
            record["mask"] = combined.model_dump(mode="json")
            record["revision_steps"] = [f"revision-critic-{iteration + 1}"]
            try:
                candidate = revise_plan(ctx, selected, combined, name=name)
            except (StepFailedError, PlanError) as failure:
                record.update(status="rejected", diagnostics=[*rejected, str(failure)])
                stopping = (
                    "repeated_failure"
                    if len(iterations) > 1 and iterations[-2]["status"] == "rejected"
                    else "revision_rejected"
                )
                if stopping == "repeated_failure":
                    break
                continue
            record["after_sha256"] = identity(candidate)
            record["status"] = "noop" if identity(selected) == identity(candidate) else "accepted"
            selected = candidate
            if record["status"] == "noop":
                stopping = "noop"
                break
        status = "succeeded"
        return selected
    finally:
        history = {
            "status": status,
            "stopping_reason": stopping if status == "succeeded" else "interrupted",
            "iterations": iterations,
            "selected_plan_sha256": identity(selected),
            "allow_critic_generator_family": params.allow_critic_generator_family,
        }
        _save_refinement(ctx, selected, params, first, history)


def _save_refinement(
    ctx: StoryContext,
    selected: NarrativePlan,
    params: RefinementParams,
    first: int,
    history: dict[str, Any],
) -> None:
    _, files = rendered_files(selected)
    for name, content in files.items():
        ctx.keep(name, content)
    ctx.keep("refinement.json", json.dumps(history, indent=2) + "\n")
    provenance = json.loads(ctx.files["planning-provenance.json"])
    provenance.update(
        params=params.model_dump(mode="json"),
        status=history["status"],
        plan_sha256=hashlib.sha256(ctx.files["plan.json"].encode()).hexdigest(),
        refinement=history,
        critic_model=None
        if not params.critic_iterations
        else ctx.roles["critic"].model_dump(mode="json"),
    )
    provenance["calls"].extend(call.model_dump(mode="json") for call in ctx.state.calls[first:])
    provenance["prompts"].update(
        {
            need.prompt.label: ctx.templates[need.prompt.label].sha256
            for need in refinement_templates(params)
        }
    )
    ctx.keep("planning-provenance.json", json.dumps(provenance, indent=2) + "\n")
