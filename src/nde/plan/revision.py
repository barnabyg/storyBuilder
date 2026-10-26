"""Explicit revision authority, checked independently of provider instructions."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from nde.gitinfo import git
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.projection import CHANNELS, serialized_plan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, rendered_files, schema_json
from nde.plan.narrative import NarrativePlanningParams, request_plan
from nde.runs.config import PromptRef
from nde.stories.pipeline import Step, StoryContext, TemplateNeed

REVISION = PromptRef(template="plan_revision", version=1)
EDITABLE = CHANNELS - {"constraints"}


class RevisionMask(Strict):
    channels: list[str] = Field(default_factory=list[str])
    units: list[str] = Field(default_factory=list[str])
    scene_ids: list[str] = Field(default_factory=list[str])


class RevisionSpec(Strict):
    version: Literal["1"] = "1"
    mask: RevisionMask
    locks: list[str] = Field(default_factory=list[str])
    instruction: str = Field(min_length=1)
    prompt: PromptRef = Field(default_factory=lambda: REVISION)


class RevisionPlanningParams(NarrativePlanningParams):
    saved_plan: str
    revision: RevisionSpec


def canonical(value: Any) -> str:
    """UTF-8 JSON, sorted object keys, default JSON separators, no ASCII escaping."""
    if isinstance(value, NarrativePlan):
        value = serialized_plan(value)
    elif isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=_model_json)


def _model_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def selector_value(plan: NarrativePlan, selector: str) -> Any:
    """Resolve fields and ID-keyed records; numeric list offsets are unsupported."""
    current: Any = plan.model_dump(mode="json")
    for part in selector.split("/"):
        if isinstance(current, dict) and part in current:
            current = cast(dict[str, Any], current)[part]
        elif isinstance(current, list):
            matches: list[Any] = [
                item
                for item in cast(list[Any], current)
                if isinstance(item, dict) and cast(dict[str, Any], item).get("id") == part
            ]
            if len(matches) != 1:
                raise PlanError(f"unknown or ambiguous selector {selector!r}")
            current = matches[0]
        else:
            raise PlanError(f"unknown selector {selector!r}")
    return current


def _within(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def selected_units(spec: RevisionSpec) -> list[str]:
    return [*spec.mask.channels, *spec.mask.units, *(f"scenes/{s}" for s in spec.mask.scene_ids)]


def validate_selection(plan: NarrativePlan, spec: RevisionSpec) -> None:
    result = validate_plan(plan)
    if not result.valid:
        raise PlanError("revision source is invalid: " + result.model_dump_json())
    selections = selected_units(spec)
    if not selections:
        raise PlanError("revision mask is empty")
    if spec.mask.scene_ids:
        raise PlanError("scene spans are not supported by this operator version")
    if len(set(selections)) != len(selections) or len(set(spec.locks)) != len(spec.locks):
        raise PlanError("duplicate revision selectors")
    if any(channel not in EDITABLE for channel in spec.mask.channels):
        raise PlanError("unsupported channel selection; constraints and envelope are immutable")
    for unit in spec.mask.units:
        root = unit.split("/", 1)[0]
        if root not in EDITABLE and not (root == "scenes" and "/" in unit):
            raise PlanError(f"unsupported revision unit {unit!r}")
        if unit.endswith("/id"):
            raise PlanError("record identities cannot be selected for editing")
    for selector in [*selections, *spec.locks]:
        selector_value(plan, selector)
    for unit in selections:
        if any(_within(unit, lock) for lock in spec.locks):
            raise PlanError(f"mask/lock conflict at {unit}")


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        return {
            leaf: content
            for key, child in cast(dict[str, Any], value).items()
            for leaf, content in _flatten(child, f"{prefix}/{key}".lstrip("/")).items()
        }
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) and "id" in item for item in cast(list[Any], value))
    ):
        value = cast(list[dict[str, Any]], value)
        leaves = {f"{prefix}/@order": canonical([item["id"] for item in value])}
        for item in value:
            leaves.update(_flatten(item, f"{prefix}/{item['id']}"))
        return leaves
    return {prefix: canonical(value)}


def changed_units(before: NarrativePlan, after: NarrativePlan) -> list[str]:
    previous = _flatten(before.model_dump(mode="json"))
    current = _flatten(after.model_dump(mode="json"))
    return sorted(
        key for key in previous.keys() | current.keys() if previous.get(key) != current.get(key)
    )


def boundary_errors(before: NarrativePlan, after: NarrativePlan, spec: RevisionSpec) -> list[str]:
    errors: list[str] = []
    for name in ("id", "schema_version", "tier", "title", "characters", "sources", "constraints"):
        if canonical(getattr(before, name)) != canonical(getattr(after, name)):
            errors.append(f"revision_identity: immutable {name} changed")
    if [scene.id for scene in before.scenes] != [scene.id for scene in after.scenes]:
        errors.append("revision_identity: scene IDs, order and count must remain unchanged")
    for previous, current in zip(before.scenes, after.scenes, strict=False):
        if [b.id for b in previous.beats] != [b.id for b in current.beats]:
            errors.append(f"revision_identity: beat IDs/order/count changed in {previous.id}")
    for selector in spec.locks:
        try:
            unchanged = canonical(selector_value(before, selector)) == canonical(
                selector_value(after, selector)
            )
        except PlanError:
            unchanged = False
        if not unchanged:
            errors.append(f"revision_lock: locked subtree changed: {selector}")
    scopes = selected_units(spec)
    errors.extend(
        f"revision_scope: unselected unit changed: {unit}"
        for unit in changed_units(before, after)
        if not any(_within(unit, scope) for scope in scopes)
    )
    return errors


def revision_templates(spec: RevisionSpec) -> list[TemplateNeed]:
    return [TemplateNeed(spec.prompt, frozenset({"brief", "plan", "revision"}))]


def revise_plan(
    ctx: StoryContext, before: NarrativePlan, spec: RevisionSpec, *, name: str = "revision"
) -> NarrativePlan:
    """Regenerate under an explicit mask; failures retain the source and every attempt."""
    validate_selection(before, spec)
    if not name or any(part in {"", ".", ".."} for part in name.split("/")) or "\\" in name:
        raise PlanError("invalid revision name")
    folder = f"revisions/{name}"
    first = len(ctx.state.calls)
    status = "failed"
    selected = before
    error: str | None = None
    ctx.keep(f"{folder}/source-plan.json", before.model_dump_json(indent=2) + "\n")

    def check(candidate: NarrativePlan) -> list[str]:
        return [
            f"{item.code} [{item.location}]: {item.message}"
            for item in validate_plan(candidate).diagnostics
            if item.severity == "error"
        ] + boundary_errors(before, candidate, spec)

    try:
        selected = request_plan(
            ctx,
            Step(
                f"revision-{name}",
                "planner",
                spec.prompt,
                {
                    "brief": ctx.brief.model_dump_json(),
                    "plan": before.model_dump_json(),
                    "revision": spec.model_dump_json(),
                },
            ),
            NarrativePlan,
            check,
        )
        status = "accepted" if identity(before) != identity(selected) else "noop"
        _, files = rendered_files(selected)
        for filename, content in files.items():
            ctx.keep(f"{folder}/{filename}", content)
            ctx.keep(filename, content)
        return selected
    except Exception as failure:
        error = str(failure)
        raise
    finally:
        _, last = rendered_files(before if status == "failed" else selected)
        for filename, content in last.items():
            ctx.keep(f"{folder}/last-accepted/{filename}", content)
        lock_ids = {
            lock: {
                "before": identity(selector_value(before, lock)),
                "after": identity(selector_value(selected, lock)),
            }
            for lock in spec.locks
        }
        provenance = {
            "version": "1",
            "status": status,
            "error": error,
            "revision": spec.model_dump(mode="json"),
            "before_sha256": identity(before),
            "after_sha256": identity(selected),
            "changed_units": changed_units(before, selected),
            "lock_identities": lock_ids,
            "schema_sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
            "git_sha": git(["rev-parse", "HEAD"], Path(__file__).parent) or "unknown",
            "model": ctx.roles["planner"].model_dump(mode="json"),
            "prompt_sha256": ctx.templates[spec.prompt.label].sha256,
            "seed": ctx.seed,
            "max_repairs": ctx.structured_config.max_repairs,
            "brief": ctx.brief.model_dump(mode="json"),
            "calls": [call.model_dump(mode="json") for call in ctx.state.calls[first:]],
        }
        ctx.keep(f"{folder}/provenance.json", json.dumps(provenance, indent=2) + "\n")
        ctx.keep(
            f"{folder}/change-summary.md",
            "\n".join(
                [
                    f"# Revision: {status}",
                    f"Before: {identity(before)}",
                    f"After: {identity(selected)}",
                    *(f"- {unit}" for unit in changed_units(before, selected)),
                ]
            )
            + "\n",
        )
