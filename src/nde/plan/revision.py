"""Explicit revision authority, checked independently of provider instructions."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from nde.gitinfo import git
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.projection import CHANNELS, serialized_plan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, rendered_files, schema_json
from nde.plan.backends import DRevisionBackend, RevisionBackend
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
    # None keeps the released D operator; see nde.plan.backends.
    backend: RevisionBackend | None = None


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


def within(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def selected_units(spec: RevisionSpec) -> list[str]:
    return [*spec.mask.channels, *spec.mask.units, *(f"scenes/{s}" for s in spec.mask.scene_ids)]


def _validate_span(plan: NarrativePlan, scene_ids: list[str]) -> None:
    if not scene_ids:
        return
    positions = {scene.id: index for index, scene in enumerate(plan.scenes)}
    if any(scene_id not in positions for scene_id in scene_ids):
        raise PlanError("scene span contains unknown scene IDs")
    indices = [positions[scene_id] for scene_id in scene_ids]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise PlanError("scene span must be contiguous, distinct and in story order")


def validate_selection(plan: NarrativePlan, spec: RevisionSpec) -> None:
    result = validate_plan(plan)
    if not result.valid:
        raise PlanError("revision source is invalid: " + result.model_dump_json())
    selections = selected_units(spec)
    if not selections:
        raise PlanError("revision mask is empty")
    _validate_span(plan, spec.mask.scene_ids)
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
        if any(within(unit, lock) for lock in spec.locks):
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
        if not any(within(unit, scope) for scope in scopes)
    )
    return errors


def candidate_errors(
    before: NarrativePlan, candidate: NarrativePlan, spec: RevisionSpec
) -> list[str]:
    """The complete-candidate check shared by every backend: whole-plan validity plus scope."""
    return [
        f"{item.code} [{item.location}]: {item.message}"
        for item in validate_plan(candidate).diagnostics
        if item.severity == "error"
    ] + boundary_errors(before, candidate, spec)


def revision_templates(spec: RevisionSpec) -> list[TemplateNeed]:
    return [TemplateNeed(spec.prompt, frozenset({"brief", "plan", "revision"}))]


def boundary_evidence(
    before: NarrativePlan, after: NarrativePlan, spec: RevisionSpec
) -> dict[str, Any]:
    """Expose explicit cross-span structure; validation always checks the entire plan."""
    selected = set(spec.mask.scene_ids)
    setups = {setup.id: setup.scene_id for setup in after.setups}
    result = validate_plan(after)
    errors = boundary_errors(before, after, spec)
    return {
        "scene_ids": spec.mask.scene_ids,
        "before_sha256": identity(before),
        "after_sha256": identity(after),
        "valid": result.valid and not errors,
        "scope_diagnostics": errors,
        "validation": result.model_dump(mode="json"),
        "incoming_causal_links": [
            link.id
            for link in after.causal_links
            if link.effect_scene_id in selected and link.cause_scene_id not in selected
        ],
        "outgoing_causal_links": [
            link.id
            for link in after.causal_links
            if link.cause_scene_id in selected and link.effect_scene_id not in selected
        ],
        "crossing_payoff_links": [
            link.id
            for link in after.payoff_links
            if (setups.get(link.setup_id) in selected) != (link.payoff_scene_id in selected)
        ],
        "knowledge_events": [
            event.model_dump(mode="json") for event in [*after.reveal_links, *after.knowledge_uses]
        ],
        "constraints": [constraint.model_dump(mode="json") for constraint in after.constraints],
    }


@dataclass
class RevisionRecord:
    """One operator invocation; failed and rejected results keep the source selected."""

    name: str
    before: NarrativePlan
    spec: RevisionSpec
    first_call: int
    prompt_sha256: str
    model: dict[str, Any]
    backend: dict[str, Any] | None = None
    status: str = "failed"
    selected: NarrativePlan | None = None
    error: str | None = None

    @property
    def folder(self) -> str:
        return f"revisions/{self.name}"


def check_name(name: str) -> None:
    if not name or any(part in {"", ".", ".."} for part in name.split("/")) or "\\" in name:
        raise PlanError("invalid revision name")


def keep_selected(ctx: StoryContext, record: RevisionRecord, selected: NarrativePlan) -> None:
    record.selected = selected
    record.status = "accepted" if identity(record.before) != identity(selected) else "noop"
    _, files = rendered_files(selected)
    for filename, content in files.items():
        ctx.keep(f"{record.folder}/{filename}", content)
        ctx.keep(filename, content)


def record_revision(ctx: StoryContext, record: RevisionRecord) -> None:
    """Shared boundary, last-accepted, provenance and change-summary artifacts."""
    before, spec, folder = record.before, record.spec, record.folder
    selected = before if record.selected is None else record.selected
    ctx.keep(
        f"{folder}/boundary.json",
        json.dumps(boundary_evidence(before, selected, spec), indent=2) + "\n",
    )
    _, last = rendered_files(before if record.status == "failed" else selected)
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
        "status": record.status,
        "error": record.error,
        "revision": spec.model_dump(mode="json"),
        "before_sha256": identity(before),
        "after_sha256": identity(selected),
        "changed_units": changed_units(before, selected),
        "lock_identities": lock_ids,
        "schema_sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
        "git_sha": git(["rev-parse", "HEAD"], Path(__file__).parent) or "unknown",
        "model": record.model,
        "prompt_sha256": record.prompt_sha256,
        "seed": ctx.seed,
        "max_repairs": ctx.structured_config.max_repairs,
        "brief": ctx.brief.model_dump(mode="json"),
        "calls": [call.model_dump(mode="json") for call in ctx.state.calls[record.first_call :]],
    }
    if record.backend is not None:
        # Additive, versioned backend metadata; released D provenance omits it.
        provenance["backend"] = record.backend
    ctx.keep(f"{folder}/provenance.json", json.dumps(provenance, indent=2) + "\n")
    ctx.keep(
        f"{folder}/change-summary.md",
        "\n".join(
            [
                f"# Revision: {record.status}",
                f"Before: {identity(before)}",
                f"After: {identity(selected)}",
                *(f"- {unit}" for unit in changed_units(before, selected)),
            ]
        )
        + "\n",
    )


def revise_plan(
    ctx: StoryContext,
    before: NarrativePlan,
    spec: RevisionSpec,
    *,
    name: str = "revision",
    backend: DRevisionBackend | None = None,
) -> NarrativePlan:
    """Regenerate under an explicit mask; failures retain the source and every attempt."""
    validate_selection(before, spec)
    check_name(name)
    record = RevisionRecord(
        name=name,
        before=before,
        spec=spec,
        first_call=len(ctx.state.calls),
        prompt_sha256=ctx.templates[spec.prompt.label].sha256,
        model=ctx.roles["planner"].model_dump(mode="json"),
        backend=None if backend is None else backend.model_dump(mode="json"),
    )
    ctx.keep(f"{record.folder}/source-plan.json", before.model_dump_json(indent=2) + "\n")
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
            lambda candidate: candidate_errors(before, candidate, spec),
        )
        keep_selected(ctx, record, selected)
        return selected
    except Exception as failure:
        record.error = str(failure)
        raise
    finally:
        record_revision(ctx, record)
