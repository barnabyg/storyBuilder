"""Candidate E inpainting behind the shared D revision mask/lock/boundary contract."""

import copy
import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError

from nde.conditional.completion import TextCompletion, replace_text
from nde.conditional.data import Conditions
from nde.conditional.generation import scientific_authority
from nde.conditional.layout import (
    INTERPRETIVE_FIELDS,
    PLACEHOLDER,
    TEXT_FIELDS,
    Capacity,
    textless,
    validate_layout,
)
from nde.conditional.training import (
    TrainingConfig,
    candidate_cost,
    inspect_checkpoint,
    sample_candidate,
)
from nde.corpus.channels import CHANNEL_FIELDS
from nde.corpus.records_io import identity as record_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError
from nde.plan.backends import ERevisionBackend
from nde.plan.narrative import request_plan
from nde.plan.revision import (
    RevisionRecord,
    RevisionSpec,
    candidate_errors,
    check_name,
    identity,
    keep_selected,
    record_revision,
    selected_units,
    validate_selection,
    within,
)
from nde.reconstruction.codec import (
    BYTE_OFFSET,
    EOS,
    MASK,
    PAD,
    canonical_json,
    detokenize,
    encode_plan,
    tokenize,
)
from nde.reconstruction.model import runtime
from nde.stories.pipeline import Step, StepFailedError, StoryContext, TemplateNeed, derive_seed

SPECIFICATION = {
    "name": "nde-conditional-inpainting",
    "version": "1",
    "observation": "source plan as the checkpoint's textless canonical layout; clamped",
    "mask": "serialized values of selected, unlocked, non-identity structural leaves",
    "width": "masked values keep their serialized width; represented records keep identity/order",
    "text": "selected unlocked text leaves completed by the planner role through the gateway",
    "check": "complete candidate: whole-plan validation plus the D scope/lock/identity boundary",
}
INPAINTING_ID = record_identity(SPECIFICATION)
IMMUTABLE_ROOTS = frozenset({"id", "schema_version", "tier", "title", "characters", "sources"})
REPRESENTED_POLICY = frozenset({*CHANNEL_FIELDS, "goal", "belief"})
type ViewPath = tuple[str | int, ...]


class CandidateError(ValueError):
    """A learned candidate cannot be accepted under the shared revision contract."""


@dataclass(frozen=True)
class Leaf:
    """One scalar in the E view: its plan address, scope selector and output tokens."""

    address: ViewPath
    selector: str
    start: int
    end: int
    reference: bool
    text: bool


@dataclass(frozen=True)
class Observation:
    view: dict[str, Any]
    tokens: list[int]
    leaves: dict[ViewPath, Leaf]
    editable: list[ViewPath]
    text_fields: list[str]
    canonical_ids: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "observation_sha256": identity(self.view),
            "masked_leaves": [self.leaves[path].selector for path in self.editable],
            "masked_paths": [list(path) for path in self.editable],
            "masked_tokens": self.tokens.count(MASK),
            "text_fields": self.text_fields,
        }


def _serialize(value: Any, path: ViewPath, spans: dict[ViewPath, tuple[int, int]]) -> bytes:
    """`canonical_json` bytes, recording the byte span of every scalar leaf."""
    if isinstance(value, dict | list):
        container = cast(dict[str, Any] | list[Any], value)
        items: Iterator[tuple[str | int, Any]] = (
            ((key, container[key]) for key in sorted(container))
            if isinstance(container, dict)
            else enumerate(container)
        )
        out = bytearray(b"{" if isinstance(container, dict) else b"[")
        for index, (key, child) in enumerate(items):
            out += b"," if index else b""
            if isinstance(key, str):
                out += canonical_json(key).encode("utf-8") + b":"
            offset = len(out)
            nested: dict[ViewPath, tuple[int, int]] = {}
            out += _serialize(child, (*path, key), nested)
            spans.update({p: (offset + s, offset + e) for p, (s, e) in nested.items()})
        out += b"}" if isinstance(container, dict) else b"]"
        return bytes(out)
    data = canonical_json(value).encode("utf-8")
    spans[path] = (0, len(data))
    return data


def _addresses(
    view: Any, plan: Any, path: ViewPath, address: ViewPath, ids: dict[str, str]
) -> Iterator[tuple[ViewPath, ViewPath]]:
    """Pair each view scalar with its source-plan location (record IDs, not positions)."""
    if isinstance(view, dict):
        for key, child in cast(dict[str, Any], view).items():
            yield from _addresses(
                child, cast(dict[str, Any], plan)[key], (*path, key), (*address, key), ids
            )
    elif isinstance(view, list):
        rows = cast(list[Any], plan)
        for index, child in enumerate(cast(list[Any], view)):
            if isinstance(child, dict) and "id" in child:
                original = ids[cast(dict[str, str], child)["id"]]
                [row] = [row for row in rows if row.get("id") == original]
                yield from _addresses(child, row, (*path, index), (*address, original), ids)
            else:
                yield from _addresses(child, rows[index], (*path, index), (*address, index), ids)
    else:
        yield path, address


def _selector(address: ViewPath) -> str:
    """Scalar list elements share their list's selector; offsets are not selectors."""
    parts: list[str] = []
    for part in address:
        if isinstance(part, int):
            break
        parts.append(part)
    return "/".join(parts)


def _key(path: ViewPath) -> str:
    return next((part for part in reversed(path) if isinstance(part, str)), "")


def _authorised(selector: str, spec: RevisionSpec) -> bool:
    return (
        selector.split("/", 1)[0] not in IMMUTABLE_ROOTS
        and any(within(selector, scope) for scope in selected_units(spec))
        and not any(within(selector, lock) for lock in spec.locks)
    )


def text_fields(plan: NarrativePlan, spec: RevisionSpec) -> list[str]:
    """Selected unlocked descriptive text: the hybrid completion manifest."""
    found: list[str] = []

    def visit(value: Any, prefix: str, collection: str, key: str) -> None:
        if isinstance(value, dict):
            row = cast(dict[str, Any], value)
            for name, child in row.items():
                if name != "evidence":
                    visit(child, f"{prefix}/{name}", collection, name)
        elif isinstance(value, list):
            items = cast(list[Any], value)
            for index, child in enumerate(items):
                if isinstance(child, dict):
                    # Text is addressable only through stable record IDs, never list offsets.
                    identifier = cast(dict[str, Any], child).get("id")
                    if identifier is not None:
                        visit(child, f"{prefix}/{identifier}", collection, key)
                else:
                    visit(child, f"{prefix}/{index}", collection, key)
        elif isinstance(value, str) and (
            key in TEXT_FIELDS
            or (collection in INTERPRETIVE_FIELDS and key in {"value", "alternatives"})
        ):
            found.append(prefix)

    for root, value in plan.model_dump(mode="json").items():
        if root not in IMMUTABLE_ROOTS | {"constraints"}:
            visit(value, root, root, root)
    return [selector for selector in found if _authorised(_selector_of(selector), spec)]


def _selector_of(selector: str) -> str:
    parts = selector.split("/")
    return "/".join(parts[:-1]) if parts[-1].isdigit() else selector


def observe(
    plan: NarrativePlan, spec: RevisionSpec, backend: ERevisionBackend, checkpoint: dict[str, Any]
) -> Observation:
    """The source as E's registered layout, masked only where the shared contract permits."""
    config = TrainingConfig.model_validate(checkpoint["binding"]["config"])
    policy = backend.target_policy
    if set(policy) - REPRESENTED_POLICY or policy.get("plot", "excluded") == "excluded":
        raise PlanError("E target policy must name registered channels and include plot")
    encoded = encode_plan(plan, dict[str, str](policy), {})
    view = cast(dict[str, Any], textless(encoded["view"]))
    validate_layout(view, Capacity.model_validate(config.capacity))
    ids = {canonical: original for original, canonical in encoded["original_ids"].items()}
    spans: dict[ViewPath, tuple[int, int]] = {}
    if _serialize(view, (), spans) != canonical_json(view).encode("utf-8"):
        raise PlanError("E layout serialization differs from the registered codec")
    tokens = tokenize(view)
    tokens += [PAD] * (config.capacity.max_tokens - len(tokens))
    leaves: dict[ViewPath, Leaf] = {}
    source = plan.model_dump(mode="json")
    for path, address in _addresses(view, source, (), (), ids):
        start, end = spans[path]
        key = _key(path)
        leaves[path] = Leaf(
            address=address,
            selector=_selector(address),
            start=1 + start,
            end=1 + end,
            reference=key == "id" or key.endswith(("_id", "_ids")),
            text=_value(view, path) == PLACEHOLDER,
        )
    editable = [
        path
        for path, leaf in leaves.items()
        if _key(path) != "id" and not leaf.text and _authorised(leaf.selector, spec)
    ]
    for path in editable:
        tokens[leaves[path].start : leaves[path].end] = [MASK] * (
            leaves[path].end - leaves[path].start
        )
    return Observation(view, tokens, leaves, editable, text_fields(plan, spec), ids)


def _value(value: Any, path: ViewPath) -> Any:
    for part in path:
        value = value[part]
    return value


def _assign(content: dict[str, Any], address: ViewPath, value: Any) -> None:
    current: Any = content
    for part in address[:-1]:
        if isinstance(current, list) and isinstance(part, str):
            [current] = [row for row in cast(list[Any], current) if row.get("id") == part]
        else:
            current = cast(Any, current)[part]
    current[address[-1]] = value


def _skeleton(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _skeleton(child, name) for name, child in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_skeleton(child, key) for child in cast(list[Any], value)]
    return value if key == "id" else None


def restore(tokens: list[int], observation: Observation, before: NarrativePlan) -> dict[str, Any]:
    """Map a complete learned layout onto the source; text authority stays pending completion."""
    if EOS not in tokens:
        raise CandidateError("learned inpainting did not terminate with EOS")
    try:
        view = detokenize(tokens[: tokens.index(EOS) + 1])
    except (ValueError, UnicodeDecodeError) as error:
        raise CandidateError(f"learned inpainting is malformed: {error}") from error
    if _skeleton(view) != _skeleton(observation.view):
        raise CandidateError(
            "revision_identity: E changed represented record identities, order, counts or fields"
        )
    content = before.model_dump(mode="json")
    for path, leaf in observation.leaves.items():
        value = _value(view, path)
        if value != _value(observation.view, path):
            if leaf.reference and isinstance(value, str):
                # Unknown generated identities stay literal and fail whole-plan validation.
                value = observation.canonical_ids.get(value, value)
            _assign(content, leaf.address, value)
    for selector in observation.text_fields:
        replace_text(content, selector, PLACEHOLDER)
    return content


def _fill(structure: NarrativePlan, fields: dict[str, str]) -> NarrativePlan:
    content = copy.deepcopy(structure.model_dump(mode="json"))
    for selector, value in fields.items():
        replace_text(content, selector, value)
    return NarrativePlan.model_validate(content)


def inpainting_templates(backend: ERevisionBackend) -> list[TemplateNeed]:
    return [TemplateNeed(backend.text_prompt, frozenset({"brief", "plan", "revision", "fields"}))]


def prepare_inpainting(
    plan: NarrativePlan, spec: RevisionSpec, backend: ERevisionBackend
) -> tuple[Observation, dict[str, Any]]:
    """Reject unsupported authority, runtime, checkpoint or evidence before any call."""
    validate_selection(plan, spec)
    runtime()
    checkpoint = inspect_checkpoint(backend.checkpoint)
    scientific_authority(checkpoint)
    candidate_cost(
        checkpoint, Conditions.model_validate(backend.conditions), backend.guidance_scale
    )
    observation = observe(plan, spec, backend, checkpoint)
    if not observation.editable and not observation.text_fields:
        raise PlanError("E cannot represent any selected structural or text unit")
    return observation, checkpoint


@dataclass
class _Work:
    tokens: int = 0
    attention_cells: int = 0
    passes: int = 0
    seconds: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    selected_attempt: int | None = None
    status: str = "pending"


@dataclass(frozen=True)
class _Run:
    ctx: StoryContext
    record: RevisionRecord
    backend: ERevisionBackend
    observation: Observation
    checkpoint: dict[str, Any]


def _attempt(run: _Run, work: _Work, number: int, deadline: float) -> NarrativePlan | None:
    backend, conditions = run.backend, Conditions.model_validate(run.backend.conditions)
    tokens, cells = candidate_cost(run.checkpoint, conditions, backend.guidance_scale)
    if (
        time.monotonic() >= deadline
        or work.tokens + tokens > backend.max_tokens
        or work.attention_cells + cells > backend.max_attention_cells
    ):
        work.status = "resource-stop"
        raise StepFailedError(f"revision-{run.record.name}", "E inpainting allowance exhausted")
    seed = derive_seed(run.ctx.seed, "inpainting", run.record.name, number)
    row: dict[str, Any] = {
        "attempt": number,
        "seed": seed,
        "status": "reserved",
        "reserved_work": {"tokens": tokens, "attention_cells": cells},
        "diagnostics": [],
    }
    work.attempts.append(row)
    path = f"{run.record.folder}/inpainting/attempt-{number:03d}.json"
    # The reservation is kept before inference, so interrupted work stays attributable.
    run.ctx.keep(path, json.dumps(row, indent=2) + "\n")
    result = sample_candidate(
        backend.checkpoint,
        conditions,
        seed=seed,
        guidance_scale=backend.guidance_scale,
        deadline=deadline,
        observed=run.observation.tokens,
    )
    work.tokens += result["work"]["tokens"]
    work.attention_cells += result["work"]["attention_cells"]
    work.passes += result["work"]["passes"]
    generated: list[int] = result["generated_tokens"]
    row.update(
        status="rejected",
        work=result["work"],
        sampler_status=result["status"],
        diagnostics=list(result.get("diagnostics", [])),
        generated_sha256=identity(generated),
        pass_logit_sha256=result.get("pass_logit_sha256", {}),
        masked_values=_masked_values(run.observation, generated),
    )
    candidate: NarrativePlan | None = None
    try:
        if result["status"] == "resource-stop":
            row["status"] = work.status = "resource-stop"
            raise StepFailedError(f"revision-{run.record.name}", "E inference allowance exhausted")
        if result["status"] != "complete":
            raise CandidateError("learned layout failed registered decoding")
        candidate = NarrativePlan.model_validate(
            restore(generated, run.observation, run.record.before)
        )
        errors = candidate_errors(run.record.before, candidate, run.record.spec)
        if errors:
            raise CandidateError("; ".join(errors))
    except (CandidateError, ValidationError) as error:
        row["diagnostics"].append(str(error))
        candidate = None
    else:
        row["status"] = "selected"
        work.selected_attempt = number
    finally:
        run.ctx.keep(path, json.dumps(row, indent=2) + "\n")
    return candidate


def _masked_values(observation: Observation, generated: list[int]) -> dict[str, str]:
    """Readable generated bytes at each masked leaf, keyed by its view path."""
    values: dict[str, str] = {}
    for path in observation.editable:
        leaf = observation.leaves[path]
        data = bytes(max(0, token - BYTE_OFFSET) for token in generated[leaf.start : leaf.end])
        values["/".join(map(str, path))] = data.decode("utf-8", errors="replace")
    return values


def _structure(run: _Run, work: _Work) -> NarrativePlan:
    started = time.monotonic()
    deadline = started + run.backend.max_seconds
    try:
        for number in range(1, run.backend.max_attempts + 1):
            candidate = _attempt(run, work, number, deadline)
            if candidate is not None:
                work.status = "selected"
                return candidate
            if not run.checkpoint["binding"]["config"]["sampling_temperature"]:
                # Argmax decoding repeats a rejected candidate exactly; do not spend on copies.
                work.status = "deterministic-rejection"
                break
        else:
            work.status = "attempt-limit"
        raise StepFailedError(
            f"revision-{run.record.name}", "E inpainting produced no valid complete candidate"
        )
    finally:
        work.seconds = time.monotonic() - started


def _complete_text(run: _Run, structure: NarrativePlan) -> NarrativePlan:
    manifest = run.observation.text_fields
    if not manifest:
        return structure
    before, spec = run.record.before, run.record.spec

    def check(value: TextCompletion) -> list[str]:
        if set(value.fields) != set(manifest):
            return ["revision_text: response fields must exactly match the authorised selectors"]
        return candidate_errors(before, _fill(structure, value.fields), spec)

    completed = request_plan(
        run.ctx,
        Step(
            f"revision-{run.record.name}-text",
            "planner",
            run.backend.text_prompt,
            {
                "brief": run.ctx.brief.model_dump_json(),
                "plan": structure.model_dump_json(),
                "revision": spec.model_dump_json(),
                "fields": json.dumps(manifest),
            },
        ),
        TextCompletion,
        check,
    )
    return _fill(structure, completed.fields)


def _metadata(run: _Run | None, backend: ERevisionBackend, work: _Work) -> dict[str, Any]:
    if run is None:
        return {
            "version": "1",
            "kind": "e",
            "inpainting_id": INPAINTING_ID,
            "checkpoint": str(backend.checkpoint),
            "status": "preflight-failed",
            "attempts": [],
            "work": {"tokens": 0, "attention_cells": 0, "passes": 0, "elapsed_seconds": 0.0},
        }
    binding = run.checkpoint["binding"]
    return {
        "version": "1",
        "kind": "e",
        "inpainting_id": INPAINTING_ID,
        "specification": SPECIFICATION,
        "checkpoint": str(run.backend.checkpoint),
        "checkpoint_id": run.checkpoint["id"],
        "checkpoint_file_sha256": hashlib.sha256(run.backend.checkpoint.read_bytes()).hexdigest(),
        "parameter_sha256": run.checkpoint["parameter_sha256"],
        "representation": binding["representation"],
        "layout_id": binding["layout_id"],
        "condition_id": binding["condition_id"],
        "guidance_policy": binding["guidance_policy"],
        "test_only": binding["config"]["test_only"],
        "scientific_evidence": binding["scientific_evidence"].get("status"),
        "target_policy": run.backend.target_policy,
        "conditions": Conditions.model_validate(run.backend.conditions).model_dump(mode="json"),
        "guidance_scale": run.backend.guidance_scale,
        "sampling_temperature": binding["config"]["sampling_temperature"],
        "text_prompt": run.backend.text_prompt.label,
        "allowances": {
            "max_attempts": run.backend.max_attempts,
            "max_tokens": run.backend.max_tokens,
            "max_attention_cells": run.backend.max_attention_cells,
            "max_seconds": run.backend.max_seconds,
        },
        **run.observation.summary(),
        "status": work.status,
        "attempts": work.attempts,
        "selected_attempt": work.selected_attempt,
        "work": {
            "tokens": work.tokens,
            "attention_cells": work.attention_cells,
            "passes": work.passes,
            "elapsed_seconds": work.seconds,
        },
        "literary_adherence": "unassessed; software contract only",
    }


def inpaint_plan(
    ctx: StoryContext,
    before: NarrativePlan,
    spec: RevisionSpec,
    backend: ERevisionBackend,
    *,
    name: str = "revision",
) -> NarrativePlan:
    """E proposes masked structure, the planner fills authorised text; D's checks decide."""
    validate_selection(before, spec)
    check_name(name)
    record = RevisionRecord(
        name=name,
        before=before,
        spec=spec,
        first_call=len(ctx.state.calls),
        prompt_sha256=ctx.templates[backend.text_prompt.label].sha256,
        model=ctx.roles["planner"].model_dump(mode="json"),
    )
    ctx.keep(f"{record.folder}/source-plan.json", before.model_dump_json(indent=2) + "\n")
    run: _Run | None = None
    work = _Work()
    try:
        try:
            observation, checkpoint = prepare_inpainting(before, spec, backend)
        except ValueError as error:
            raise StepFailedError(f"revision-{name}", str(error)) from error
        run = _Run(ctx, record, backend, observation, checkpoint)
        ctx.keep(
            f"{record.folder}/inpainting/observation.json",
            json.dumps({"view": observation.view, **observation.summary()}, indent=2) + "\n",
        )
        selected = _complete_text(run, _structure(run, work))
        keep_selected(ctx, record, selected)
        return selected
    except Exception as failure:
        record.error = str(failure)
        raise
    finally:
        record.backend = _metadata(run, backend, work)
        record_revision(ctx, record)
