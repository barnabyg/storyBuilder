"""Target-free structural proposals with bounded rejection and code-enforced locks."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from nde.conditional.data import Conditions
from nde.conditional.layout import Capacity, validate_layout
from nde.conditional.training import candidate_cost, inspect_checkpoint, sample_candidate
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    save_json,
    snapshot,
)
from nde.llm.config import Strict
from nde.narrative.interpretation import iter_annotations
from nde.narrative.models import NarrativePlan
from nde.narrative.state import Constraint
from nde.narrative.state_validation import constraint_status
from nde.narrative.validation import validate_plan
from nde.plan.revision import selector_value
from nde.reconstruction import g3
from nde.reconstruction.codec import seal
from nde.reconstruction.resources import TimeAllowance

PLACEHOLDER = "Unfilled generated text"
PROPOSAL_KIND = "conditional-structural-proposal"


class Generation(Strict):
    version: Literal["1"] = "1"
    mode: Literal["full-generation"] = "full-generation"
    checkpoint: Path
    conditions: Conditions = Field(default_factory=Conditions)
    locks: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    constraints: list[Constraint] = Field(default_factory=list[Constraint])
    seed: int = 7
    guidance_scale: float = Field(default=1, ge=0, le=8, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=100)
    max_tokens: int = Field(default=1000000, ge=1)
    max_attention_cells: int = Field(default=100000000, ge=1)
    max_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _authority(self) -> "Generation":
        for selector, value in self.locks.items():
            parts = selector.split("/")
            if (
                "/" not in selector
                or any(part in {"", "sources", "evidence", "constraints", "id"} for part in parts)
                or isinstance(value, (dict, list))
            ):
                raise ValueError(
                    "locks must select existing scalar leaves outside evidence/envelope"
                )
        keys: dict[tuple[str | None, str, str], str | None] = {}
        for constraint in self.constraints:
            if constraint.kind == "literary":
                continue
            if not constraint.expected:
                raise ValueError("hard constraints require an explicit expected value")
            key = constraint.character_id, constraint.state_kind, constraint.key
            if key in keys and keys[key] != constraint.expected:
                raise ValueError("conflicting hard constraints")
            keys[key] = constraint.expected
        if len({item.id for item in self.constraints}) != len(self.constraints):
            raise ValueError("duplicate constraint identities")
        return self


def _set_leaf(content: dict[str, Any], selector: str, value: JsonValue) -> None:
    current: Any = content
    parts = selector.split("/")
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = cast(dict[str, Any], current)[part]
        elif isinstance(current, list):
            matches = [row for row in cast(list[Any], current) if row.get("id") == part]
            if len(matches) != 1:
                raise ValueError(f"lock references missing/ambiguous identity: {selector}")
            current = matches[0]
        else:
            raise ValueError(f"lock is not an existing leaf: {selector}")
    if not isinstance(current, dict):
        raise ValueError(f"lock is not an existing scalar leaf: {selector}")
    container = cast(dict[str, Any], current)
    leaf = parts[-1]
    if leaf not in container or isinstance(container[leaf], (dict, list)):
        raise ValueError(f"lock is not an existing scalar leaf: {selector}")
    container[leaf] = value


def _source_free(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in cast(dict[str, Any], value).items():
            if key in {"sources", "evidence"} and child:
                raise ValueError("generated fiction cannot assert source evidence")
            _source_free(child)
    elif isinstance(value, list):
        for child in cast(list[Any], value):
            _source_free(child)


def completion_fields(plan: NarrativePlan, locks: dict[str, Any]) -> list[str]:
    """Only descriptive leaves may be filled; structural symbols and readings stay learned."""
    fields = ["title"] if plan.title == PLACEHOLDER else []
    for character in plan.characters:
        if character.name in {"", PLACEHOLDER, character.id}:
            fields.append(f"characters/{character.id}/name")
    for scene in plan.scenes:
        for name in ("title", "summary"):
            if getattr(scene, name) in {"", PLACEHOLDER, "retained scene"}:
                fields.append(f"scenes/{scene.id}/{name}")
        fields.extend(
            f"scenes/{scene.id}/beats/{beat.id}/summary"
            for beat in scene.beats
            if beat.summary in {"", PLACEHOLDER}
        )
    for root in ("setups", "knowledge_facts"):
        for row in getattr(plan, root):
            if row.description in {"", PLACEHOLDER}:
                fields.append(f"{root}/{row.id}/description")
    for channel, row in iter_annotations(plan):
        if row.reading.value in {"", PLACEHOLDER}:
            fields.append(f"{channel}/{row.id}/reading/value")
        fields.extend(
            f"{channel}/{row.id}/reading/alternatives/{index}"
            for index, value in enumerate(row.reading.alternatives)
            if value in {"", PLACEHOLDER}
        )
    return sorted(
        field
        for field in fields
        if not any(field == lock or field.startswith(lock + "/") for lock in locks)
    )


def _candidate_plan(view: dict[str, Any], config: Generation, capacity: Capacity) -> NarrativePlan:
    _source_free(view)
    if view.get("constraints"):
        raise ValueError("model cannot supply caller constraint authority")
    plan = validate_layout(view, capacity)
    content = plan.model_dump(mode="json")
    content.update(id="generated-plan", title=PLACEHOLDER)
    content["constraints"] = [item.model_dump(mode="json") for item in config.constraints]
    for selector, value in config.locks.items():
        _set_leaf(content, selector, value)
    plan = NarrativePlan.model_validate(content)
    scene_range = config.conditions.scene_range
    if scene_range is not None and not scene_range[0] <= len(plan.scenes) <= scene_range[1]:
        raise ValueError("generated scene count outside caller range")
    if len(plan.scenes) > capacity.max_scenes or len(plan.characters) > capacity.max_characters:
        raise ValueError("generated layout exceeds registered capacity")
    result = validate_plan(plan)
    if not result.valid:
        raise ValueError(result.model_dump_json())
    _check_hard_constraints(plan)
    if any(selector_value(plan, selector) != value for selector, value in config.locks.items()):
        raise ValueError("generated proposal did not preserve every explicit lock")
    return plan


def _check_hard_constraints(plan: NarrativePlan) -> None:
    if any(
        constraint.kind != "literary" and constraint_status(plan, constraint) != "satisfied"
        for constraint in plan.constraints
    ):
        raise ValueError(
            "hard constraint is unresolved or violated; unknown facts are not authority"
        )


def read_proposal(path: Path) -> dict[str, Any]:
    file = path / "proposal.json" if path.is_dir() else path
    proposal = read_sealed(file)
    if proposal.get("kind") != PROPOSAL_KIND or proposal.get("status") != "structural-proposal":
        raise ValueError("not an accepted E structural proposal")
    check_snapshot(proposal["inputs"])
    plan = NarrativePlan.model_validate(proposal["plan"])
    _source_free(proposal["plan"])
    if not validate_plan(plan).valid:
        raise ValueError("saved structural proposal is invalid")
    _check_hard_constraints(plan)
    authority = Generation(
        checkpoint=file,
        conditions=Conditions.model_validate(proposal["conditions"]),
        locks=TypeAdapter(dict[str, JsonValue]).validate_python(proposal["locks"], strict=True),
        constraints=plan.constraints,
    )
    for selector, value in authority.locks.items():
        if selector_value(plan, selector) != value:
            raise ValueError("saved structural proposal lock differs")
    manifest = TypeAdapter(list[str]).validate_python(proposal["completion_fields"], strict=True)
    if completion_fields(plan, authority.locks) != sorted(manifest):
        raise ValueError("saved completion authority differs from proposal fields")
    return proposal


def _preflight(
    config_path: Path, out: Path
) -> tuple[Generation, dict[str, Any], Capacity, dict[str, str]]:
    config = Generation.model_validate_json(config_path.read_text(encoding="utf-8"))
    checkpoint = inspect_checkpoint(config.checkpoint)
    scientific_authority(checkpoint)
    capacity = Capacity.model_validate(checkpoint["binding"]["config"]["capacity"])
    if (
        config.conditions.scene_range is not None
        and config.conditions.scene_range[1] > capacity.max_scenes
    ):
        raise ValueError("caller scene range exceeds registered checkpoint capacity")
    if out.exists() and any(out.iterdir()):
        raise ValueError("generation output already exists; completed attempts are immutable")
    out.mkdir(parents=True, exist_ok=True)
    inputs = snapshot([config_path, config.checkpoint])
    return config, checkpoint, capacity, inputs


def scientific_authority(checkpoint: dict[str, Any]) -> None:
    training = checkpoint["binding"]["config"]
    if training["test_only"]:
        return
    protocol = training.get("g3_protocol")
    if protocol is None:
        raise ValueError("real E generation requires genuine passed G3 evidence")
    decision = g3.assess(Path(protocol))
    if decision["test_only"] or decision["outcome"] != "passed":
        raise ValueError("real E generation requires genuine passed G3 evidence")
    if checkpoint["binding"]["scientific_evidence"].get("decision_id") != decision["id"]:
        raise ValueError("checkpoint scientific evidence changed")


def generate(config_path: Path, out: Path) -> dict[str, Any]:
    started = time.monotonic()
    config, checkpoint, capacity, inputs = _preflight(config_path, out)
    binding_id = identity({"inputs": inputs, "config": config.model_dump(mode="json")})
    timer = TimeAllowance(out / "allowance.json", binding_id, config.max_seconds, started)
    tokens = cells = 0
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    status = "interrupted"
    try:
        for index in range(config.max_attempts):
            reserved_tokens, reserved_cells = candidate_cost(
                checkpoint, config.conditions, guidance_scale=config.guidance_scale
            )
            if (
                timer.expired()
                or tokens + reserved_tokens > config.max_tokens
                or cells + reserved_cells > config.max_attention_cells
            ):
                status = "resource-stop"
                break
            # Reserve before entering the model. An interrupted attempt remains attributable.
            save_json(
                out / f"attempt-{index + 1:03d}-pending.json",
                seal(
                    {
                        "binding_id": binding_id,
                        "tokens": reserved_tokens,
                        "attention_cells": reserved_cells,
                    }
                ),
            )
            before_tokens, before_cells = tokens, cells
            tokens += reserved_tokens
            cells += reserved_cells
            row: dict[str, Any] = {
                "attempt": index + 1,
                "seed": config.seed + index,
                "status": "interrupted",
                "diagnostics": [],
                "reserved_work": {"tokens": reserved_tokens, "attention_cells": reserved_cells},
                "recovery": "bounded rejection; next attempt masks all editable structural units",
            }
            attempts.append(row)
            timer.save(tokens=tokens, cells=cells)
            candidate = sample_candidate(
                config.checkpoint,
                config.conditions,
                seed=config.seed + index,
                guidance_scale=config.guidance_scale,
                deadline=timer.deadline,
            )
            work = candidate["work"]
            tokens = before_tokens + work["tokens"]
            cells = before_cells + work["attention_cells"]
            row.update(
                candidate=candidate,
                status="rejected",
                diagnostics=list(candidate.get("diagnostics", [])),
            )
            try:
                if (
                    candidate["status"] == "resource-stop"
                    or timer.expired()
                    or tokens > config.max_tokens
                    or cells > config.max_attention_cells
                ):
                    status = row["status"] = "resource-stop"
                    raise ValueError("resource allowance exhausted during model inference")
                if not isinstance(candidate.get("view"), dict):
                    raise ValueError("learned candidate is incomplete or malformed")
                plan = _candidate_plan(candidate["view"], config, capacity)
            except (ValueError, KeyError, TypeError) as error:
                row["diagnostics"].append(str(error))
            else:
                row["status"] = "selected"
                selected = create_record(
                    out / "proposal.json",
                    {
                        "kind": PROPOSAL_KIND,
                        "status": "structural-proposal",
                        "mode": config.mode,
                        "checkpoint_id": checkpoint["id"],
                        "test_only": checkpoint["binding"]["config"]["test_only"],
                        "conditions": config.conditions.model_dump(mode="json"),
                        "locks": config.locks,
                        "plan": plan.model_dump(mode="json"),
                        "completion_fields": completion_fields(plan, config.locks),
                        "inputs": inputs,
                        "selected_attempt": index + 1,
                        "literary_adherence": (
                            "unassessed; structural proposal requires authorized text completion"
                        ),
                    },
                )
                status = "structural-proposal"
            create_record(out / f"attempt-{index + 1:03d}.json", row)
            timer.save(tokens=tokens, cells=cells)
            if selected is not None or status == "resource-stop":
                break
        else:
            status = "attempt-limit"
    finally:
        result = {
            "kind": "conditional-generation-report",
            "status": status,
            "proposal_id": None if selected is None else selected["id"],
            "attempts": attempts,
            "work": {
                "tokens": tokens,
                "attention_cells": cells,
                "elapsed_seconds": timer.elapsed(),
            },
            "binding_id": binding_id,
            "inputs": inputs,
        }
        report_files(out, "Conditional full generation", result)
        timer.save(tokens=tokens, cells=cells, status="complete")
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="conditional_generation_command", required=True)
    command = commands.add_parser("generate")
    command.add_argument("config", type=Path)
    command.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("proposal", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        result = (
            generate(args.config, args.out)
            if args.conditional_generation_command == "generate"
            else read_proposal(args.proposal)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "structural-proposal" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde conditional-generate: {error}", file=sys.stderr)
        return 2
