"""Conditional layout targets learn collection lengths, identities and references."""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import Field

from nde.conditional.data import load_dataset
from nde.corpus.channels import CHANNEL_FIELDS
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_json,
    read_sealed,
    report_files,
    snapshot,
)
from nde.llm.config import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.reconstruction.codec import canonical_json, detokenize, seal, tokenize
from nde.reconstruction.tasks import SOFT_WEIGHT

SPECIFICATION = {
    "name": "nde-conditional-layout",
    "version": "1",
    "mode": "full-generation",
    "serialization": "complete canonical JSON UTF-8, BOS/EOS; fixed registered output capacity",
    "generated": "all collection counts, order, anonymous IDs, reference topology and typed values",
    "text": "free text replaced with explicit completion placeholders; no source/evidence",
    "identity": "collection/index canonical IDs; list positions define chronology",
    "termination": "learned EOS within registered capacity; missing EOS is invalid",
}
LAYOUT_ID = identity(SPECIFICATION)
PLACEHOLDER = "Unfilled generated text"
TEXT_FIELDS = frozenset({"name", "title", "summary", "description", "note"})
INTERPRETIVE_FIELDS = frozenset(
    {"arcs", "reader_affect", "tension", "themes", "motifs", "discourse", "subtext"}
)
PLAN_FIELDS = frozenset(NarrativePlan.model_fields) - {
    "schema_version",
    "id",
    "tier",
    "title",
    "sources",
    "constraints",
}


class Capacity(Strict):
    max_scenes: int = Field(default=16, ge=1, le=256)
    max_characters: int = Field(default=16, ge=0, le=256)
    max_records: int = Field(default=256, ge=1, le=4096)
    max_tokens: int = Field(default=4096, ge=16, le=65536)


def textless(value: Any, key: str = "", collection: str = "") -> Any:
    if isinstance(value, dict):
        row = cast(dict[str, Any], value)
        return {
            field: textless(item, field, collection or field)
            for field, item in row.items()
            if field not in {"evidence", "sources"}
        }
    if isinstance(value, list):
        return [textless(item, key, collection) for item in cast(list[Any], value)]
    if isinstance(value, str) and (
        key in TEXT_FIELDS
        or (collection in INTERPRETIVE_FIELDS and key in {"value", "alternatives"})
    ):
        return PLACEHOLDER
    return value


def materialize(view: dict[str, Any]) -> NarrativePlan:
    """Fill only schema-required text, preserving every generated structural value."""
    if set(view) - PLAN_FIELDS:
        raise ValueError("layout contains unsupported/evaluator-only plan fields")
    data = copy.deepcopy(view)
    for character in data.get("characters", []):
        character.setdefault("name", PLACEHOLDER)
    for scene in data.get("scenes", []):
        scene.setdefault("title", PLACEHOLDER)
        scene.setdefault("summary", PLACEHOLDER)
    return NarrativePlan.model_validate(
        {"schema_version": "1", "id": "conditional-proposal", "title": PLACEHOLDER, **data}
    )


def validate_layout(view: dict[str, Any], capacity: Capacity) -> NarrativePlan:
    if len(view.get("scenes", [])) > capacity.max_scenes:
        raise ValueError("layout exceeds scene capacity; truncation forbidden")
    if len(view.get("characters", [])) > capacity.max_characters:
        raise ValueError("layout exceeds character capacity; truncation forbidden")
    count = 0

    def inspect(value: Any) -> None:
        nonlocal count
        if isinstance(value, dict):
            row = cast(dict[str, Any], value)
            if "id" in row:
                count += 1
            if {"sources", "evidence", "provenance", "source_text"} & set(row):
                raise ValueError("source/evaluator evidence is forbidden in generated layouts")
            for item in row.values():
                inspect(item)
        elif isinstance(value, list):
            for item in cast(list[Any], value):
                inspect(item)

    inspect(view)
    if count > capacity.max_records or len(tokenize(view)) > capacity.max_tokens:
        raise ValueError("layout exceeds record/token capacity; truncation forbidden")
    plan = materialize(view)
    diagnostics = validate_plan(plan)
    if not diagnostics.valid:
        raise ValueError(diagnostics.model_dump_json())
    return plan


def _loss_weights(view: dict[str, Any], policy: dict[str, str]) -> list[float]:
    weights = [1.0]

    def emit(text: str, weight: float) -> None:
        weights.extend([weight] * len(text.encode("utf-8")))

    def channel_weight(name: str) -> float:
        return {"hard": 1.0, "soft": SOFT_WEIGHT, "excluded": 0.0}[policy.get(name, "excluded")]

    def visit(value: Any, weight: float, *, collection: str = "", root: bool = False) -> None:
        if isinstance(value, dict):
            row = cast(dict[str, Any], value)
            if collection == "states" and "kind" in row:
                weight = channel_weight(row["kind"])
            emit("{", weight)
            for index, (key, item) in enumerate(sorted(row.items())):
                emit(("," if index else "") + canonical_json(key) + ":", weight)
                selected = weight
                if root:
                    channel = next(
                        (name for name, fields in CHANNEL_FIELDS.items() if key in fields), "plot"
                    )
                    selected = channel_weight(channel)
                if (
                    row.get("status") in {"unknown", "ambiguous"}
                    and key in {"value", "alternatives", "confidence"}
                    and item is not None
                    and item != []
                ):
                    selected = 0.0
                visit(item, selected, collection=key if root else collection)
            emit("}", weight)
        elif isinstance(value, list):
            emit("[", weight)
            for index, item in enumerate(cast(list[Any], value)):
                if index:
                    emit(",", weight)
                visit(item, weight, collection=collection)
            emit("]", weight)
        else:
            emit(canonical_json(value), weight)

    visit(view, 1.0, root=True)
    return [*weights, 1.0]


def encode_target(target: dict[str, Any], capacity: Capacity) -> dict[str, Any]:
    """Only approved, filtered target views enter this training-only conversion."""
    if target["target_policy"].get("plot", "excluded") == "excluded":
        raise ValueError(
            "layout learning requires approved plot/count targets, not context scaffolding"
        )
    view = textless(target["view"])
    validate_layout(view, capacity)
    return seal(
        {
            "kind": "conditional-layout-target",
            "layout_id": LAYOUT_ID,
            "capacity": capacity.model_dump(mode="json"),
            "parent_target_id": target["id"],
            "target_policy": target["target_policy"],
            "view": view,
            "tokens": tokenize(view),
            "loss_weights": _loss_weights(view, target["target_policy"]),
            "loss_policy": (
                "hard=1; soft=0.25; uncertain asserted values=0; explicit null/syntax retained"
            ),
            "counts": {
                key: len(cast(list[Any], value))
                for key, value in view.items()
                if isinstance(value, list)
            },
        }
    )


def decode_layout(tokens: list[int], capacity: Capacity) -> dict[str, Any]:
    if len(tokens) > capacity.max_tokens:
        raise ValueError("layout exceeds registered output capacity")
    view = detokenize(tokens)
    if not isinstance(view, dict):
        raise ValueError("layout must decode to an object")
    result = cast(dict[str, Any], view)
    validate_layout(result, capacity)
    return result


def encode_dataset(
    dataset_path: Path, out: Path, capacity: Capacity, *, capacity_path: Path | None = None
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    result = create_record(
        out / "layouts.json",
        {
            "kind": "conditional-layout-dataset",
            "layout_id": LAYOUT_ID,
            "dataset_id": dataset["id"],
            "dataset_path": str(dataset_path.resolve()),
            "inputs": snapshot(
                [
                    dataset_path / "dataset.json" if dataset_path.is_dir() else dataset_path,
                    *([capacity_path] if capacity_path is not None else []),
                ]
            ),
            "capacity": capacity.model_dump(mode="json"),
            "examples": [
                {"example_id": row["id"], "layout": encode_target(row["target"], capacity)}
                for row in dataset["examples"]
            ],
        },
    )
    report_files(out, "Conditional layout targets (training only)", result)
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="layout_command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("dataset", type=Path)
    encode.add_argument("capacity", type=Path)
    encode.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.layout_command == "encode":
            result = encode_dataset(
                args.dataset,
                args.out,
                Capacity.model_validate(read_json(args.capacity)),
                capacity_path=args.capacity,
            )
        else:
            result = read_sealed(args.path)
            if (
                result.get("layout_id") != LAYOUT_ID
                or result.get("kind") != "conditional-layout-dataset"
            ):
                raise ValueError("unsupported conditional layout identity")
            check_snapshot(result["inputs"])
            dataset = load_dataset(Path(result["dataset_path"]))
            if dataset["id"] != result["dataset_id"]:
                raise ValueError("conditional layout dataset changed")
            capacity = Capacity.model_validate(result["capacity"])
            expected = [
                {"example_id": row["id"], "layout": encode_target(row["target"], capacity)}
                for row in dataset["examples"]
            ]
            if result["examples"] != expected:
                raise ValueError("layout targets/counts/policies differ from authorized parent")
            for row in result["examples"]:
                decoded = decode_layout(row["layout"]["tokens"], capacity)
                if decoded != row["layout"]["view"]:
                    raise ValueError("layout token/view identity differs")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"nde conditional-layout: {error}", file=sys.stderr)
        return 2
