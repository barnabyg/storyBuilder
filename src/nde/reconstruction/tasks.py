"""Reproducible corruption tasks with an explicit model-visible boundary."""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from pydantic import Field

from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.corpus.reliability import CHANNELS
from nde.llm.config import Strict
from nde.narrative.validation import validate_plan
from nde.reconstruction import codec

MIN_RATE, MAX_RATE = 0.1, 0.7
SOFT_WEIGHT = 0.25
type Record = dict[str, Any]


class TaskConfig(Strict):
    seed: int = 7
    repetitions: int = Field(default=1, ge=1, le=100)
    schedules: list[dict[str, Any]] = Field(min_length=1)


def _schedule(schedule: Record) -> None:
    allowed = {"kind", "rate", "channel", "start", "count"}
    if set(schedule) - allowed or schedule.get("kind") not in {
        "fields",
        "scenes",
        "channels",
        "span",
    }:
        raise ValueError("unsupported corruption schedule")
    rate = schedule.get("rate")
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not MIN_RATE <= rate <= MAX_RATE
    ):
        raise ValueError("mask rate must be between 0.1 and 0.7")
    if "channel" in schedule and schedule["channel"] not in CHANNELS:
        raise ValueError("channel must be a registered channel name")
    for name in ("start", "count"):
        if name in schedule and (
            schedule["kind"] != "span"
            or not isinstance(schedule[name], int)
            or isinstance(schedule[name], bool)
            or schedule[name] < (1 if name == "count" else 0)
        ):
            raise ValueError("span boundaries must be nonnegative integers with positive count")


def _groups(record: Record, schedule: Record) -> list[list[int]]:
    eligible = [
        index
        for index, slot in enumerate(record["slots"])
        if slot["eligible"]
        and ("channel" not in schedule or slot["channel"] == schedule["channel"])
    ]
    if schedule["kind"] in {"fields", "span"}:
        return [[index] for index in eligible]
    if schedule["kind"] == "channels":
        channels = sorted({record["slots"][index]["channel"] for index in eligible})
        return [
            [index for index in eligible if record["slots"][index]["channel"] == channel]
            for channel in channels
        ]
    # Every eligible field of a record touching a scene shares its scene mask.
    scene_groups: dict[str, list[int]] = {}
    setups = {item["id"]: item["scene_id"] for item in record["view"].get("setups", [])}
    for index in eligible:
        path = record["slots"][index]["path"]
        scenes: set[str] = set()
        if path[0] == "scenes" and isinstance(path[1], int):
            scenes.add(record["view"]["scenes"][path[1]]["id"])
        elif len(path) > 1 and isinstance(path[1], int):
            item = record["view"][path[0]][path[1]]
            for key, value in item.items():
                if key.endswith("scene_id") and isinstance(value, str):
                    scenes.add(value)
                elif key.endswith("scene_ids"):
                    scenes.update(value)
            if item.get("setup_id") in setups:
                scenes.add(setups[item["setup_id"]])
        for scene in scenes:
            scene_groups.setdefault(scene, []).append(index)
    return [scene_groups[scene] for scene in sorted(scene_groups)]


def make_task(record: Record, schedule: Record, *, seed: int = 7, repetition: int = 0) -> Record:
    codec.decode_record(record)
    _schedule(schedule)
    if repetition < 0:
        raise ValueError("negative corruption repetition")
    groups = _groups(record, schedule)
    stable_seed = identity(
        {"record": record["id"], "schedule": schedule, "seed": seed, "repetition": repetition}
    )
    # S311: repeatable scientific corruption, never security-sensitive randomness.
    rng = random.Random(stable_seed)  # noqa: S311
    count = math.ceil(len(groups) * schedule["rate"])
    if schedule["kind"] == "span":
        count = schedule.get("count", count)
        start = schedule.get(
            "start",
            rng.randrange(len(groups) - count + 1) if groups and count <= len(groups) else 0,
        )
        if start + count > len(groups):
            raise ValueError("contiguous span exceeds eligible field boundary")
        chosen = groups[start : start + count]
    else:
        chosen = rng.sample(groups, count)
    positions = sorted({position for group in chosen for position in group})
    visible = {"slots": copy.deepcopy(record["slots"])}
    targets: list[Record] = []
    for index, slot in enumerate(visible["slots"]):
        slot["masked"] = index in positions
        if slot["masked"]:
            targets.append(
                {
                    "position": index,
                    "value": copy.deepcopy(slot["value"]),
                    "policy": slot["policy"],
                    "weight": 1.0 if slot["policy"] == "hard" else SOFT_WEIGHT,
                }
            )
            slot["value"] = None
    eligible_fields = len({position for group in groups for position in group})
    return codec.seal(
        {
            "codec_id": codec.CODEC_ID,
            "vocabulary_id": codec.VOCABULARY_ID,
            "record_id": record["id"],
            "visible": visible,
            "visible_id": identity(visible),
            "allowed_positions": positions,
            "targets": targets,
            "provenance": copy.deepcopy(record["provenance"]),
            "metadata": {
                "schedule": copy.deepcopy(schedule),
                "requested_rate": schedule["rate"],
                "actual_rate": len(positions) / eligible_fields if eligible_fields else 0.0,
                "unit": schedule["kind"],
                "rounding": "ceiling of eligible units",
                "eligible_units": len(groups),
                "masked_units": len(chosen),
                "eligible_fields": eligible_fields,
                "masked_fields": len(positions),
                "seed": seed,
                "stable_seed": stable_seed,
                "repetition": repetition,
            },
        }
    )


def validate_task(task: Record) -> None:
    if task.get("id") != identity(task) or task.get("codec_id") != codec.CODEC_ID:
        raise ValueError("stale corruption task or codec")
    if task.get("vocabulary_id") != codec.VOCABULARY_ID or task.get("visible_id") != identity(
        task["visible"]
    ):
        raise ValueError("stale visible information or vocabulary")
    positions = task["allowed_positions"]
    if positions != sorted(set(positions)) or any(
        not isinstance(p, int) or p < 0 or p >= len(task["visible"]["slots"]) for p in positions
    ):
        raise ValueError("invalid prediction positions")
    if [item["position"] for item in task["targets"]] != positions:
        raise ValueError("target/prediction positions differ")
    for index, slot in enumerate(task["visible"]["slots"]):
        if slot.get("masked") != (index in positions):
            raise ValueError("mask/prediction positions differ")
        if slot["masked"] and (not slot["eligible"] or slot["value"] is not None):
            raise ValueError("hidden slot leaks a target or is ineligible")
    for target in task["targets"]:
        slot = task["visible"]["slots"][target["position"]]
        expected_weight = 1.0 if slot["policy"] == "hard" else SOFT_WEIGHT
        if target["policy"] != slot["policy"] or target["weight"] != expected_weight:
            raise ValueError("target policy/loss weight changed")


def model_input(task: Record) -> list[int]:
    """Both objectives receive exactly this whitelist, never evaluator-only fields."""
    validate_task(task)
    return codec.tokenize(
        [
            ["/".join(map(str, slot["path"])), {"mask": True} if slot["masked"] else slot["value"]]
            for slot in task["visible"]["slots"]
        ]
    )


def reconstruct(task: Record, predictions: dict[int, Any]) -> Record:
    validate_task(task)
    if set(predictions) != set(task["allowed_positions"]):
        raise ValueError(
            "predictions must fill exactly the allowed positions; observations immutable"
        )
    slots = copy.deepcopy(task["visible"]["slots"])
    for position, value in predictions.items():
        slots[position]["value"] = copy.deepcopy(value)
    return codec.view_from_slots(slots)


def candidate_check(task: Record, predictions: dict[int, Any]) -> Record:
    try:
        view = reconstruct(task, predictions)
        validation = validate_plan(codec.materialize_view(view))
        return {
            "view": view,
            "valid": validation.valid,
            "diagnostics": validation.model_dump(mode="json")["diagnostics"],
            "validation_scope": "retained structure; excluded channels unavailable",
        }
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return {
            "view": None,
            "valid": False,
            "diagnostics": [str(error)],
            "validation_scope": "retained structure; excluded channels unavailable",
        }


def create_tasks(encoding: Path, config: Path, out: Path) -> Record:
    source = codec.load_encoding(encoding)
    path = encoding / "encoding.json" if encoding.is_dir() else encoding
    settings = TaskConfig.model_validate_json(config.read_text(encoding="utf-8"))
    generated = [
        make_task(record, schedule, seed=settings.seed, repetition=repetition)
        for record in source["records"]
        for schedule in settings.schedules
        for repetition in range(settings.repetitions)
    ]
    if len({task["id"] for task in generated}) != len(generated):
        raise ValueError("duplicate registered corruption schedules")
    result = create_record(
        out / "tasks.json",
        {
            "codec_id": codec.CODEC_ID,
            "vocabulary_id": codec.VOCABULARY_ID,
            "encoding_path": str(path.resolve()),
            "encoding_id": source["id"],
            "test_only": source["test_only"],
            "config": settings.model_dump(mode="json"),
            "tasks": generated,
            "inputs": snapshot([path, config]),
        },
    )
    report_files(out, "Saved structural corruption tasks", result)
    return result


def load_tasks(path: Path) -> Record:
    path = path / "tasks.json" if path.is_dir() else path
    result = read_sealed(path)
    check_snapshot(result["inputs"])
    source = codec.load_encoding(Path(result["encoding_path"]))
    if source["id"] != result["encoding_id"]:
        raise ValueError("stale task encoding")
    settings = TaskConfig.model_validate(result["config"])
    expected = [
        make_task(record, schedule, seed=settings.seed, repetition=repetition)
        for record in source["records"]
        for schedule in settings.schedules
        for repetition in range(settings.repetitions)
    ]
    if expected != result["tasks"]:
        raise ValueError("corruption tasks differ from registered inputs")
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="tasks_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("path", type=Path)
    create.add_argument("--schedule", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        result = (
            create_tasks(args.path, args.schedule, args.out)
            if args.tasks_command == "create"
            else load_tasks(args.path)
        )
        print(json.dumps({"id": result["id"], "tasks": len(result["tasks"])}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde reconstruction tasks: {error}", file=sys.stderr)
        return 2
