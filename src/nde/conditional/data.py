"""Traceable condition-to-plan examples from approved, author-disjoint exports."""

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, JsonValue, field_validator, model_validator

from nde.corpus.export import validate_export
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.corpus.splits import load_cohort
from nde.llm.config import Strict
from nde.narrative.models import NarrativePlan
from nde.reconstruction.codec import CODEC_ID, SCHEMA_ID, encode_plan, seal

type Record = dict[str, Any]
type PositiveInteger = Annotated[int, Field(strict=True, gt=0)]
LABELS = ("genre", "themes", "ending_intent", "target_words", "scene_range")
MAX_LABEL_LENGTH = 100
CONDITION_SPECIFICATION = {
    "name": "nde-brief-conditions",
    "version": "1",
    "genre": "literary label, not hard canon",
    "themes": "short literary labels, not annotated theme records",
    "ending_intent": "natural-language intent, never invented symbolic canon",
    "target_words": "inclusive requested word-count range, never measured source length",
    "scene_range": "inclusive permitted caller range, never hidden target count",
    "missing": "unknown; no inferred default literary truth",
    "authority": "conditioning only; no structural locks or evaluator metadata",
}


class Conditions(Strict):
    """The entire inference-visible boundary; provenance and targets stay outside."""

    genre: str | None = Field(default=None, min_length=1, max_length=MAX_LABEL_LENGTH)
    themes: list[str] = Field(default_factory=list[str], max_length=12)
    ending_intent: str | None = Field(default=None, min_length=1, max_length=500)
    target_words: tuple[PositiveInteger, PositiveInteger] | None = None
    scene_range: tuple[PositiveInteger, PositiveInteger] | None = None

    @field_validator("genre", "ending_intent")
    @classmethod
    def literary_text(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip() or any(character in value for character in "{}[]")
        ):
            raise ValueError("condition must be literary text, not serialized target content")
        return value

    @field_validator("themes")
    @classmethod
    def theme_labels(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(
            not value.strip()
            or len(value) > MAX_LABEL_LENGTH
            or any(character in value for character in "{}[]")
            for value in values
        ):
            raise ValueError("themes must be unique short literary labels")
        return values

    @field_validator("target_words", "scene_range")
    @classmethod
    def inclusive_range(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and (value[0] < 1 or value[1] < value[0]):
            raise ValueError("condition ranges require positive ordered inclusive endpoints")
        return value


CONDITION_ID = identity(
    {"specification": CONDITION_SPECIFICATION, "schema": Conditions.model_json_schema()}
)


class Label(Strict):
    value: JsonValue = None
    origin: Literal["missing", "metadata", "derived"] = "missing"
    status: Literal["known", "uncertain", "unknown"] = "unknown"
    uncertainty: float = Field(default=1, ge=0, le=1)
    method: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    input_files: list[Path] = Field(default_factory=list[Path])

    @model_validator(mode="after")
    def traceable(self) -> Self:
        if (self.status == "unknown") != (self.value is None):
            raise ValueError("unknown labels require null; known/uncertain labels require a value")
        if self.origin == "missing" and self.status != "unknown":
            raise ValueError("missing labels must stay unknown")
        if self.status == "uncertain" and not self.uncertainty:
            raise ValueError("uncertain labels require positive uncertainty")
        if self.origin == "derived" and self.method is None:
            raise ValueError("derived labels require a derivation method")
        if (self.model is None) != (self.prompt is None):
            raise ValueError("model derivations require both model and prompt identities")
        if self.origin != "derived" and any((self.method, self.model, self.prompt)):
            raise ValueError("derivation identities are only valid for derived labels")
        return self


class ConditionRecord(Strict):
    work_id: str = Field(min_length=1)
    family: str | None = Field(default=None, min_length=1)
    labels: dict[str, Label] = Field(default_factory=dict[str, Label])

    @field_validator("labels")
    @classmethod
    def registered_labels(cls, values: dict[str, Label]) -> dict[str, Label]:
        if set(values) - set(LABELS):
            raise ValueError("unregistered condition fields may leak target content")
        if any(
            values[name].origin == "derived"
            for name in ("target_words", "scene_range")
            if name in values
        ):
            raise ValueError("ranges must be supplied requests, never derived target scaffolding")
        return values


class Preparation(Strict):
    export: Path | None = None
    evaluation: Path | None = None
    conditions: list[ConditionRecord] = Field(default_factory=list[ConditionRecord])
    max_scenes: int = Field(default=256, ge=1)
    max_tokens: int = Field(default=65536, ge=1)

    @model_validator(mode="after")
    def one_source(self) -> Self:
        if (self.export is None) == (self.evaluation is None):
            raise ValueError("choose approved training export or authorized evaluation")
        return self


def _file(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def _source(policy: Preparation) -> tuple[Record, Record, Path, Path, str, str]:
    if policy.export is not None:
        path = _file(policy.export, "dataset.json")
        exported = validate_export(path)
        return exported, exported, path, path, "train", "optimizer"
    if policy.evaluation is None:
        raise ValueError("missing evaluation authorization")
    path = _file(policy.evaluation, "authorization.json")
    source = read_sealed(path)
    check_snapshot(source["inputs"])
    exported_path = Path(source["export_path"])
    exported = validate_export(exported_path)
    if (
        source.get("kind") != "reconstruction-evaluation-authorization"
        or source.get("role") != "evaluator-only"
        or source.get("split") not in {"validation", "test"}
        or source["export_id"] != exported["id"]
        or source["target_policy"] != exported["target_policy"]
        or source["cohort_path"] != exported["cohort_path"]
        or source["test_only"] != exported["test_only"]
    ):
        raise ValueError("stale or unsupported evaluator-only authorization")
    return source, exported, path, exported_path, source["split"], "evaluator-only"


def _labels(policy: Preparation, rows: list[Record]) -> dict[tuple[str, str], dict[str, Label]]:
    selections: dict[tuple[str, str], dict[str, Label]] = {}
    for supplied in policy.conditions:
        matches = [
            row
            for row in rows
            if row["work_id"] == supplied.work_id
            and (supplied.family is None or row["family"] == supplied.family)
        ]
        if not matches:
            raise ValueError("condition work/family is outside the authorized population")
        for row in matches:
            key = row["work_id"], row["family"]
            if key in selections:
                raise ValueError("duplicate or overlapping condition work/family selection")
            selections[key] = {name: supplied.labels.get(name, Label()) for name in LABELS}
    return selections


def _build(config: Path) -> Record:
    policy = Preparation.model_validate_json(config.read_text(encoding="utf-8"))
    source, exported, source_path, exported_path, split, role = _source(policy)
    cohort_path = Path(exported["cohort_path"])
    cohort = load_cohort(cohort_path)
    assignments = {row.work_id: row for row in cohort.assignments}
    rows = cast(list[Record], source["records"])
    selections = _labels(policy, rows)
    seen: set[tuple[str, str]] = set()
    examples: list[Record] = []
    files = [config, source_path, exported_path, cohort_path]
    for row in rows:
        key = row["work_id"], row["family"]
        assignment = assignments.get(row["work_id"])
        if (
            key in seen
            or assignment is None
            or assignment.split != split
            or assignment.author_id != row["author_id"]
            or assignment.text_sha256 != row["source_sha256"]
            or (role == "optimizer" and (not assignment.eligible or not row["eligible"]))
        ):
            raise ValueError("duplicate or protected/stale work/family/author/source")
        seen.add(key)
        labels = selections.get(key, {name: Label() for name in LABELS})
        values = {name: label.value for name, label in labels.items() if label.value is not None}
        conditions = Conditions.model_validate(values)
        annotation = Path(row["annotation_path"])
        plan = NarrativePlan.model_validate_json(annotation.read_text(encoding="utf-8"))
        if {item.sha256 for item in plan.sources} != {assignment.text_sha256}:
            raise ValueError("annotation source differs from its frozen work")
        target = encode_plan(
            plan,
            exported["target_policy"],
            row,
            max_scenes=policy.max_scenes,
            max_tokens=policy.max_tokens,
        )
        provenance = {
            **{field: row[field] for field in ("work_id", "author_id", "source_sha256", "family")},
            "schema_id": SCHEMA_ID,
            "target_policy_id": identity(exported["target_policy"]),
            "parent_export_id": exported["id"],
            "annotation_path": str(annotation.resolve()),
            "split": split,
            "literary_truth": "unvalidated condition labels; retained approved annotation targets",
        }
        examples.append(
            seal(
                {
                    "kind": "conditional-example",
                    "conditions": conditions.model_dump(mode="json"),
                    "condition_provenance": {
                        name: label.model_dump(mode="json") for name, label in labels.items()
                    },
                    "target": target,
                    "provenance": provenance,
                    "work_id": row["work_id"],
                    "author_id": row["author_id"],
                    "family": row["family"],
                    "group_id": identity(
                        {"author_id": row["author_id"], "work_id": row["work_id"]}
                    ),
                }
            )
        )
        files.extend(
            [annotation, *(file for label in labels.values() for file in label.input_files)]
        )
    if not examples:
        raise ValueError("empty conditional population")
    return {
        "kind": "conditional-dataset",
        "representation": "nde-conditional-examples-v1",
        "condition_id": CONDITION_ID,
        "condition_specification": CONDITION_SPECIFICATION,
        "codec_id": CODEC_ID,
        "schema_id": SCHEMA_ID,
        "role": role,
        "split": split,
        "test_only": exported["test_only"],
        "export_id": exported["id"],
        "source_id": source["id"],
        "config_path": str(config.resolve()),
        "config": policy.model_dump(mode="json"),
        "examples": examples,
        "independent_works": len({row["work_id"] for row in examples}),
        "independent_authors": len({row["author_id"] for row in examples}),
        "inputs": snapshot(files),
    }


def prepare_dataset(config: Path, out: Path) -> Record:
    record = _build(config)
    path = out / "dataset.json"
    if path.exists():
        existing = load_dataset(path)
        if existing != seal(record):
            raise ValueError("completed examples differ; use a new output directory")
        return existing
    result = create_record(path, record)
    report_files(out, "Conditional training examples", result)
    return result


def load_dataset(path: Path, *, optimizer: bool = False) -> Record:
    result = read_sealed(_file(path, "dataset.json"))
    if result.get("kind") != "conditional-dataset":
        raise ValueError("not a conditional dataset")
    check_snapshot(result["inputs"])
    if optimizer and (result["role"] != "optimizer" or result["split"] != "train"):
        raise ValueError("evaluator-only conditions and plans cannot enter optimizer loaders")
    if result != seal(_build(Path(result["config_path"]))):
        raise ValueError("conditional examples differ from authorized source/configuration/schema")
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="conditional_data_command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        result = (
            prepare_dataset(args.config, args.out)
            if args.conditional_data_command == "prepare"
            else load_dataset(args.path)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde conditional-data: {error}", file=sys.stderr)
        return 2
