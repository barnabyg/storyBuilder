"""Author-safe reconstruction examples, with separately authorized evaluation views."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nde.corpus.export import validate_export
from nde.corpus.records_io import (
    check_snapshot,
    cohort_file,
    create_record,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.corpus.splits import load_cohort
from nde.llm.config import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.reconstruction.codec import encode_plan, load_encoding
from nde.reconstruction.tasks import make_task


class EvaluationAnnotation(Strict):
    work_id: str
    family: str
    path: Path


class EvaluationAuthorization(Strict):
    """Explicit owner-supplied permission to read a protected evaluation population."""

    export: Path
    cohort: Path
    split: Literal["validation", "test"]
    authorized_by: str = Field(min_length=1)
    purpose: Literal["reconstruction-evaluation"] = "reconstruction-evaluation"
    annotations: list[EvaluationAnnotation] = Field(min_length=1)
    test_only: bool = False


class Preparation(Strict):
    encoding: Path | None = None
    evaluation: Path | None = None
    granularity: Literal["whole-story", "window", "scene"] = "whole-story"
    window_scenes: int = Field(default=3, ge=1)
    stride: int = Field(default=1, ge=1)
    seed: int = 7
    schedules: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"kind": "fields", "rate": 0.3}]
    )
    repetitions: int = Field(default=1, ge=1, le=100)
    max_scenes: int = Field(default=256, ge=1)
    max_tokens: int = Field(default=65536, ge=1)


def _path(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def authorize_evaluation(config: Path, out: Path) -> dict[str, Any]:
    policy = EvaluationAuthorization.model_validate_json(config.read_text(encoding="utf-8"))
    export = validate_export(policy.export)
    cohort_path = cohort_file(policy.cohort).resolve()
    if cohort_path != Path(export["cohort_path"]).resolve():
        raise ValueError("evaluation must retain the approved frozen cohort")
    if policy.test_only != export["test_only"]:
        raise ValueError("evaluation and export synthetic status differ")
    cohort = load_cohort(cohort_path)
    assignments = {row.work_id: row for row in cohort.assignments}
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    files = [config, _path(policy.export, "dataset.json"), cohort_path]
    for annotation in policy.annotations:
        row = assignments.get(annotation.work_id)
        key = annotation.work_id, annotation.family
        if row is None or row.split != policy.split or key in seen:
            raise ValueError("evaluation work/family is duplicated or outside authorized split")
        seen.add(key)
        plan = NarrativePlan.model_validate_json(annotation.path.read_text(encoding="utf-8"))
        if not validate_plan(plan).valid or not plan.sources:
            raise ValueError("evaluation annotation must be a valid source-linked plan")
        if {source.sha256 for source in plan.sources} != {row.text_sha256}:
            raise ValueError("evaluation annotation source differs from cohort")
        files.append(annotation.path)
        records.append(
            {
                "work_id": row.work_id,
                "author_id": row.author_id,
                "source_sha256": row.text_sha256,
                "family": annotation.family,
                "split": row.split,
                "annotation_path": str(annotation.path.resolve()),
            }
        )
    return create_record(
        out / "authorization.json",
        {
            "kind": "reconstruction-evaluation-authorization",
            "role": "evaluator-only",
            "policy": policy.model_dump(mode="json"),
            "export_path": str(_path(policy.export, "dataset.json").resolve()),
            "export_id": export["id"],
            "cohort_path": str(cohort_path),
            "test_only": policy.test_only,
            "split": policy.split,
            "target_policy": export["target_policy"],
            "records": records,
            "inputs": snapshot(files),
        },
    )


def _window(plan: NarrativePlan, start: int, end: int) -> NarrativePlan:
    """Retain local annotations; out-of-view links remain unavailable, never negative labels."""
    data = plan.model_dump(mode="json")
    data["scenes"] = data["scenes"][start:end]
    scenes = {scene.id for scene in plan.scenes[start:end]}
    data["causal_links"] = [
        row
        for row in data["causal_links"]
        if {row["cause_scene_id"], row["effect_scene_id"]} <= scenes
    ]
    data["setups"] = [row for row in data["setups"] if row["scene_id"] in scenes]
    setups = {row["id"] for row in data["setups"]}
    data["payoff_links"] = [
        row
        for row in data["payoff_links"]
        if row["setup_id"] in setups and row["payoff_scene_id"] in scenes
    ]
    # Knowledge can depend on earlier reveals. Retain only initial-holder-safe local uses.
    data["reveal_links"] = [row for row in data["reveal_links"] if row["scene_id"] in scenes]
    initial = {fact.id: set(fact.initial_holder_ids) for fact in plan.knowledge_facts}
    data["knowledge_uses"] = [
        row
        for row in data["knowledge_uses"]
        if row["scene_id"] in scenes and row["holder_id"] in initial.get(row["fact_id"], set())
    ]
    for field in ("states", "relationships"):
        data[field] = [row for row in data[field] if row["scene_id"] in scenes]
    for field in (
        "arcs",
        "reader_affect",
        "tension",
        "themes",
        "motifs",
        "discourse",
        "subtext",
    ):
        data[field] = [
            row
            for row in data[field]
            if set(row["scene_ids"]) <= scenes
            and (
                row.get("trajectory") is None
                or {point["scene_id"] for point in row["trajectory"]["points"]} <= scenes
            )
        ]
    motif_ids = {row["id"] for row in data["motifs"]}
    data["motif_echoes"] = [
        row
        for row in data["motif_echoes"]
        if {row["from_motif_id"], row["to_motif_id"]} <= motif_ids
    ]
    data["constraints"] = []
    return NarrativePlan.model_validate(data)


def _examples(
    record: dict[str, Any], policy: Preparation, targets: dict[str, str]
) -> list[dict[str, Any]]:
    provenance = record["provenance"]
    plan = NarrativePlan.model_validate_json(
        Path(provenance["annotation_path"]).read_text(encoding="utf-8")
    )
    count = len(plan.scenes)
    width = (
        count
        if policy.granularity == "whole-story"
        else (1 if policy.granularity == "scene" else policy.window_scenes)
    )
    starts = [0] if policy.granularity == "whole-story" else range(0, count, policy.stride)
    examples: list[dict[str, Any]] = []
    for start in starts:
        end = min(count, start + width)
        partial = start != 0 or end != count
        local = _window(plan, start, end) if partial else plan
        original = plan.model_dump(mode="json")
        retained = local.model_dump(mode="json")
        metadata = {
            **provenance,
            "scene_start": start,
            "scene_end": end,
            "boundary_context": "out-of-view unknown" if partial else "complete work",
            "boundary_omissions": {
                field: [
                    row["id"]
                    for row in cast(list[dict[str, Any]], values)
                    if row not in retained[field]
                ]
                for field, values in original.items()
                if partial and isinstance(values, list) and field not in {"sources", "characters"}
            },
            "source_offsets": [
                span.model_dump(mode="json")
                for scene in plan.scenes[start:end]
                for span in scene.evidence
            ],
        }
        encoded = encode_plan(
            local,
            targets,
            metadata,
            max_scenes=policy.max_scenes,
            max_tokens=policy.max_tokens,
            ineligible_channels=("causality", "information") if partial else (),
        )
        tasks = [
            make_task(encoded, schedule, seed=policy.seed, repetition=repetition)
            for schedule in policy.schedules
            for repetition in range(policy.repetitions)
        ]
        example = {
            "record": encoded,
            "tasks": tasks,
            "scene_start": start,
            "scene_end": end,
            **{key: provenance[key] for key in ("work_id", "author_id", "source_sha256", "family")},
        }
        example["id"] = identity(example)
        examples.append(example)
    return examples


def prepare_dataset(config: Path, out: Path) -> dict[str, Any]:
    policy = Preparation.model_validate_json(config.read_text(encoding="utf-8"))
    if (policy.encoding is None) == (policy.evaluation is None) or not policy.schedules:
        raise ValueError("choose training encoding or authorized evaluation and nonempty schedules")
    if policy.encoding is not None:
        source_path = _path(policy.encoding, "encoding.json")
        source = load_encoding(source_path)
        export_path = Path(source["export_path"])
        export = validate_export(export_path)
        records = source["records"]
        split, role = "train", "optimizer"
    else:
        if policy.evaluation is None:
            raise ValueError("missing evaluation authorization")
        source_path = _path(policy.evaluation, "authorization.json")
        source = read_sealed(source_path)
        check_snapshot(source["inputs"])
        export_path = Path(source["export_path"])
        export = validate_export(export_path)
        if source["export_id"] != export["id"] or source["role"] != "evaluator-only":
            raise ValueError("stale evaluation authorization")
        records = [
            encode_plan(
                NarrativePlan.model_validate_json(Path(row["annotation_path"]).read_text("utf-8")),
                source["target_policy"],
                row,
                max_scenes=policy.max_scenes,
                max_tokens=policy.max_tokens,
            )
            for row in source["records"]
        ]
        split, role = source["split"], "evaluator-only"
    examples = [
        example
        for record in records
        for example in _examples(record, policy, export["target_policy"])
    ]
    if not examples:
        raise ValueError("empty reconstruction population")
    record = create_record(
        out / "dataset.json",
        {
            "kind": "reconstruction-dataset",
            "role": role,
            "split": split,
            "encoding_id": source["id"],
            "source_path": str(source_path.resolve()),
            "source_kind": "encoding" if role == "optimizer" else "authorization",
            "export_path": str(export_path.resolve()),
            "export_id": export["id"],
            "cohort_path": export["cohort_path"],
            "test_only": export["test_only"],
            "codec_id": examples[0]["record"]["codec_id"],
            "vocabulary_id": examples[0]["record"]["vocabulary_id"],
            "granularity": policy.granularity,
            "config": policy.model_dump(mode="json"),
            "examples": examples,
            "inputs": snapshot([config, source_path, export_path]),
            "independent_works": len({row["work_id"] for row in examples}),
            "independent_authors": len({row["author_id"] for row in examples}),
            "boundary_policy": (
                "partial graph channels have zero target eligibility; outside is unknown"
            ),
        },
    )
    load_dataset(out)
    report_files(out, "Reconstruction dataset", record)
    return record


def load_dataset(path: Path, *, optimizer: bool = False) -> dict[str, Any]:
    dataset = read_sealed(_path(path, "dataset.json"))
    if dataset.get("kind") != "reconstruction-dataset":
        raise ValueError("not a reconstruction dataset")
    check_snapshot(dataset["inputs"])
    export = validate_export(Path(dataset["export_path"]))
    if export["id"] != dataset["export_id"]:
        raise ValueError("dataset export identity changed")
    if optimizer and (dataset["role"] != "optimizer" or dataset["split"] != "train"):
        raise ValueError("evaluator-only labels cannot enter optimizer loaders")
    source_path = Path(dataset["source_path"])
    source = (
        load_encoding(source_path)
        if dataset["source_kind"] == "encoding"
        else read_sealed(source_path)
    )
    if source["id"] != dataset["encoding_id"]:
        raise ValueError("dataset source identity changed")
    check_snapshot(source["inputs"])
    expected_split = "train" if dataset["source_kind"] == "encoding" else source["split"]
    expected_role = "optimizer" if expected_split == "train" else "evaluator-only"
    if dataset["split"] != expected_split or dataset["role"] != expected_role:
        raise ValueError("dataset role differs from its authorized source")
    cohort = load_cohort(Path(dataset["cohort_path"]))
    assignments = {row.work_id: row for row in cohort.assignments}
    for example in dataset["examples"]:
        row = assignments.get(example["work_id"])
        if (
            row is None
            or row.split != dataset["split"]
            or row.author_id != example["author_id"]
            or row.text_sha256 != example["source_sha256"]
        ):
            raise ValueError("reconstruction work/author/source escaped its frozen split")
        if example["id"] != identity(example) or example["record"]["id"] != identity(
            example["record"]
        ):
            raise ValueError("example identity changed")
        if any(task["id"] != identity(task) for task in example["tasks"]):
            raise ValueError("task identity changed")
    policy = Preparation.model_validate(dataset["config"])
    if expected_role == "optimizer":
        records = source["records"]
    else:
        records = [
            encode_plan(
                NarrativePlan.model_validate_json(
                    Path(row["annotation_path"]).read_text(encoding="utf-8")
                ),
                export["target_policy"],
                row,
                max_scenes=policy.max_scenes,
                max_tokens=policy.max_tokens,
            )
            for row in source["records"]
        ]
    expected = [
        example
        for record in records
        for example in _examples(record, policy, export["target_policy"])
    ]
    if dataset["examples"] != expected:
        raise ValueError("dataset examples differ from authorized sources or masking policy")
    return dataset


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="data_command", required=True)
    for name in ("prepare", "authorize-evaluation"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--out", type=Path, required=True)
    commands.add_parser("inspect").add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.data_command == "authorize-evaluation":
            record = authorize_evaluation(args.config, args.out)
        elif args.data_command == "prepare":
            record = prepare_dataset(args.config, args.out)
        else:
            record = load_dataset(args.path)
        print(json.dumps(record, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"nde reconstruction-data: {error}", file=sys.stderr)
        return 2
