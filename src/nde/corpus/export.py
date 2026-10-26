"""Versioned training views preserve original plans and filter target channels explicitly."""

import argparse
import sys
from pathlib import Path
from typing import Any

from nde.corpus import g2, reliability
from nde.corpus.families import family_report
from nde.corpus.records_io import (
    check_snapshot,
    cohort_file,
    create_record,
    file_sha,
    read_sealed,
    snapshot,
)
from nde.corpus.splits import load_cohort
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan


def export_dataset(gate: Path, out: Path, *, test_only: bool = False) -> dict[str, Any]:

    decision = g2.validate_gate(gate)
    if test_only != decision["test_only"] or decision["criteria_outcome"] != "passed":
        raise ValueError(
            "export requires approved criteria; synthetic exports must be explicitly test-only"
        )
    if not test_only and decision["outcome"] != "passed":
        raise ValueError("pending/failed G2 cannot approve real training targets")
    policy = decision["policy"]
    pilot_dir, review_dir = Path(policy["pilot"]), Path(policy["review"])
    pilot_packet = read_sealed(pilot_dir / "pilot.json")
    cohort_path = cohort_file(Path(pilot_packet["config"]["cohort"]))
    cohort = load_cohort(cohort_path)
    assignments = {a.work_id: a for a in cohort.assignments}
    targets = {
        c: decision["channels"].get(c, {}).get("target_policy", "excluded")
        for c in reliability.CHANNELS
    }
    records: list[dict[str, Any]] = []
    files = [gate / "decision.json", review_dir / "packet.json", cohort_path]
    for item in pilot_packet["config"]["works"]:
        assignment = assignments[item["work_id"]]
        if assignment.split != "train" or not assignment.eligible:
            continue
        comparison = family_report(pilot_dir / "stories" / assignment.work_id)
        if comparison["status"] != "complete":
            raise ValueError("approved pilot annotation is incomplete")
        for family in comparison["families"]:
            path = Path(family["directory"]) / "annotation" / "plan.json"
            plan = NarrativePlan.model_validate_json(path.read_text(encoding="utf-8"))
            if not validate_plan(plan).valid:
                raise ValueError("export annotation fails narrative validation")
            if not plan.sources or plan.sources[0].sha256 != assignment.text_sha256:
                raise ValueError("export annotation source differs from protected cohort")
            files.append(path)
            records.append(
                {
                    "work_id": assignment.work_id,
                    "author_id": assignment.author_id,
                    "family": family["family"],
                    "split": "train",
                    "eligible": True,
                    "annotation_path": str(path.resolve()),
                    "annotation_sha256": file_sha(path),
                    "source_sha256": assignment.text_sha256,
                    "schema_sha256": pilot_packet["schema_sha256"],
                    "targets": {c: policy for c, policy in targets.items() if policy != "excluded"},
                    "representation": (
                        "original source-linked annotation with filtered training target policy; "
                        "not a reduced tier-L plan"
                    ),
                }
            )
    if not records:
        raise ValueError("no eligible training annotations in approved cohort")
    return create_record(
        out / "dataset.json",
        {
            "test_only": test_only,
            "gate_path": str(gate.resolve()),
            "gate_id": decision["id"],
            "cohort_path": str(cohort_path.resolve()),
            "cohort_sha256": file_sha(cohort_path),
            "reliability_path": str(review_dir.resolve()),
            "reliability_sha256": file_sha(review_dir / "packet.json"),
            "target_policy": targets,
            "records": records,
            "inputs": snapshot(files),
        },
    )


def validate_export(path: Path) -> dict[str, Any]:

    path = path / "dataset.json" if path.is_dir() else path
    dataset = read_sealed(path)
    check_snapshot(dataset["inputs"])
    gate = g2.validate_gate(Path(dataset["gate_path"]))
    if gate["id"] != dataset["gate_id"] or gate["test_only"] != dataset["test_only"]:
        raise ValueError("export gate identity changed")
    if gate["criteria_outcome"] != "passed" or (
        not dataset["test_only"] and gate["outcome"] != "passed"
    ):
        raise ValueError("export no longer has approval")
    cohort = load_cohort(Path(dataset["cohort_path"]))
    assignments = {a.work_id: a for a in cohort.assignments}
    protected_authors = {
        a.author_id for a in cohort.assignments if a.split not in {"train", "excluded"}
    }
    for row in dataset["records"]:
        assignment = assignments.get(row["work_id"])
        if (
            assignment is None
            or assignment.split != "train"
            or not assignment.eligible
            or assignment.author_id != row["author_id"]
            or row["author_id"] in protected_authors
            or assignment.text_sha256 != row["source_sha256"]
            or row["split"] != "train"
        ):
            raise ValueError("protected/stale work or author cannot be a training target/exemplar")
        if row["targets"] != {c: p for c, p in dataset["target_policy"].items() if p != "excluded"}:
            raise ValueError("export target policy changed")
        if any(
            dataset["target_policy"].get(c)
            != gate["channels"].get(c, {}).get("target_policy", "excluded")
            for c in reliability.CHANNELS
        ):
            raise ValueError("export policy differs from current channel classification")
    return dataset


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="export_command", required=True)
    create = commands.add_parser("create")
    create.add_argument("gate", type=Path)
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--test-only", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.export_command == "create":
            export_dataset(args.gate, args.out, test_only=args.test_only)
        else:
            validate_export(args.path)
        return 0
    except (OSError, ValueError) as error:
        print(f"nde corpus-export: {error}", file=sys.stderr)
        return 2
