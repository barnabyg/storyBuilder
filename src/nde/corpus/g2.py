"""Registered G2 evidence binds review, coverage and costs without starting extraction."""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from nde.corpus import pilot, reliability
from nde.corpus.families import FamilyConfig
from nde.corpus.records_io import (
    check_snapshot,
    cohort_file,
    create_record,
    evidence_files,
    file_sha,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.corpus.splits import load_cohort
from nde.llm.config import Strict

ACCOUNTING_TOLERANCE_USD = 1e-10


class G2Policy(Strict):
    pilot: Path
    review: Path
    min_stories: int = Field(default=50, ge=1)
    min_owner_per_channel: int = Field(default=3, ge=1)
    min_coverage: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    owner_threshold: float = Field(default=0.8, ge=0.8, le=1, allow_inf_nan=False)
    scaling_size: int = Field(default=300, ge=300, le=1000)
    scaling_budget_usd: float = Field(gt=0, allow_inf_nan=False)


def _pilot_review_inputs(
    policy: G2Policy, packet: dict[str, Any], pilot_packet: dict[str, Any]
) -> list[Path]:
    check_snapshot(packet["inputs"])
    expected = {w["work_id"]: w for w in pilot_packet["config"]["works"]}
    inputs = packet["policy"]["inputs"]
    if {item["work_id"] for item in inputs} != set(expected):
        raise ValueError("review population must cover the fixed pilot population")
    if (
        cohort_file(Path(packet["policy"]["cohort"])).resolve()
        != cohort_file(Path(pilot_packet["config"]["cohort"])).resolve()
    ):
        raise ValueError("review cohort differs from registered pilot cohort")
    files: list[Path] = []
    for item in inputs:
        work = expected[item["work_id"]]
        families = policy.pilot / "stories" / item["work_id"]
        feature = policy.pilot / "features" / item["work_id"] / "features.json"
        required = {
            "work_dir": Path(work["work_dir"]),
            "segmentation": Path(work["segmentation"]),
            "families": families,
            "features": feature,
        }
        if any(Path(item[key]).resolve() != path.resolve() for key, path in required.items()):
            raise ValueError(
                "review evidence differs from exact pilot annotation/feature/source inputs"
            )
        files += [feature, Path(work["segmentation"]), *evidence_files(families)]
    return files


def register(config: Path, out: Path) -> dict[str, Any]:
    policy = G2Policy.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    packet = read_sealed(policy.review / "packet.json")
    if list((policy.review / "judgements").glob("*.json")):
        raise ValueError("G2 thresholds must be frozen before collecting owner labels")
    if set(reliability.CORE) - set(packet["policy"]["channels"]):
        raise ValueError("G2 registration requires all core channels")
    pilot_packet = read_sealed(policy.pilot / "pilot.json")
    files = _pilot_review_inputs(policy, packet, pilot_packet)
    return create_record(
        out,
        {
            "policy": policy.model_dump(mode="json"),
            "inputs": snapshot(
                [config, policy.review / "packet.json", policy.pilot / "pilot.json", *files]
            ),
            "test_only": packet["test_only"] or pilot_packet["test_only"],
        },
    )


def assess(protocol: Path) -> dict[str, Any]:
    registered = read_sealed(protocol)
    policy = G2Policy.model_validate(registered["policy"])
    pending: list[str] = []
    review: dict[str, Any]
    extraction: dict[str, Any]
    try:
        check_snapshot(registered["inputs"])
        _pilot_review_inputs(
            policy,
            read_sealed(policy.review / "packet.json"),
            read_sealed(policy.pilot / "pilot.json"),
        )
        review = reliability.report(policy.review)
        extraction = pilot.report(policy.pilot)
    except (OSError, ValueError) as error:
        pending.append(f"registered evidence unavailable: {error}")
        review = {
            "id": None,
            "channels": {},
            "core_failures": [],
            "test_only": registered["test_only"],
            "available": False,
        }
        extraction = {
            "id": None,
            "test_only": registered["test_only"],
            "available": False,
            "completed_stories": 0,
            "coverage": 0,
            "status": "unavailable",
            "unresolved_accounting_usd": None,
            "scaling": {"per_completed_story_usd": None, "incomplete_data": True},
        }
    failed: list[str] = []
    for channel in reliability.CORE:
        row = review["channels"].get(channel)
        if row is None or row["classification"] == "pending":
            pending.append(f"{channel}: required reliability evidence missing/stale")
        elif row["classification"] != "reliable":
            failed.append(f"{channel}: core channel {row['classification']}; stop before training")
        if row is not None:
            if row["known_owner"] < policy.min_owner_per_channel:
                pending.append(f"{channel}: owner denominator below registered minimum")
            elif row["owner_rate"] is None or row["owner_rate"] < policy.owner_threshold:
                failed.append(f"{channel}: accurate-or-defensible rate below threshold")
    if (
        extraction["completed_stories"] < policy.min_stories
        or extraction["coverage"] < policy.min_coverage
    ):
        pending.append("pilot coverage below registered minimum")
    if (
        extraction["status"] != "complete"
        or extraction["unresolved_accounting_usd"] is None
        or extraction["unresolved_accounting_usd"] > ACCOUNTING_TOLERANCE_USD
    ):
        pending.append("pilot incomplete or accounting unresolved")
    estimate = extraction["scaling"]["per_completed_story_usd"]
    if estimate is None or extraction["scaling"]["incomplete_data"]:
        pending.append("complete scaling-cost evidence unavailable")
    elif estimate * policy.scaling_size > policy.scaling_budget_usd:
        failed.append("projected scaling cost exceeds registered budget")
    criteria = "pending" if pending else "failed" if failed else "passed"
    test_only = registered["test_only"] or review["test_only"] or extraction["test_only"]
    if test_only:
        pending.append("synthetic evidence cannot establish G2")
    result: dict[str, Any] = {
        "version": "1",
        "protocol_path": str(protocol.resolve()),
        "protocol_sha256": file_sha(protocol),
        "test_only": test_only,
        "outcome": "pending" if pending else criteria,
        "criteria_outcome": criteria,
        "pending": pending,
        "failures": failed,
        "policy": registered["policy"],
        "reliability_id": review["id"],
        "pilot_report_id": extraction["id"],
        "channels": review["channels"],
        "scaling": extraction["scaling"],
        "action": "stop before training: core causal/knowledge evidence failed"
        if review["core_failures"]
        else "await required evidence"
        if pending
        else "controlled expansion may be invoked separately"
        if criteria == "passed"
        else "retain current corpus; criteria failed",
    }
    result["id"] = identity(result)
    return result


def write_decision(protocol: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise ValueError("G2 report exists; preserve it and create a new version")
    result = assess(protocol)
    policy = G2Policy.model_validate(result["policy"])
    files = [protocol, *evidence_files(policy.review), *evidence_files(policy.pilot)]
    result["inputs"] = snapshot(files)
    result = create_record(out / "decision.json", result)
    report_files(out, "G2 corpus readiness", result)
    return result


def validate_gate(directory: Path) -> dict[str, Any]:
    saved = read_sealed(directory / "decision.json")
    check_snapshot(saved["inputs"])
    replay = assess(Path(saved["protocol_path"]))
    for key, value in replay.items():
        if key != "id" and saved.get(key) != value:
            raise ValueError(f"G2 decision no longer reproduces: {key}")
    return saved


def scale(gate: Path, config: Path, out: Path) -> dict[str, Any]:

    decision = validate_gate(gate)
    if decision["outcome"] != "passed" or decision["test_only"]:
        raise ValueError("only a genuine passed G2 decision can authorise real scaling")
    planned = pilot.PilotConfig.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    if planned.test_only:
        raise ValueError("real scaling cannot use synthetic configuration")
    prior = pilot.PilotConfig.model_validate(
        read_sealed(Path(decision["policy"]["pilot"]) / "pilot.json")["config"]
    )
    old_family = FamilyConfig.model_validate(
        yaml.safe_load(prior.extraction_config.read_text(encoding="utf-8"))
    )
    new_family = FamilyConfig.model_validate(
        yaml.safe_load(planned.extraction_config.read_text(encoding="utf-8"))
    )
    if old_family != new_family:
        raise ValueError("expansion extraction configuration differs from approved pilot")
    old, new = load_cohort(prior.cohort), load_cohort(planned.cohort)
    old_rows = {a.work_id: a for a in old.assignments}
    new_rows = {a.work_id: a for a in new.assignments}
    if any(new_rows.get(key) != value for key, value in old_rows.items()):
        raise ValueError("expansion changed existing frozen assignments")
    if len(planned.works) > decision["policy"]["scaling_size"]:
        raise ValueError("expansion exceeds registered scaling size")
    if planned.per_run_usd > decision["policy"]["scaling_budget_usd"]:
        raise ValueError("expansion cap exceeds registered scaling budget")
    return pilot.execute(config, out, expansion=True)


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="g2_command", required=True)
    registration = commands.add_parser("register")
    registration.add_argument("config", type=Path)
    registration.add_argument("--out", type=Path, required=True)
    decision = commands.add_parser("report")
    decision.add_argument("protocol", type=Path)
    decision.add_argument("--out", type=Path, required=True)
    expansion = commands.add_parser("scale")
    expansion.add_argument("config", type=Path)
    expansion.add_argument("--gate", type=Path, required=True)
    expansion.add_argument("--out", type=Path, required=True)


def run(args: argparse.Namespace) -> int:
    try:
        if args.g2_command == "register":
            register(args.config, args.out)
        elif args.g2_command == "report":
            write_decision(args.protocol, args.out)
        else:
            scale(args.gate, args.config, args.out)
        return 0
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"nde corpus-g2: {error}", file=sys.stderr)
        return 2
