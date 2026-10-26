"""Registered G3 evidence; no training or automatic Phase 7 side effects."""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from nde import g1_report
from nde.corpus import g2
from nde.corpus.export import validate_export
from nde.corpus.records_io import (
    check_snapshot,
    create_record,
    evidence_files,
    file_sha,
    identity,
    read_sealed,
    report_files,
    snapshot,
)
from nde.llm.config import Strict
from nde.reconstruction.overlap import load_view
from nde.reconstruction.scoring import validate_scores

PRIMARY_MIN, PRIMARY_MAX = 0.3, 0.5
OBJECTIVES = {"denoising", "ar"}
FAIL_ACTION = (
    "Stop before Phase 7. Retain the registered C/D reference planner; "
    "preserve this protocol and all results. Do not extend the sample or "
    "launch another gate automatically."
)


class G3Policy(Strict):
    training: Path
    evaluation: Path
    base_model: Path
    experiment: Path
    score_reports: list[Path] = Field(min_length=1)
    overlap_report: Path
    reference: Path
    g1_batch: Path | None = None
    g1_realisation: list[Path] = Field(default_factory=list[Path])
    g2_gate: Path | None = None
    owner_review: Path | None = None
    require_meaning_review: bool = False
    minimum_owner_samples: int = Field(default=1, ge=1)
    meaning_preservation_minimum: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    validator_target: float = Field(default=0.95, ge=0, le=1, allow_inf_nan=False)
    positive_margin: float = Field(default=0.01, gt=0, le=1, allow_inf_nan=False)
    aggregation: str = "work"
    checkpoint_selection: str = "registered final step; validation-only tuning"
    test_only: bool = False


def _file(path: Path, name: str) -> Path:
    return path / name if path.is_dir() else path


def _matrix_binding(policy: G3Policy, experiment: dict[str, Any]) -> dict[str, Any]:
    config = experiment["config"]
    if len(config["train_datasets"]) != 1:
        raise ValueError("G3 requires one audited training population per registered gate")
    if (
        _file(Path(config["base_model"]), "model.json").resolve()
        != _file(policy.base_model, "model.json").resolve()
        or _file(Path(config["evaluation_dataset"]), "dataset.json").resolve()
        != _file(policy.evaluation, "dataset.json").resolve()
        or _file(policy.training, "dataset.json").resolve()
        not in {_file(Path(path), "dataset.json").resolve() for path in config["train_datasets"]}
        or config.get("selection_split") != "validation"
    ):
        raise ValueError("G3 matrix base/data or validation-only selection differs")
    jobs = {job["id"]: job for job in experiment["jobs"]}
    assigned = [job_id for pair in experiment["pairs"] for job_id in pair["job_ids"]]
    if (
        len(jobs) != len(experiment["jobs"])
        or len(assigned) != len(set(assigned))
        or set(assigned) != set(jobs)
    ):
        raise ValueError("G3 matrix has duplicate, missing or unpaired jobs")
    pairs: dict[str, Any] = {}
    for pair in experiment["pairs"]:
        if (
            len(pair["job_ids"]) != len(OBJECTIVES)
            or {jobs[job_id]["objective"] for job_id in pair["job_ids"]} != OBJECTIVES
        ):
            raise ValueError("G3 matrix requires matched denoising/AR objective pairs")
        training = load_view(Path(pair["training_dataset_path"]))
        if training["id"] != pair["training_dataset_id"]:
            raise ValueError("G3 matrix training subset identity changed")
        pairs[str(Path(pair["score_path"]).resolve())] = pair
    if len(pairs) != len(experiment["pairs"]):
        raise ValueError("G3 matrix score paths must be unique")
    return pairs


def _gate_evidence(policy: G3Policy) -> dict[str, Any]:
    pending: list[str] = []
    files: list[Path] = []
    gate_ids: dict[str, str | None] = {"g1": None, "g2": None}
    gate_test_only = False
    if policy.g1_batch is None:
        pending.append("genuine passed G1 evidence is missing")
    else:
        decision = g1_report.decision(policy.g1_batch, policy.g1_realisation)
        gate_ids["g1"] = decision["id"]
        gate_test_only |= decision["test_only"]
        if decision["outcome"] != "passed" or decision["test_only"]:
            pending.append("G1 has not genuinely passed")
        files += evidence_files(policy.g1_batch)
        for path in policy.g1_realisation:
            files += evidence_files(path)
    if policy.g2_gate is None:
        pending.append("genuine passed G2 evidence is missing")
    else:
        decision = g2.validate_gate(policy.g2_gate)
        gate_ids["g2"] = decision["id"]
        gate_test_only |= decision["test_only"]
        if decision["outcome"] != "passed" or decision["test_only"]:
            pending.append("G2 has not genuinely passed")
        files += evidence_files(policy.g2_gate)
    return {
        "gate_ids": gate_ids,
        "test_only": gate_test_only,
        "pending": pending,
        "inputs": snapshot(files),
    }


def _export_lineage(
    policy: G3Policy,
    training: dict[str, Any],
    evaluation: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    original = training
    while "parent_path" in original:
        original = load_view(Path(original["parent_path"]))
    if original.get("export_path"):
        export = validate_export(Path(original["export_path"]))
        if export["id"] != evaluation.get("export_id") or (
            gate["gate_ids"]["g2"] is not None and export["gate_id"] != gate["gate_ids"]["g2"]
        ):
            raise ValueError("G3 training/evaluation export differs from registered G2 evidence")
    elif not policy.test_only:
        raise ValueError("G3 is not scientifically ready: approved training export lineage missing")


def _readiness(policy: G3Policy) -> dict[str, Any]:
    training, evaluation = load_view(policy.training), load_view(policy.evaluation)
    if training.get("role") != "optimizer" or training.get("split") != "train":
        raise ValueError("G3 training must use optimizer-only training data")
    if evaluation.get("role") != "evaluator-only" or evaluation.get("split") != "test":
        raise ValueError("G3 requires protected fixed test data")
    if training.get("codec_id") != evaluation.get("codec_id"):
        raise ValueError("G3 data codec identities differ")
    base_file = _file(policy.base_model, "model.json")
    base, experiment = read_sealed(base_file), read_sealed(policy.experiment)
    weights_file = base_file.parent / "weights.json"
    if (
        file_sha(weights_file) != base["weights_sha256"]
        or identity(read_sealed(weights_file)["weights"]) != base["parameter_sha256"]
    ):
        raise ValueError("local base weights changed")
    check_snapshot(experiment.get("inputs", {}))
    if experiment.get("kind") != "reconstruction-experiment" or not experiment.get("jobs"):
        raise ValueError("registered experiment matrix is missing")
    pairs = _matrix_binding(policy, experiment)
    if {str(path.resolve()) for path in policy.score_reports} != {
        str(Path(pair["score_path"]).resolve()) for pair in experiment["pairs"]
    }:
        raise ValueError("G3 score coverage must equal the frozen paired experiment matrix")
    if policy.aggregation not in {"work", "author"}:
        raise ValueError("G3 aggregation must group by work or author")
    if not policy.reference.is_file():
        raise ValueError("identified C/D reference baseline is missing")
    gate = _gate_evidence(policy)
    _export_lineage(policy, training, evaluation, gate)
    pending = gate["pending"]
    files = [
        _file(policy.training, "dataset.json"),
        _file(policy.evaluation, "dataset.json"),
        base_file,
        weights_file,
        policy.experiment,
        policy.reference,
    ]
    test_only = (
        policy.test_only
        or gate["test_only"]
        or training.get("test_only", False)
        or evaluation.get("test_only", False)
        or base.get("test_only", False)
        or experiment.get("test_only", False)
    )
    if test_only:
        pending.append("synthetic/test-only data or model cannot establish scientific G3")
    task_ids = sorted(task["id"] for example in evaluation["examples"] for task in example["tasks"])
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("G3 requires nonempty unique fixed evaluation tasks")
    return {
        "training_id": training["id"],
        "evaluation_id": evaluation["id"],
        "base_id": base["id"],
        "initial_parameter_sha256": base["parameter_sha256"],
        "parameter_count": base["parameter_count"],
        "pairs": pairs,
        "inference_budget": {
            "max_tokens": experiment["config"]["training"].get("inference_max_tokens", 1000000)
        },
        "experiment_id": experiment["id"],
        "task_ids": task_ids,
        "codec_id": training["codec_id"],
        "gate_ids": gate["gate_ids"],
        "test_only": test_only,
        "pending": pending,
        "ready": not pending,
        "inputs": {**snapshot(files), **gate["inputs"]},
    }


def readiness(config: Path) -> dict[str, Any]:
    policy = G3Policy.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    return _readiness(policy)


def register(config: Path, out: Path) -> dict[str, Any]:
    policy = G3Policy.model_validate(yaml.safe_load(config.read_text(encoding="utf-8")))
    current = _readiness(policy)
    if not current["ready"] and not policy.test_only:
        raise ValueError("G3 is not scientifically ready: " + "; ".join(current["pending"]))
    if len(set(policy.score_reports)) != len(policy.score_reports) or any(
        path.exists() for path in policy.score_reports
    ):
        raise ValueError("freeze G3 before expected scores exist; duplicate reports are invalid")
    experiment = read_sealed(policy.experiment)
    if any(
        Path(job["prediction_path"]).exists()
        for job in experiment["jobs"]
        if "prediction_path" in job
    ):
        raise ValueError("freeze G3 before held-out predictions exist")
    if policy.owner_review is not None and policy.owner_review.exists():
        raise ValueError("freeze G3 before collecting meaning-preservation judgements")
    return create_record(
        out,
        {
            "kind": "g3-protocol",
            "policy": policy.model_dump(mode="json"),
            "readiness": current,
            "test_only": current["test_only"],
            "inputs": {**current["inputs"], **snapshot([config])},
            "primary_rate_range": [PRIMARY_MIN, PRIMARY_MAX],
            "analysis": (
                "macro means by registered work/author; candidate minus comparator must "
                "meet positive margin on each typed long-range edge family"
            ),
            "fail_action": FAIL_ACTION,
        },
    )


def _primary(rows: list[dict[str, Any]], aggregation: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if PRIMARY_MIN <= row["mask"]["actual_rate"] <= PRIMARY_MAX
        and PRIMARY_MIN <= row["mask"]["requested_rate"] <= PRIMARY_MAX
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    key = "author_id" if aggregation == "author" else "work_id"
    for row in selected:
        groups[row["provenance"][key]].append(row)
    result: dict[str, Any] = {
        "tasks": len(selected),
        "groups": len(groups),
        "outside_primary": len(rows) - len(selected),
    }
    for side in ("candidate", "comparator"):
        validators: list[float] = []
        edge_values: dict[str, list[float]] = defaultdict(list)
        for group in groups.values():
            validators.append(sum(row[side]["valid"] for row in group) / len(group))
            for edge in ("causal", "setup_payoff", "knowledge"):
                available = [
                    row[side]["long_range_edges"][edge]["f1"]
                    for row in group
                    if row[side]["long_range_edges"][edge]["expected"]
                ]
                if available:
                    edge_values[edge].append(sum(available) / len(available))
        result[side] = {
            "validator": sum(validators) / len(validators) if validators else None,
            "edges": {
                edge: sum(values) / len(values) if values else None
                for edge, values in edge_values.items()
            },
        }
    return result


def _collect_scores(
    policy: G3Policy, registered: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    pending: list[str] = []
    scores: list[dict[str, Any]] = []
    expected_ids = set(registered["readiness"]["task_ids"])
    for path in policy.score_reports:
        try:
            score = read_sealed(path)
            check_snapshot(score.get("inputs", {}))
            if "source_paths" in score:
                validate_scores(path)
            elif not score.get("test_only", False):
                raise ValueError(
                    "real score evidence lacks reproducible prediction/checkpoint bindings"
                )
            if (
                score.get("kind") != "paired-reconstruction-scores"
                or score.get("dataset_id") != registered["readiness"]["evaluation_id"]
                or score.get("codec_id") != registered["readiness"]["codec_id"]
            ):
                raise ValueError("score dataset/codec differs from registration")
            ids = [row["task_id"] for row in score["rows"]]
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                raise ValueError("incomplete expected result coverage")
            pending += _matching_problems(registered, score, path)
            if score.get("test_only", False):
                pending.append("synthetic/test-only score cannot establish scientific G3")
            if any(
                row[side]["status"] != "complete"
                for row in score["rows"]
                for side in ("candidate", "comparator")
            ):
                pending.append("missing or incomplete reconstruction results remain in denominator")
            scores.append(score)
        except (OSError, ValueError, KeyError) as error:
            pending.append(f"missing/inconsistent score {path}: {error}")
    return scores, pending


def _matching_problems(registered: dict[str, Any], score: dict[str, Any], path: Path) -> list[str]:
    pending: list[str] = []
    if score["matching_mismatches"]:
        pending.append("unequal model/data/budget configuration")
    frozen = registered["readiness"]
    pair = frozen["pairs"][str(path.resolve())]
    inference_budget = pair.get("inference_budget", frozen["inference_budget"])
    expected = {
        "base_id": frozen["base_id"],
        "initial_parameter_sha256": frozen["initial_parameter_sha256"],
        "parameter_count": frozen["parameter_count"],
        "dataset_id": pair["training_dataset_id"],
        "task_set_id": pair["training_task_set_id"],
        "seed": pair["seed"],
        "training_budget": pair["training_budget"],
        "inference_budget": inference_budget,
    }
    if any(score["matching"].get(key) != value for key, value in expected.items()):
        pending.append("score configuration differs from frozen paired job")
    budget = pair["training_budget"]
    for side in ("candidate", "comparator"):
        work = score["work"][side]
        if work["training_steps"] != budget["max_steps"]:
            pending.append(f"{side}: selected checkpoint has incomplete registered training")
        for field, cap in (
            ("training_tokens", "max_tokens"),
            ("training_attention_cells", "max_attention_cells"),
            ("training_elapsed_seconds", "max_seconds"),
        ):
            if work[field] > budget[cap]:
                pending.append(f"{side}: recorded {field} exceeds registered budget")
        if work["inference_tokens"] > inference_budget["max_tokens"]:
            pending.append(f"{side}: inference budget exceeded")
        if "max_seconds" in inference_budget and (
            "inference_elapsed_seconds" not in work
            or work["inference_elapsed_seconds"] > inference_budget["max_seconds"]
        ):
            pending.append(f"{side}: inference time budget exceeded or unavailable")
    return pending


def _overlap_status(policy: G3Policy, registered: dict[str, Any]) -> tuple[str | None, list[str]]:
    pending: list[str] = []
    overlap_id: str | None = None
    try:
        overlap = read_sealed(policy.overlap_report)
        check_snapshot(overlap.get("inputs", {}))
        if (
            overlap.get("training_id") != registered["readiness"]["training_id"]
            or overlap.get("query_id") != registered["readiness"]["evaluation_id"]
        ):
            raise ValueError("overlap population differs from registration")
        overlap_id = overlap["id"]
        if overlap["unresolved_overlap"]:
            pending.append("unresolved source/author overlap")
    except (OSError, ValueError, KeyError) as error:
        pending.append(f"overlap audit missing/stale: {error}")
    return overlap_id, pending


def _thresholds(primary: dict[str, Any], policy: G3Policy) -> tuple[list[str], list[str]]:
    pending: list[str] = []
    failures: list[str] = []
    if primary["tasks"] == 0:
        pending.append("no registered 30-50% actual-rate primary tasks")
    else:
        if primary["candidate"]["validator"] < policy.validator_target:
            failures.append("candidate validator pass rate below registered target")
        for edge in ("causal", "setup_payoff", "knowledge"):
            candidate = primary["candidate"]["edges"].get(edge)
            comparator = primary["comparator"]["edges"].get(edge)
            if candidate is None or comparator is None:
                pending.append(f"{edge}: no measurable long-range recovery denominator")
            elif candidate <= comparator or (
                candidate - comparator < policy.positive_margin
                and not math.isclose(
                    candidate - comparator, policy.positive_margin, rel_tol=0, abs_tol=1e-12
                )
            ):
                failures.append(f"{edge}: positive candidate-minus-comparator margin not met")
    return pending, failures


def _review_status(review: dict[str, Any], policy: G3Policy, scores: list[dict[str, Any]]) -> str:
    if set(review["score_ids"]) != {score["id"] for score in scores}:
        return "stale"
    if review.get("test_only", False):
        return "test-only"
    if review.get("kind") != "owner-meaning-review" or not review.get("reviewer"):
        return "unverified-owner"
    samples = review["samples"]
    expected = {(score["id"], row["task_id"]) for score in scores for row in score["rows"]}
    sample_ids = [(sample["score_id"], sample["task_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - expected:
        return "stale"
    known = [sample for sample in samples if sample["judgement"] in {"preserved", "changed"}]
    if len(known) < policy.minimum_owner_samples or len(known) != len(samples):
        return "pending"
    rate = sum(sample["judgement"] == "preserved" for sample in known) / len(known)
    return "complete" if rate >= policy.meaning_preservation_minimum else "failed"


def _owner_status(policy: G3Policy, scores: list[dict[str, Any]]) -> str:
    if policy.owner_review is None:
        return "missing"
    try:
        review = read_sealed(policy.owner_review)
        check_snapshot(review.get("inputs", {}))
        return _review_status(review, policy, scores)
    except (OSError, ValueError, KeyError):
        return "stale"


def assess(protocol: Path) -> dict[str, Any]:
    registered = read_sealed(protocol)
    if registered.get("kind") != "g3-protocol":
        raise ValueError("not a G3 protocol")
    policy = G3Policy.model_validate(registered["policy"])
    pending: list[str] = []
    try:
        check_snapshot(registered["inputs"])
        current = _readiness(policy)
        if current != registered["readiness"]:
            pending.append("registered readiness evidence changed")
        pending += current["pending"]
    except (OSError, ValueError, KeyError) as error:
        pending.append(f"stale readiness inputs: {error}")
    scores, score_pending = _collect_scores(policy, registered)
    overlap_id, overlap_pending = _overlap_status(policy, registered)
    primary = _primary([row for score in scores for row in score["rows"]], policy.aggregation)
    metric_pending, failures = _thresholds(primary, policy)
    pending += score_pending + overlap_pending + metric_pending
    owner_status = _owner_status(policy, scores)
    if policy.require_meaning_review and owner_status == "failed":
        failures.append("registered owner meaning-preservation threshold not met")
    elif policy.require_meaning_review and owner_status != "complete":
        pending.append("registered owner meaning-preservation review remains missing/stale")
    scientific_pending = [
        item
        for item in pending
        if "synthetic/test-only" not in item and not item.startswith("genuine passed")
    ]
    criteria = "pending" if scientific_pending else "failed" if failures else "passed"
    result = {
        "kind": "g3-decision",
        "protocol_id": registered["id"],
        "protocol_path": str(protocol.resolve()),
        "test_only": registered["test_only"]
        or any(score.get("test_only", False) for score in scores),
        "criteria_outcome": criteria,
        "outcome": "pending" if pending else criteria,
        "pending": pending,
        "failures": failures,
        "primary": primary,
        "score_ids": [score["id"] for score in scores],
        "overlap_id": overlap_id,
        "owner_review": {
            "status": owner_status,
            "path": str(policy.owner_review) if policy.owner_review is not None else None,
            "sha256": file_sha(policy.owner_review)
            if policy.owner_review is not None and policy.owner_review.is_file()
            else None,
        },
        "reference": str(policy.reference),
        "action": FAIL_ACTION
        if failures
        else "Preserve registered evidence; no automatic Phase 7 launch.",
        "inputs": snapshot(
            [
                protocol,
                *[path for path in policy.score_reports if path.is_file()],
                *([policy.overlap_report] if policy.overlap_report.is_file() else []),
                *(
                    [policy.owner_review]
                    if policy.owner_review is not None and policy.owner_review.is_file()
                    else []
                ),
            ]
        ),
    }
    result["id"] = identity(result)
    return result


def write_decision(protocol: Path, out: Path) -> dict[str, Any]:
    result = create_record(out / "decision.json", assess(protocol))
    report_files(out, "Registered G3 reconstruction experiment", result)
    return result


def configure_g3(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="g3_command", required=True)
    for name in ("readiness", "register"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "register":
            command.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("protocol", type=Path)
    report.add_argument("--out", type=Path, required=True)


def run_g3(args: argparse.Namespace) -> int:
    try:
        if args.g3_command == "readiness":
            result = readiness(args.config)
        elif args.g3_command == "register":
            result = register(args.config, args.out)
        else:
            result = write_decision(args.protocol, args.out)
        print(result.get("outcome", "ready" if result.get("ready") else "pending"))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"nde reconstruction-g3: {error}", file=sys.stderr)
        return 2
