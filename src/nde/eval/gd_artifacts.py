"""Registered generation and sealed G-D admission using the existing reader."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nde.controllability import replay_report
from nde.eval.batch import BatchError, Scoring, Source, is_closed, read_key, read_view
from nde.eval.g1 import file_checksum
from nde.eval.gd import GDInputs, GDProtocol, require_gd
from nde.eval.readiness import (
    GateBatchSettings,
    ReadinessReport,
    Record,
    build_gate_batch,
    check_readiness,
    cohort_checksum,
)
from nde.eval.rubric import Rubric
from nde.files import write_text_lf
from nde.narrative.ledger import plan_identity
from nde.narrative.workflow import load_plan
from nde.runs.config import load_config
from nde.runs.records import read_manifest, read_stories
from nde.runs.runner import RunOptions, run_experiment

GD_BINDING = "gd.json"
GD_GENERATION = "gd-generation.json"


class GenerationBinding(Record):
    schema_version: Literal["1"] = "1"
    protocol_sha256: str
    config_sha256: str
    g1_decision_sha256: str
    selection_sha256: str
    test_only: bool
    manifest_sha256: str


class GDBinding(Record):
    schema_version: Literal["1"] = "1"
    protocol: GDProtocol
    protocol_sha256: str
    protocol_commit: str
    protocol_path: str
    config_path: str
    selection_path: str
    g1_batch: str
    g1_decision_path: str
    controls: dict[str, str]
    readiness: ReadinessReport
    view_sha256: str
    key_sha256: str
    artifact_checksums: dict[str, str]
    guess_conditions: dict[str, Literal["D", "C"]] = Field(
        default_factory=lambda: {"condition 1": "D", "condition 2": "C"}
    )
    owner_code_work_at: datetime | None = None


def generate_gd(inputs: GDInputs, options: RunOptions) -> Path:
    checked = require_gd(inputs)
    protocol = checked.protocol
    if protocol is None or checked.protocol_sha256 is None:
        raise BatchError("registered G-D protocol missing")
    path = run_experiment(inputs.config, options)
    binding = _generation_binding(protocol, checked.protocol_sha256, path)
    write_text_lf(path / GD_GENERATION, binding.model_dump_json(indent=2) + "\n")
    return path


def _generation_binding(protocol: GDProtocol, checksum: str, path: Path) -> GenerationBinding:
    return GenerationBinding(
        protocol_sha256=checksum,
        config_sha256=protocol.config_sha256,
        g1_decision_sha256=protocol.g1_decision_sha256,
        selection_sha256=protocol.selection_sha256,
        test_only=protocol.test_only,
        manifest_sha256=file_checksum(path / "manifest.json"),
    )


def _generation(protocol: GDProtocol, checksum: str, path: Path) -> None:
    saved = GenerationBinding.model_validate_json(
        (path / GD_GENERATION).read_text(encoding="utf-8")
    )
    if saved != _generation_binding(protocol, checksum, path):
        raise BatchError("run is not bound to this registered G-D generation")
    if read_manifest(path).config_hash != protocol.config_sha256:
        raise BatchError("generated configuration differs from G-D registration")


def _artifact_checksums(sources: tuple[Source, Source]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for source in sources:
        for record in read_stories(source.run_dir):
            if record.condition != source.condition:
                continue
            for filename in record.files:
                path = (source.run_dir / record.path / filename).resolve()
                if not path.is_relative_to(source.run_dir.resolve()):
                    raise BatchError("story artifact escapes its run directory")
                artifacts[str(path)] = file_checksum(path)
        path = (source.run_dir / GD_GENERATION).resolve()
        artifacts[str(path)] = file_checksum(path)
    return artifacts


def control_evidence(
    protocol: GDProtocol, sources: tuple[Source, Source], controls: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Each fixed brief must have a report for the exact accepted D plan."""
    if set(controls) != set(protocol.cohort.brief_checksums):
        raise BatchError("controllability evidence must cover every registered brief")
    d = next(source for source in sources if source.condition == "D")
    stories = {
        record.brief: record
        for record in read_stories(d.run_dir)
        if record.condition == "D" and record.repetition == 0
    }
    reports: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for brief, path in sorted(controls.items()):
        report: dict[str, Any] = replay_report(path).model_dump(mode="json")
        if (
            report.get("scope") != "plan-only"
            or not report.get("lock_stable")
            or not report.get("authorised_scope")
            or not report.get("validator_passed")
        ):
            raise BatchError(f"{brief}: control lock, scope or validation checks failed")
        targets = report.get("targets", [])
        actual = {target["id"]: target["request_sha256"] for target in targets}
        if actual != protocol.control_targets or len(actual) != len(targets):
            raise BatchError(f"{brief}: requested controllability targets differ from registration")
        if any(target.get("status") == "unknown" for target in targets):
            raise BatchError(f"{brief}: controllability targets are pending")
        if brief not in stories:
            raise BatchError(f"{brief}: accepted D story missing")
        plan = d.run_dir / stories[brief].path / "plan.json"
        if report.get("after_sha256") != plan_identity(load_plan(plan)):
            raise BatchError(f"{brief}: controls concern a different accepted D plan")
        reports[brief] = report
        artifacts[str(path.resolve())] = file_checksum(path)
        for filename, checksum in report["inputs"].items():
            if file_checksum(Path(filename)) != checksum:
                raise BatchError(f"{brief}: controllability input missing or changed")
            artifacts[filename] = checksum
    return reports, artifacts


@dataclass(frozen=True)
class GDBatchSettings:
    batches_dir: Path
    name: str
    rubric: Rubric
    controls: dict[str, Path]
    seed: int | None = None
    owner_code_work_at: datetime | None = None


def build_gd_batch(
    inputs: GDInputs, sources: tuple[Source, Source], settings: GDBatchSettings
) -> Path:
    checked = require_gd(inputs)
    protocol = checked.protocol
    if protocol is None or checked.protocol_sha256 is None or checked.protocol_commit is None:
        raise BatchError("registered G-D protocol identity missing")
    if tuple(source.condition for source in sources) != ("D", "C"):
        raise BatchError("give the D candidate first and C comparator second")
    if settings.rubric.version != protocol.rubric_version:
        raise BatchError("reader rubric differs from G-D registration")
    if settings.owner_code_work_at is not None and settings.owner_code_work_at.utcoffset() is None:
        raise BatchError("owner code-work timestamp must include timezone")
    loaded = load_config(inputs.config)
    if loaded.briefs_dir is None:
        raise BatchError("registered benchmark missing")
    for source in sources:
        _generation(protocol, checked.protocol_sha256, source.run_dir)
    readiness = check_readiness(protocol.cohort, loaded.briefs_dir, sources)
    _, control_artifacts = control_evidence(protocol, sources, settings.controls)
    artifacts = _artifact_checksums(sources) | control_artifacts
    artifacts[str(inputs.config.resolve())] = file_checksum(inputs.config)
    scoring = Scoring(
        settings.rubric,
        optional=protocol.reading_mode == "required_only",
        primary="coherence",
        required_dimensions=protocol.required_dimensions,
    )
    path = build_gate_batch(
        protocol.cohort,
        loaded.briefs_dir,
        sources,
        GateBatchSettings(settings.batches_dir, settings.name, scoring, settings.seed),
    )
    # Direction and all planner metadata remain sealed; guess labels are opaque.
    view = read_view(path)
    view.guess_options = ["condition 1", "condition 2"]
    write_text_lf(path / "view.json", view.model_dump_json(indent=2) + "\n")
    binding = GDBinding(
        protocol=protocol,
        protocol_sha256=checked.protocol_sha256,
        protocol_commit=checked.protocol_commit,
        protocol_path=str(inputs.protocol.resolve()),
        config_path=str(inputs.config.resolve()),
        selection_path=str(inputs.selection.resolve()),
        g1_batch=str(inputs.g1_batch.resolve()),
        g1_decision_path=str(inputs.g1_decision.resolve()),
        controls={brief: str(p.resolve()) for brief, p in settings.controls.items()},
        readiness=readiness,
        view_sha256=file_checksum(path / "view.json"),
        key_sha256=file_checksum(path / "sealed/key.json"),
        artifact_checksums=artifacts,
        owner_code_work_at=settings.owner_code_work_at,
    )
    write_text_lf(path / "sealed" / GD_BINDING, binding.model_dump_json(indent=2) + "\n")
    return path


def read_gd_binding(batch: Path) -> GDBinding:
    if not is_closed(batch):
        raise BatchError("G-D batch is still open")
    return GDBinding.model_validate_json(
        (batch / "sealed" / GD_BINDING).read_text(encoding="utf-8")
    )


def binding_problems(batch: Path, binding: GDBinding) -> list[str]:
    expected = {
        binding.protocol_path: binding.protocol_sha256,
        binding.selection_path: binding.protocol.selection_sha256,
        binding.g1_decision_path: binding.protocol.g1_decision_sha256,
        str(batch / "view.json"): binding.view_sha256,
        str(batch / "sealed/key.json"): binding.key_sha256,
        **binding.artifact_checksums,
    }
    for root, files in binding.readiness.source_checksums.items():
        expected.update({str(Path(root) / name): checksum for name, checksum in files.items()})
    problems: list[str] = []
    if binding.guess_conditions != {"condition 1": "D", "condition 2": "C"}:
        problems.append("sealed condition-guess mapping changed")
    for filename, checksum in expected.items():
        try:
            if file_checksum(Path(filename)) != checksum:
                problems.append(f"bound artifact changed: {filename}")
        except OSError:
            problems.append(f"bound artifact missing: {filename}")
    try:
        saved = GDProtocol.model_validate_json(
            Path(binding.protocol_path).read_text(encoding="utf-8")
        )
        if saved != binding.protocol:
            problems.append("embedded G-D protocol differs from registered artifact")
        readiness = ReadinessReport.model_validate_json(
            (batch / "sealed/readiness/readiness.json").read_text(encoding="utf-8")
        )
        if readiness != binding.readiness:
            problems.append("embedded readiness differs from saved admission")
        inputs = GDInputs(
            Path(binding.protocol_path),
            Path(binding.config_path),
            Path(binding.selection_path),
            Path(binding.g1_batch),
            Path(binding.g1_decision_path),
        )
        require_gd(inputs)
        key = read_key(batch)
        if tuple(source.condition for source in key.sources) != ("D", "C"):
            raise BatchError("candidate direction differs from registration")
        first, second = (Source(Path(source.run), source.condition) for source in key.sources)
        _, controls = control_evidence(
            binding.protocol,
            (first, second),
            {brief: Path(path) for brief, path in binding.controls.items()},
        )
        if any(
            binding.artifact_checksums.get(path) != checksum for path, checksum in controls.items()
        ):
            problems.append("controllability evidence is not bound to the admitted batch")
    except (OSError, ValueError, BatchError) as error:
        problems.append(f"bound prerequisite evidence invalid: {error}")
    if (
        binding.readiness.status != "ready"
        or binding.readiness.cohort != binding.protocol.cohort
        or binding.readiness.cohort_sha256 != cohort_checksum(binding.protocol.cohort)
    ):
        problems.append("readiness does not admit the exact registered cohort")
    return problems
