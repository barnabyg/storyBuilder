"""Pre-registered G1 generation and sealed admission; never inferred from outcomes."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlparse

from pydantic import Field, model_validator

from nde import briefs
from nde.eval.batch import BatchError, Scoring, Source, is_closed
from nde.eval.readiness import (
    GateBatchSettings,
    GateCohort,
    ReadinessReport,
    Record,
    build_gate_batch,
    check_readiness,
    cohort_checksum,
)
from nde.eval.rubric import Rubric
from nde.evidence.records import Report as SchemaReport
from nde.evidence.records import Snapshot
from nde.evidence.report import evaluate
from nde.files import checksum_lf, write_text_lf
from nde.gitinfo import git
from nde.runs.config import ConfigError, LoadedConfig, load_config
from nde.runs.records import read_manifest, read_stories
from nde.runs.runner import RunOptions, run_experiment
from nde.tiers import load_selection, selection_problems

G1_BINDING = "g1.json"
GENERATION_BINDING = "g1-generation.json"
G1_SAMPLE_SIZE = 30
SHA256_LENGTH = 64


class OriginalitySafeguard(Record):
    """Require mean paired C-minus-B originality >= -margin; no inferential claim."""

    analysis: Literal["mean_paired_difference"]
    margin: float = Field(ge=0, le=4, allow_inf_nan=False)


class GenerationLimits(Record):
    max_repairs: int = Field(ge=0)
    max_length_attempts: int = Field(ge=1)
    max_revision_passes: int = Field(ge=0)
    per_run_usd: float = Field(gt=0, allow_inf_nan=False)
    per_day_usd: float = Field(gt=0, allow_inf_nan=False)


class G1Protocol(Record):
    """Commit this complete protocol before making any registered provider call."""

    schema_version: Literal["1"] = "1"
    id: str = Field(min_length=1)
    test_only: bool = False
    cohort: GateCohort
    candidate: Literal["C"] = "C"
    comparator: Literal["B"] = "B"
    tier: Literal["S", "M", "L"]
    primary: Literal["coherence"] = "coherence"
    ties: Literal["non_wins"] = "non_wins"
    win_threshold: int = Field(default=21, ge=1)
    required_dimensions: tuple[Literal["coherence"], Literal["originality"]] = (
        "coherence",
        "originality",
    )
    originality: OriginalitySafeguard
    reading_mode: Literal["full", "required_only"]
    rubric_version: Literal[1] = 1
    config_sha256: str
    selection_sha256: str
    schema_evidence_sha256: str
    generation_limits: GenerationLimits
    stopping_rule: Literal["fixed_sample_no_top_up"] = "fixed_sample_no_top_up"
    fail_action: Literal["stop_or_one_fresh_preregistered_followup"] = (
        "stop_or_one_fresh_preregistered_followup"
    )

    @model_validator(mode="after")
    def _registered_design(self) -> Self:
        if self.cohort.conditions != ("C", "B") or self.cohort.repeat != 1:
            raise ValueError("G1 requires C first, B second, once per distinct brief")
        sample = len(self.cohort.brief_checksums)
        if not self.test_only and sample != G1_SAMPLE_SIZE:
            raise ValueError("real G1 requires exactly 30 distinct frozen test briefs")
        if self.win_threshold > sample:
            raise ValueError("win threshold exceeds the registered sample")
        for value in (self.config_sha256, self.selection_sha256, self.schema_evidence_sha256):
            if len(value) != SHA256_LENGTH or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("protocol identities must be lowercase SHA-256 values")
        return self


@dataclass(frozen=True)
class G1Inputs:
    protocol: Path
    config: Path
    selection: Path
    schema_evidence: Path


class G1Check(Record):
    status: Literal["ready", "blocked"]
    test_only: bool = False
    protocol: G1Protocol | None = None
    protocol_sha256: str | None = None
    protocol_commit: str | None = None
    problems: list[str] = Field(default_factory=list[str])


class GenerationBinding(Record):
    schema_version: Literal["1"] = "1"
    protocol_sha256: str
    config_sha256: str
    selection_sha256: str
    schema_evidence_sha256: str
    test_only: bool
    manifest_sha256: str


class G1Binding(Record):
    schema_version: Literal["1"] = "1"
    protocol: G1Protocol
    protocol_sha256: str
    protocol_path: str
    protocol_commit: str
    selection_path: str
    schema_evidence_path: str
    readiness: ReadinessReport
    view_sha256: str
    key_sha256: str
    artifact_checksums: dict[str, str]
    owner_code_work_at: datetime | None = None

    @property
    def test_only(self) -> bool:
        return self.protocol.test_only


def file_checksum(path: Path) -> str:
    return checksum_lf(path.read_bytes())


def _protocol_commit(path: Path) -> str:
    root_name = git(["rev-parse", "--show-toplevel"], path.parent)
    if root_name is None:
        raise ValueError("pre-registration must belong to a Git repository")
    root = Path(root_name).resolve()
    relative = path.resolve().relative_to(root).as_posix()
    if git(["ls-files", "--error-unmatch", "--", relative], root) is None:
        raise ValueError("pre-registration is uncommitted")
    if git(["diff", "--quiet", "HEAD", "--", relative], root) is None:
        raise ValueError("pre-registration differs from its committed contents")
    commit = git(["rev-parse", "HEAD"], root)
    if commit is None:
        raise ValueError("pre-registration has no committed revision")
    return commit


def _schema_evidence(protocol: G1Protocol, directory: Path) -> None:
    path = directory / "evidence.json"
    if file_checksum(path) != protocol.schema_evidence_sha256:
        raise ValueError("G-schema evidence checksum differs from registration")
    saved = SchemaReport.model_validate_json(path.read_text(encoding="utf-8"))
    snapshot = Snapshot.model_validate_json((directory / "inputs.json").read_text(encoding="utf-8"))
    replayed = evaluate(snapshot)
    if saved != replayed:
        raise ValueError("G-schema packet does not reproduce from its saved inputs")
    if saved.gate_status != "passed" or saved.mechanical_status != "passed":
        raise ValueError("G-schema owner gate has not passed")
    if not protocol.test_only:
        owner_text = " ".join(
            f"{row.owner or ''} {row.literary_evidence or ''} {row.title}" for row in saved.stories
        ).lower()
        if any(marker in owner_text for marker in ("test-only", "test only", "synthetic")):
            raise ValueError("test-only G-schema reviews cannot admit a real G1 run")


def _selection(protocol: G1Protocol, inputs: G1Inputs) -> None:
    if file_checksum(inputs.selection) != protocol.selection_sha256:
        raise ValueError("tier selection checksum differs from registration")
    selection = load_selection(inputs.selection)
    problems = selection_problems(selection, inputs.config)
    if problems:
        raise ValueError("tier selection is stale or incomplete: " + "; ".join(problems))
    if selection.tier != protocol.tier or selection.test_only != protocol.test_only:
        raise ValueError("selected tier or test-only status differs from registration")


def _configuration(protocol: G1Protocol, loaded: LoadedConfig) -> None:
    config = loaded.config
    stories = config.stories
    if loaded.hash != protocol.config_sha256:
        raise ValueError("experiment configuration differs from registration")
    if config.calls or config.judges is not None:
        raise ValueError("G1 generation cannot add arbitrary calls or judge tuning inputs")
    if stories is None or loaded.briefs_dir is None or stories.equal_budget is None:
        raise ValueError("G1 requires a story configuration with equal budgets")
    conditions = {condition.name: condition for condition in stories.conditions}
    if set(conditions) != {"B", "C"}:
        raise ValueError("G1 configuration requires exactly conditions C and B")
    if conditions["B"].pipeline != "baseline_b" or conditions["C"].pipeline != "baseline_c":
        raise ValueError("G1 requires the shared-prose Baseline B and C pipelines")
    if conditions["C"].params.get("tier", "L") != protocol.tier:
        raise ValueError("C configuration tier differs from registration")
    limits = protocol.generation_limits
    actual = GenerationLimits(
        max_repairs=config.structured.max_repairs,
        max_length_attempts=stories.length.max_attempts,
        max_revision_passes=stories.equal_budget.max_revision_passes,
        per_run_usd=config.budget.per_run_usd or 0,
        per_day_usd=config.budget.per_day_usd or 0,
    )
    if actual != limits:
        raise ValueError("generation, repair or hard budget limits differ from registration")
    if stories.repeat != 1 or stories.length.tolerance != protocol.cohort.length_tolerance:
        raise ValueError("length or repetition settings differ from registration")
    if (
        set(stories.equal_budget.conditions) != {"B", "C"}
        or stories.equal_budget.tolerance != protocol.cohort.budget_tolerance
    ):
        raise ValueError("equal-budget settings differ from registration")
    if config.budget.allow_unpriced_models:
        raise ValueError("registered G1 cannot bypass hard caps with unpriced models")
    _frozen_configuration(protocol, loaded)


def _frozen_configuration(protocol: G1Protocol, loaded: LoadedConfig) -> None:
    stories, root = loaded.config.stories, loaded.briefs_dir
    if stories is None or root is None:
        raise ValueError("registered story benchmark is missing")
    if not set(protocol.cohort.split_checksums).issubset({"dev", "test"}):
        raise ValueError("registered frozen split is unsupported")
    available = briefs.load_briefs(root)
    chosen = stories.briefs.ids or [
        brief.id for brief in available.values() if brief.split == stories.briefs.split
    ]
    if set(chosen) != set(protocol.cohort.brief_checksums) or len(chosen) != len(set(chosen)):
        raise ValueError("configuration does not generate the exact registered cohort")
    check = briefs.check(root)
    found = {file.brief.id: file for files in check.briefs.values() for file in files}
    for brief_id, checksum in protocol.cohort.brief_checksums.items():
        if brief_id not in found or found[brief_id].sha256 != checksum:
            raise ValueError(f"{brief_id}: frozen input differs from registration")
    split = "test" if not protocol.test_only else next(iter(protocol.cohort.split_checksums))
    if protocol.cohort.split_checksums != {
        split: briefs.split_checksum(root, "test" if split == "test" else "dev")
    }:
        raise ValueError("frozen benchmark identity differs from registration")
    if (
        not protocol.test_only
        and briefs.read_frozen(root).get("test") != protocol.cohort.brief_checksums
    ):
        raise ValueError("real G1 must use the complete frozen test cohort")
    if protocol.test_only and any(
        urlparse(provider.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}
        for provider in loaded.config.providers.values()
    ):
        raise ValueError("test-only G1 generation is restricted to local fake providers")


def check_g1(inputs: G1Inputs) -> G1Check:
    """No provider calls; missing scientific evidence is a blocking prerequisite."""
    result = G1Check(status="blocked")
    try:
        protocol = G1Protocol.model_validate_json(inputs.protocol.read_text(encoding="utf-8"))
        result.protocol, result.test_only = protocol, protocol.test_only
        result.protocol_sha256 = file_checksum(inputs.protocol)
        result.protocol_commit = _protocol_commit(inputs.protocol)
        loaded = load_config(inputs.config)
        _configuration(protocol, loaded)
        _schema_evidence(protocol, inputs.schema_evidence)
        _selection(protocol, inputs)
    except (OSError, ValueError, ConfigError, briefs.BriefError) as error:
        result.problems.append(str(error))
    else:
        result.status = "ready"
    return result


def require_g1(inputs: G1Inputs) -> G1Check:
    result = check_g1(inputs)
    if result.status != "ready":
        raise BatchError("G1 prerequisites blocked: " + "; ".join(result.problems))
    return result


def generate_g1(inputs: G1Inputs, options: RunOptions) -> Path:
    """Validate committed scientific prerequisites before any provider call."""
    checked = require_g1(inputs)
    protocol = checked.protocol
    if protocol is None or checked.protocol_sha256 is None:
        raise BatchError("validated protocol is missing")
    path = run_experiment(inputs.config, options)
    binding = GenerationBinding(
        protocol_sha256=checked.protocol_sha256,
        config_sha256=protocol.config_sha256,
        selection_sha256=protocol.selection_sha256,
        schema_evidence_sha256=protocol.schema_evidence_sha256,
        test_only=protocol.test_only,
        manifest_sha256=file_checksum(path / "manifest.json"),
    )
    write_text_lf(path / GENERATION_BINDING, binding.model_dump_json(indent=2) + "\n")
    return path


def _generation(protocol: G1Protocol, checksum: str, run: Path) -> None:
    saved = GenerationBinding.model_validate_json(
        (run / GENERATION_BINDING).read_text(encoding="utf-8")
    )
    expected = GenerationBinding(
        protocol_sha256=checksum,
        config_sha256=protocol.config_sha256,
        selection_sha256=protocol.selection_sha256,
        schema_evidence_sha256=protocol.schema_evidence_sha256,
        test_only=protocol.test_only,
        manifest_sha256=file_checksum(run / "manifest.json"),
    )
    if saved != expected or read_manifest(run).config_hash != protocol.config_sha256:
        raise BatchError("run is not bound to this registered G1 generation")


def _artifacts(sources: tuple[Source, Source]) -> dict[str, str]:
    found: dict[str, str] = {}
    for source in sources:
        for record in read_stories(source.run_dir):
            if record.condition != source.condition:
                continue
            for filename in record.files:
                path = (source.run_dir / record.path / filename).resolve()
                if not path.is_relative_to(source.run_dir.resolve()):
                    raise BatchError("story artifact escapes run directory")
                found[str(path)] = file_checksum(path)
        path = source.run_dir / GENERATION_BINDING
        found[str(path.resolve())] = file_checksum(path)
    return found


@dataclass(frozen=True)
class G1BatchSettings:
    batches_dir: Path
    name: str
    rubric: Rubric
    seed: int | None = None
    owner_code_work_at: datetime | None = None


def build_g1_batch(
    inputs: G1Inputs, sources: tuple[Source, Source], settings: G1BatchSettings
) -> Path:
    checked = require_g1(inputs)
    protocol = checked.protocol
    if protocol is None or checked.protocol_sha256 is None or checked.protocol_commit is None:
        raise BatchError("validated protocol identity is missing")
    loaded = load_config(inputs.config)
    if loaded.briefs_dir is None:
        raise BatchError("registered benchmark is missing")
    if settings.rubric.version != protocol.rubric_version:
        raise BatchError("reader rubric differs from registration")
    if settings.owner_code_work_at is not None and settings.owner_code_work_at.utcoffset() is None:
        raise BatchError("owner code-work timestamp must include its timezone")
    for source in sources:
        _generation(protocol, checked.protocol_sha256, source.run_dir)
    readiness = check_readiness(protocol.cohort, loaded.briefs_dir, sources)
    artifacts = _artifacts(sources)
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
    binding = G1Binding(
        protocol=protocol,
        protocol_sha256=checked.protocol_sha256,
        protocol_path=str(inputs.protocol.resolve()),
        protocol_commit=checked.protocol_commit,
        selection_path=str(inputs.selection.resolve()),
        schema_evidence_path=str(inputs.schema_evidence.resolve()),
        readiness=readiness,
        view_sha256=file_checksum(path / "view.json"),
        key_sha256=file_checksum(path / "sealed/key.json"),
        artifact_checksums=artifacts,
        owner_code_work_at=settings.owner_code_work_at,
    )
    write_text_lf(path / "sealed" / G1_BINDING, binding.model_dump_json(indent=2) + "\n")
    return path


def read_g1_binding(batch: Path) -> G1Binding:
    if not is_closed(batch):
        raise BatchError("G1 batch is still open")
    return G1Binding.model_validate_json(
        (batch / "sealed" / G1_BINDING).read_text(encoding="utf-8")
    )


def binding_problems(batch: Path, binding: G1Binding) -> list[str]:
    """Validate bound artifacts for a closed result; do not silently reinterpret changes."""
    expected = {
        str(batch / "view.json"): binding.view_sha256,
        str(batch / "sealed/key.json"): binding.key_sha256,
        binding.protocol_path: binding.protocol_sha256,
        binding.selection_path: binding.protocol.selection_sha256,
        str(
            Path(binding.schema_evidence_path) / "evidence.json"
        ): binding.protocol.schema_evidence_sha256,
        **binding.artifact_checksums,
    }
    for root, files in binding.readiness.source_checksums.items():
        expected.update(
            {str(Path(root) / filename): checksum for filename, checksum in files.items()}
        )
    problems: list[str] = []
    for filename, checksum in expected.items():
        try:
            if file_checksum(Path(filename)) != checksum:
                problems.append(f"bound artifact changed: {filename}")
        except OSError:
            problems.append(f"bound artifact missing: {filename}")
    try:
        saved_protocol = G1Protocol.model_validate_json(
            Path(binding.protocol_path).read_text(encoding="utf-8")
        )
        if saved_protocol != binding.protocol:
            problems.append("embedded protocol differs from the committed protocol artifact")
    except (OSError, ValueError) as error:
        problems.append(f"bound protocol cannot be validated: {error}")
    try:
        saved_readiness = ReadinessReport.model_validate_json(
            (batch / "sealed/readiness/readiness.json").read_text(encoding="utf-8")
        )
        if saved_readiness != binding.readiness:
            problems.append("embedded readiness differs from saved admission evidence")
    except (OSError, ValueError) as error:
        problems.append(f"bound readiness cannot be validated: {error}")
    if binding.readiness.cohort_sha256 != cohort_checksum(binding.protocol.cohort):
        problems.append("bound readiness cohort checksum differs from registration")
    if binding.readiness.status != "ready" or binding.readiness.cohort != binding.protocol.cohort:
        problems.append("bound readiness does not admit the registered cohort")
    return problems


def write_check(result: G1Check, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "g1-check.json", result.model_dump_json(indent=2) + "\n")
    lines = [f"# G1 prerequisites: {result.status}", f"Test only: {result.test_only}"]
    lines.extend(f"- {problem}" for problem in result.problems)
    write_text_lf(out / "g1-check.md", "\n\n".join(lines) + "\n")
