"""No-spend prerequisites for the separately registered D-versus-C gate."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlparse

from pydantic import Field, model_validator

from nde import briefs, g1_report
from nde.annotations.records import digest
from nde.cd_comparison import selected_c_problems
from nde.eval import g1
from nde.eval.batch import BatchError, read_key
from nde.eval.readiness import GateCohort, Record
from nde.files import write_text_lf
from nde.gitinfo import git
from nde.narrative.workflow import schema_json
from nde.runs.config import ConfigError, LoadedConfig, load_config
from nde.runs.prompts import PromptTemplate
from nde.runs.records import read_manifest
from nde.stories.execute import prepare_stories
from nde.tiers import load_selection


class PairedComparison(Record):
    """Descriptive ordinal-score safeguard, not inferential noninferiority."""

    analysis: Literal["mean_paired_difference"]
    margin: float = Field(ge=0, le=4, allow_inf_nan=False)


class GDProtocol(Record):
    schema_version: Literal["1"] = "1"
    id: str = Field(min_length=1)
    test_only: bool = False
    candidate: Literal["D"] = "D"
    comparator: Literal["C"] = "C"
    cohort: GateCohort
    n: int = Field(ge=1)
    tier: Literal["S", "M", "L"]
    primary: Literal["coherence"] = "coherence"
    coherence: PairedComparison
    originality: PairedComparison
    ties: Literal["zero_paired_difference"] = "zero_paired_difference"
    required_dimensions: tuple[Literal["coherence"], Literal["originality"]] = (
        "coherence",
        "originality",
    )
    reading_mode: Literal["full", "required_only"]
    rubric_version: Literal[1] = 1
    g1_decision_sha256: str
    selection_sha256: str
    config_sha256: str
    schema_sha256: str
    models_sha256: str
    prompt_sha256: dict[str, str] = Field(min_length=1)
    control_targets: dict[str, str] = Field(min_length=1)
    generation_limits: g1.GenerationLimits
    critic_iterations: int = Field(ge=0)
    prior_benchmark_exposure: str = Field(min_length=1)
    stopping_rule: Literal["fixed_sample_no_top_up"] = "fixed_sample_no_top_up"
    fail_action: Literal["retain_C_reference"] = "retain_C_reference"

    @model_validator(mode="after")
    def _design(self) -> Self:
        if self.cohort.conditions != ("D", "C") or self.cohort.repeat != 1:
            raise ValueError("G-D requires D first, C second, once per distinct brief")
        if self.n != len(self.cohort.brief_checksums):
            raise ValueError("fixed n must equal the distinct registered brief count")
        checksums = [
            self.g1_decision_sha256,
            self.selection_sha256,
            self.config_sha256,
            self.schema_sha256,
            self.models_sha256,
            *self.prompt_sha256.values(),
            *self.control_targets.values(),
        ]
        if any(
            len(value) != g1.SHA256_LENGTH or any(c not in "0123456789abcdef" for c in value)
            for value in checksums
        ):
            raise ValueError("registered identities must be lowercase SHA-256 values")
        return self


@dataclass(frozen=True)
class GDInputs:
    protocol: Path
    config: Path
    selection: Path
    g1_batch: Path
    g1_decision: Path


class GDCheck(Record):
    status: Literal["ready", "blocked"]
    test_only: bool = False
    protocol: GDProtocol | None = None
    protocol_sha256: str | None = None
    protocol_commit: str | None = None
    problems: list[str] = Field(default_factory=list[str])


def committed_protocol(path: Path) -> str:
    root_name = git(["rev-parse", "--show-toplevel"], path.parent)
    if root_name is None:
        raise ValueError("G-D pre-registration must belong to a Git repository")
    root = Path(root_name).resolve()
    relative = path.resolve().relative_to(root).as_posix()
    if git(["ls-files", "--error-unmatch", "--", relative], root) is None:
        raise ValueError("G-D pre-registration is uncommitted")
    if git(["diff", "--quiet", "HEAD", "--", relative], root) is None:
        raise ValueError("G-D pre-registration differs from committed contents")
    revision = git(["rev-parse", "HEAD"], root)
    if revision is None:
        raise ValueError("G-D pre-registration has no committed revision")
    return revision


def g1_evidence(protocol: GDProtocol, inputs: GDInputs) -> dict[str, Any]:
    """Recompute the closed gate; a supplied passed JSON blob is insufficient."""
    if g1.file_checksum(inputs.g1_decision) != protocol.g1_decision_sha256:
        raise ValueError("G1 decision identity differs from registration")
    raw = json.loads(inputs.g1_decision.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("G1 decision must be a structured result record")
    saved = cast(dict[str, Any], raw)
    if saved.get("id") != digest(
        json.dumps({k: v for k, v in saved.items() if k != "id"}, sort_keys=True)
    ):
        raise ValueError("G1 decision content identity changed")
    replayed = g1_report.decision(inputs.g1_batch, [])
    for field in (
        "outcome",
        "criteria_outcome",
        "test_only",
        "protocol_sha256",
        "readiness",
        "coherence",
        "originality",
        "judgements_sha256",
    ):
        if saved.get(field) != replayed.get(field):
            raise ValueError(f"G1 decision does not reproduce: {field}")
    if replayed["criteria_outcome"] != "passed" or (replayed["pending"] and not protocol.test_only):
        raise ValueError("scientific G1 decision has not passed with consistent evidence")
    binding = g1.read_g1_binding(inputs.g1_batch)
    if not protocol.test_only:
        if binding.test_only or replayed["outcome"] != "passed":
            raise ValueError("synthetic or pending G1 evidence cannot authorise real Phase 4")
        key = read_key(inputs.g1_batch)
        source_config = Path(read_manifest(Path(key.sources[0].run)).config_path)
        g1.require_g1(
            g1.G1Inputs(
                Path(binding.protocol_path),
                source_config,
                Path(binding.selection_path),
                Path(binding.schema_evidence_path),
            )
        )
    if binding.protocol.selection_sha256 != protocol.selection_sha256:
        raise ValueError("C baseline differs from the baseline established by G1")
    return replayed


def _identities(protocol: GDProtocol, inputs: GDInputs, loaded: LoadedConfig) -> None:
    if loaded.hash != protocol.config_sha256:
        raise ValueError("G-D configuration differs from registration")
    if g1.file_checksum(inputs.selection) != protocol.selection_sha256:
        raise ValueError("selected C baseline checksum differs from registration")
    selected = load_selection(inputs.selection)
    if selected.tier != protocol.tier or selected.test_only != protocol.test_only:
        raise ValueError("selected C tier or test-only status differs from registration")
    if digest(schema_json()) != protocol.schema_sha256:
        raise ValueError("narrative schema differs from registration")
    models = {
        name: binding.model_dump(mode="json") for name, binding in loaded.config.roles.items()
    }
    if digest(json.dumps(models, sort_keys=True)) != protocol.models_sha256:
        raise ValueError("model identities differ from registration")
    templates: dict[str, PromptTemplate] = {}
    prepare_stories(loaded, templates)
    if {label: template.sha256 for label, template in templates.items()} != protocol.prompt_sha256:
        raise ValueError("prompt identities differ from registration")
    problems = selected_c_problems(selected, inputs.config)
    if problems:
        raise ValueError("selected C baseline is stale or incomplete: " + "; ".join(problems))


def _configuration(protocol: GDProtocol, loaded: LoadedConfig) -> None:
    config, stories = loaded.config, loaded.config.stories
    if stories is None or loaded.briefs_dir is None or stories.equal_budget is None:
        raise ValueError("G-D requires shared-prose C/D stories and equal-budget matching")
    if {item.name: item.pipeline for item in stories.conditions} != {
        "D": "baseline_d",
        "C": "baseline_c",
    }:
        raise ValueError("G-D requires exactly D baseline_d and C baseline_c")
    if config.calls or config.judges is not None or config.plans is not None:
        raise ValueError("registered generation cannot include other calls or judge tuning")
    limits = g1.GenerationLimits(
        max_repairs=config.structured.max_repairs,
        max_length_attempts=stories.length.max_attempts,
        max_revision_passes=stories.equal_budget.max_revision_passes,
        per_run_usd=config.budget.per_run_usd or 0,
        per_day_usd=config.budget.per_day_usd or 0,
    )
    if limits != protocol.generation_limits or config.budget.allow_unpriced_models:
        raise ValueError("generation or hard spending limits differ from registration")
    d = next(item for item in stories.conditions if item.name == "D")
    if d.params.get("critic_iterations", 0) != protocol.critic_iterations:
        raise ValueError("critic iteration limit differs from registration")
    if (
        stories.repeat != 1
        or stories.length.tolerance != protocol.cohort.length_tolerance
        or set(stories.equal_budget.conditions) != {"C", "D"}
        or stories.equal_budget.tolerance != protocol.cohort.budget_tolerance
    ):
        raise ValueError("repetition, length or budget matching settings differ from registration")
    _benchmark(protocol, loaded)


def _benchmark(protocol: GDProtocol, loaded: LoadedConfig) -> None:
    root, stories = loaded.briefs_dir, loaded.config.stories
    if root is None or stories is None:
        raise ValueError("registered benchmark missing")
    available = briefs.load_briefs(root)
    ids = stories.briefs.ids or [
        b.id for b in available.values() if b.split == stories.briefs.split
    ]
    if set(ids) != set(protocol.cohort.brief_checksums) or len(ids) != protocol.n:
        raise ValueError("configuration must generate exactly the fixed distinct cohort")
    if not protocol.test_only and set(protocol.cohort.split_checksums) != {"test"}:
        raise ValueError("real G-D requires a frozen test cohort")
    frozen = briefs.read_frozen(root)
    for brief_id, checksum in protocol.cohort.brief_checksums.items():
        if brief_id not in available:
            raise ValueError(f"registered brief missing: {brief_id}")
        split = available[brief_id].split
        path = root / split / f"{brief_id}.yaml"
        if frozen.get(split, {}).get(brief_id) != checksum or g1.file_checksum(path) != checksum:
            raise ValueError(f"registered brief missing, changed or not frozen: {brief_id}")
    for split, checksum in protocol.cohort.split_checksums.items():
        if (
            split not in {"dev", "test"}
            or briefs.split_checksum(root, "test" if split == "test" else "dev") != checksum
        ):
            raise ValueError("frozen split identity differs from registration")
    if protocol.test_only and any(
        urlparse(p.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}
        for p in loaded.config.providers.values()
    ):
        raise ValueError("test-only G-D is restricted to local fake providers")


def check_gd(inputs: GDInputs) -> GDCheck:
    result = GDCheck(status="blocked")
    try:
        protocol = GDProtocol.model_validate_json(inputs.protocol.read_text(encoding="utf-8"))
        result.protocol, result.test_only = protocol, protocol.test_only
        result.protocol_sha256 = g1.file_checksum(inputs.protocol)
        result.protocol_commit = committed_protocol(inputs.protocol)
        g1_evidence(protocol, inputs)
        loaded = load_config(inputs.config)
        _configuration(protocol, loaded)
        _identities(protocol, inputs, loaded)
    except (OSError, ValueError, ConfigError, BatchError, briefs.BriefError) as error:
        result.problems.append(str(error))
    else:
        result.status = "ready"
    return result


def require_gd(inputs: GDInputs) -> GDCheck:
    checked = check_gd(inputs)
    if checked.status != "ready":
        raise BatchError("G-D prerequisites blocked: " + "; ".join(checked.problems))
    return checked


def write_check(result: GDCheck, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / "gd-check.json", result.model_dump_json(indent=2) + "\n")
    lines = [
        f"# G-D prerequisites: {result.status}",
        f"Test only: {result.test_only}",
        *result.problems,
    ]
    write_text_lf(out / "gd-check.md", "\n\n".join(lines) + "\n")
