"""Saved evidence inputs and explicit gate outcomes."""

from typing import Any, Literal

from pydantic import Field

from nde.corpus.provenance import HumanStory
from nde.narrative.base import Strict
from nde.narrative.diagnostics import Diagnostic

type Status = Literal["passed", "failed", "pending"]


class Mutation(Strict):
    category: str
    family: str
    description: str
    control: str
    path: list[str | int] = Field(min_length=1)
    value: Any
    expected_code: str
    expected_severity: str = "error"


class Suite(Strict):
    version: Literal["1"] = "1"
    controls: dict[str, dict[str, Any]]
    mutations: list[Mutation]


class AnnotationInput(Strict):
    session: str | None = None
    files: dict[str, str] = Field(default_factory=dict[str, str])
    timer: str = ""
    measured_seconds: float | None = None
    capture_errors: list[str] = Field(default_factory=list[str])


class StoryInput(Strict):
    provenance: HumanStory
    source_base64: str | None = None
    source_error: str | None = None
    annotation: AnnotationInput = Field(default_factory=AnnotationInput)


class Snapshot(Strict):
    version: Literal["1"] = "1"
    schema_text: str
    git_sha: str
    manifest_sha256: str
    suite: Suite
    stories: list[StoryInput]


class MutationResult(Strict):
    category: str
    family: str
    description: str
    expected_code: str
    expected_severity: str
    caught: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list[Diagnostic])
    parse_error: str | None = None


class SuiteResult(Strict):
    suite_sha256: str
    total: int
    caught: int
    catch_rate: float
    control_results: dict[str, bool]
    control_diagnostics: dict[str, list[Diagnostic]]
    results: list[MutationResult]
    problems: list[str]


class StoryEvidence(Strict):
    id: str
    title: str
    provenance: dict[str, Any]
    source_sha256: str | None = None
    text_sha256: str | None = None
    word_count: int | None = None
    source_status: str = "pending"
    annotation_status: str = "missing"
    generation_cost_usd: float | None = None
    draft_attempts: int | None = None
    identifiers: dict[str, str] = Field(default_factory=dict[str, str])
    revision: int | None = None
    correction_seconds: float | None = None
    owner_review: str = "pending"
    literary_evidence: str | None = None
    owner: str | None = None
    owner_confirmed_trajectories: bool | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list[Diagnostic])
    trajectories: list[str] = Field(default_factory=list[str])
    trajectory_signature: list[str] = Field(default_factory=list[str])
    pending: list[str] = Field(default_factory=list[str])
    failures: list[str] = Field(default_factory=list[str])
    mechanical_failures: list[str] = Field(default_factory=list[str])


class Comparison(Strict):
    story_a: str
    story_b: str
    distinct: bool | None


class Report(Strict):
    version: Literal["1"] = "1"
    report_id: str
    schema_sha256: str
    git_sha: str
    manifest_sha256: str
    mechanical_status: Status
    gate_status: Status
    mutations: SuiteResult
    stories: list[StoryEvidence]
    comparisons: list[Comparison]
    pending: list[str]
    failures: list[str]
