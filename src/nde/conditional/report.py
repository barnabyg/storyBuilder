"""Candidate E condition adherence, structural diversity and training-similarity evidence.

One versioned report binds E story runs and E revision runs to their briefs,
allowed constraints, checkpoint, generation/revision histories, final plans and
stories, training populations and the identified reference baseline. It keeps
three kinds of evidence apart:

- deterministic compliance: documented predicates over explicitly representable
  requests (scene ranges, target length on the final prose, locks, revision
  scope and plan validity), each success, failure or unknown, with complete
  expected denominators;
- model diagnostics: what the local model and its checks did, never adherence;
- literary judgement: genre, themes and natural-language endings stay pending
  until the owner assesses them against the current output. Model estimates and
  synthetic assessments are counted, never accepted.

Diversity, training similarity and reference similarity share one ID-independent
feature space: the checkpoint's retained codec view with identities replaced by
typed list positions and descriptive text removed. Reports never decide G-E,
never write anything a training loader accepts and never read held-out briefs.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import Field, JsonValue, ValidationError, model_validator

from nde import realisation
from nde.annotations.records import digest
from nde.briefs import BriefError
from nde.conditional.data import Conditions
from nde.conditional.inpainting import training_policy
from nde.conditional.training import inspect_checkpoint
from nde.corpus.records_io import create_record, file_sha, identity, read_sealed, snapshot
from nde.diversity import multiset_distance, typed_edges
from nde.files import checksum_lf, write_text_lf
from nde.gd_report import decision as gd_decision
from nde.llm.config import Strict
from nde.narrative.ledger import plan_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan
from nde.plan.alternatives import AlternativesError, check_report, structural_content
from nde.plan.backends import ERevisionBackend
from nde.plan.conditional import ConditionalPlanningParams, brief_conditions
from nde.plan.revision import (
    RevisionPlanningParams,
    RevisionSpec,
    boundary_errors,
    canonical,
    selector_value,
)
from nde.plan.revision import identity as plan_sha256
from nde.reconstruction.codec import CODEC_ID, encode_plan
from nde.reconstruction.overlap import check_index, load_view, neighbour_rows, view_distance
from nde.reconstruction.overlap import records as view_records
from nde.runs.config import ConfigError, ExperimentConfig
from nde.runs.records import Manifest, StoryRecord, read_manifest, read_stories
from nde.schema.brief import Brief
from nde.stories.execute import select_briefs
from nde.stories.normalise import word_count
from nde.stories.selected import selected_text
from nde.tiers import load_selection

KIND = "conditional-evidence-report"
# Version 2 separates operational no-output from non-adherence failure.
REPORT_VERSION = "2"
NO_OUTPUT = "no-output"
REPORT = "report.json"
STRUCTURE_METRIC = "retained-structure-multiset-jaccard-v1"
EDGE_METRIC = "ordinal-position-multiset-jaccard-v1"
JUDGEMENT_CONDITIONS = ("genre", "themes", "ending_intent")
# A unit whose saved evidence cannot be read: unknown, never failure.
UNREADABLE = "unreadable"
# A unit whose saved evidence contradicts its recorded inputs: also unknown.
UNVERIFIED = "unverified"
PREDICATES = {
    "scene_range": "success iff the selected plan's scene count lies in the requested "
    "inclusive scene_range condition",
    "target_words": "success iff the final selected prose's word count lies in the requested "
    "inclusive target_words range widened by the run's registered length tolerance; unknown "
    "for plan-only revisions, which have no prose",
    "lock": "success iff the E story lock's ID-keyed selector holds exactly the locked value "
    "in the selected plan; failure if the selector is absent",
    "revision_lock": "success iff the locked selector's canonical JSON is unchanged from the "
    "recorded source plan; failure if it no longer resolves",
    "revision_scope": "success iff the selected plan changes only units the revision mask "
    "authorises, with identities kept (the shared revision boundary)",
    "validity": "success iff the selected plan passes complete plan validation",
    "no-output": "a unit that failed operationally without a saved, verified plan (or, for "
    "target_words, without final prose) has no output to judge: it stays in the denominator "
    "and is neither success nor non-adherence failure",
    "missing": "an expected unit without any record is unknown, never success",
    "unreadable": "a unit whose saved provenance, plan or story cannot be read is unknown",
    "unverified": "a unit whose saved plan or source does not bind its recorded identities "
    "is unknown",
}
STRUCTURE = (
    "The checkpoint's retained codec view (target policy channels; identities and references "
    "as typed list positions) with descriptive and interpretive reading text set to null. "
    "Features are each channel's canonical records as a multiset; distance is one minus "
    "shared over combined counts across the channels both views retain. Unknown and "
    "uncertain readings stay in the features and mark the comparison uncertain. Renaming "
    "identities alone gives distance zero."
)
EDGES = (
    "Typed causal, setup/payoff, reveal and knowledge-use edges between ordinal scene, "
    "fact and character positions (the diversity-report edge metric); null for invalid plans."
)
SIMILARITY_LIMITATIONS = [
    "Structural similarity is a candidate match, not proof of memorisation or its absence.",
    "Only the listed known training populations were compared. Text outside the retained "
    "structural view (titles, summaries, descriptions, prose) is not compared.",
]
NOT_A_DECISION = (
    "This report is evidence for a registered G-E comparison. It is not a G-E decision, "
    "literary adherence, owner review or scientific advancement."
)

type Status = Literal["success", "failure", "unknown", "no-output"]
type Record = dict[str, Any]


class StoryRunRef(Strict):
    run: Path
    condition: str = Field(default="E", min_length=1)
    label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")


class RevisionRunRef(Strict):
    run: Path
    label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]+$")


class ReferenceRunRef(Strict):
    run: Path
    condition: Literal["C", "D"]


class ReferenceRequest(Strict):
    # A G-D decision folder; D is the reference only when it replays as an actual pass.
    gd_decision: Path | None = None
    runs: list[ReferenceRunRef] = Field(default_factory=list[ReferenceRunRef])
    # The owner's frozen C tier selection, identifying the C baseline.
    c_selection: Path | None = None


class PopulationRef(Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    path: Path
    # `g3` marks the reconstruction population behind the checkpoint's G3 evidence.
    role: Literal["g3", "other"] = "other"


class ReportRequest(Strict):
    version: Literal["1"] = "1"
    story_runs: list[StoryRunRef] = Field(default_factory=list[StoryRunRef])
    revision_runs: list[RevisionRunRef] = Field(default_factory=list[RevisionRunRef])
    reference: ReferenceRequest = Field(default_factory=ReferenceRequest)
    training_populations: list[PopulationRef] = Field(default_factory=list[PopulationRef])
    protected_populations: list[PopulationRef] = Field(default_factory=list[PopulationRef])
    realisation: list[Path] = Field(default_factory=list[Path])
    judgements: list[Path] = Field(default_factory=list[Path])
    # Used only when the checkpoint's training dataset is unavailable; recorded as declared.
    target_policy: dict[str, Literal["hard", "soft", "excluded"]] | None = None
    neighbours: int = Field(default=3, ge=1, le=50)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if not self.story_runs and not self.revision_runs:
            raise ValueError("report requires at least one E story or revision run")
        names = [p.name for p in [*self.training_populations, *self.protected_populations]]
        if len(names) != len(set(names)):
            raise ValueError("population names must be distinct")
        return self


class JudgementItem(Strict):
    unit: str = Field(min_length=1)
    # The final story (or plan-only revision) identity the assessment was made against.
    output_sha256: str = Field(min_length=64, max_length=64)
    condition: Literal["genre", "themes", "ending_intent"]
    rating: Literal["adheres", "does_not_adhere", "unknown"]
    note: str = Field(min_length=1)


class JudgementReview(Strict):
    reviewer: str = Field(min_length=1)
    test_only: bool = False
    # Model estimates identify their model and prompt; they never count as assessment.
    diagnostic_provenance: dict[str, str] | None = None
    assessments: list[JudgementItem] = Field(min_length=1)


@dataclass
class _Unit:
    """One expected E output: a story repetition or a revision alternative."""

    id: str
    kind: Literal["story", "revision"]
    label: str
    run: Path
    brief: Brief
    repetition: int
    status: str
    seed: int | None = None
    path: str | None = None
    plan: NarrativePlan | None = None
    before: NarrativePlan | None = None
    spec: RevisionSpec | None = None
    conditions: Record = field(default_factory=dict[str, Any])
    locks: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    record: StoryRecord | None = None
    prose: str | None = None
    tolerance: float = 0.0
    checkpoint_id: str | None = None
    checkpoint_path: str | None = None
    diagnostics: Record = field(default_factory=dict[str, Any])
    view: Record | None = None
    valid: bool = False

    def compare_with(self, policy: dict[str, str]) -> None:
        """Derive the retained structure and validity that every comparison uses."""
        if self.plan is not None:
            self.valid = validate_plan(self.plan).valid
            self.view = retained_structure(self.plan, policy)

    @property
    def plan_sha256(self) -> str | None:
        return None if self.plan is None else plan_sha256(self.plan)

    @property
    def output_sha256(self) -> str | None:
        """Final selected prose for stories; the selected plan for plan-only revisions."""
        if self.kind == "story":
            return None if self.prose is None else digest(self.prose)
        return self.plan_sha256


@dataclass
class _Context:
    base: Path
    request: ReportRequest
    files: set[Path] = field(default_factory=set[Path])
    problems: list[str] = field(default_factory=list[str])

    def resolve(self, path: Path) -> Path:
        return (self.base / path).resolve()

    def bind(self, *paths: Path) -> None:
        self.files.update(path for path in paths if path.is_file())


@dataclass(frozen=True)
class _Run:
    path: Path
    label: str
    manifest: Manifest
    config: ExperimentConfig
    config_dir: Path


def _read(path: Path) -> Record:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(Record, value)


def _run(ctx: _Context, path: Path, label: str | None) -> _Run:
    folder = ctx.resolve(path)
    manifest = read_manifest(folder)
    saved = folder / "config.json"
    try:
        config = ExperimentConfig.model_validate_json(saved.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"run {folder} has no saved configuration: {error}") from error
    ctx.bind(folder / "manifest.json", saved)
    return _Run(folder, label or folder.name, manifest, config, Path(manifest.config_path).parent)


def _briefs(ctx: _Context, run: _Run, selection: Any, briefs_dir: str) -> list[Brief]:
    folder = (run.config_dir / briefs_dir).resolve()
    try:
        briefs = select_briefs(selection, folder)
    except (ConfigError, BriefError) as error:
        raise ValueError(f"run {run.path}: briefs unavailable: {error}") from error
    if any(brief.split != "dev" for brief in briefs):
        # Held-out briefs are protected; their outputs are never development evidence.
        raise ValueError(f"run {run.path} uses held-out briefs; only dev runs can be reported")
    ctx.bind(*(folder / brief.split / f"{brief.id}.yaml" for brief in briefs))
    return briefs


def _brief_sha256(brief: Brief) -> str:
    return hashlib.sha256(brief.model_dump_json().encode()).hexdigest()


def _object(value: Any) -> Record:
    return cast(Record, value) if isinstance(value, dict) else {}


def _objects(value: Any) -> list[Record]:
    rows = cast(list[Any], value) if isinstance(value, list) else []
    return [cast(Record, row) for row in rows if isinstance(row, dict)]


def _attempts(value: Any) -> list[Record]:
    return [
        {
            "attempt": row.get("attempt"),
            "status": row.get("status"),
            "diagnostics": row.get("diagnostics", []),
        }
        for row in _objects(value)
    ]


def _story_diagnostics(provenance: Record) -> Record:
    generation = _object(provenance.get("generation"))
    work = _objects(provenance.get("local_work"))
    return {
        "generation_status": generation.get("status"),
        "attempts": _attempts(generation.get("attempts")),
        "guidance_scale": _object(provenance.get("params")).get("guidance_scale"),
        "local_work": {
            name: sum(row.get(name, 0) for row in work)
            for name in ("passes", "tokens", "attention_cells")
        },
        "completion_calls": len(_objects(provenance.get("calls"))),
        "scientific_status": provenance.get("scientific_status"),
    }


def _story_unit(  # noqa: PLR0913 - one unit's run, request, brief, slot and saved record
    ctx: _Context,
    run: _Run,
    params: ConditionalPlanningParams,
    *,
    brief: Brief,
    repetition: int,
    rec: StoryRecord | None,
) -> _Unit:
    stories = run.config.stories
    unit = _Unit(
        id=f"story/{run.label}/{brief.id}/r{repetition + 1}",
        kind="story",
        label=run.label,
        run=run.path,
        brief=brief,
        repetition=repetition,
        status="missing" if rec is None else rec.status,
        # The registered request, known even for a story that left no record.
        conditions=brief_conditions(brief, params.condition_fields).model_dump(mode="json"),
        locks=dict(params.locks),
        tolerance=0.0 if stories is None else stories.length.tolerance,
        checkpoint_path=str((run.config_dir / params.checkpoint).resolve()),
    )
    if rec is None:
        return unit
    unit.seed, unit.path, unit.record = rec.seed, rec.path, rec
    folder = run.path / rec.path
    try:
        provenance = _read(folder / "planning-provenance.json")
    except (OSError, ValueError) as error:
        ctx.problems.append(f"{unit.id}: planning provenance unavailable: {error}")
        unit.status = UNREADABLE
        return unit
    ctx.bind(folder / "planning-provenance.json")
    if provenance.get("brief_sha256") != _brief_sha256(brief):
        ctx.problems.append(f"{unit.id}: brief differs from the one the plan was made from")
    recorded = provenance.get("conditions")
    if _object(provenance.get("params")).get("locks", {}) != unit.locks or (
        recorded is not None and recorded != unit.conditions
    ):
        ctx.problems.append(f"{unit.id}: recorded conditions or locks differ from the request")
    unit.checkpoint_id = _object(provenance.get("checkpoint")).get("id")
    unit.diagnostics = _story_diagnostics(provenance)
    _story_output(ctx, unit, provenance, run.path, rec)
    return unit


def _story_output(
    ctx: _Context, unit: _Unit, provenance: Record, run: Path, rec: StoryRecord
) -> None:
    """Attach the story's verified plan and, when it succeeded, its final prose."""
    folder = run / rec.path
    plan_file, selection_file = folder / "plan.json", folder / "conditional-selection.json"
    if rec.status != "ok" and not plan_file.is_file():
        # Planning itself failed: there is no output, which is not non-adherence.
        return
    ctx.bind(plan_file, selection_file)
    try:
        plan = load_plan(plan_file)
        selection = _read(selection_file)
    except (OSError, ValueError, PlanError) as error:
        ctx.problems.append(f"{unit.id}: selected plan unavailable: {error}")
        unit.status = UNREADABLE
        return
    if selection.get("plan_sha256") != plan_identity(plan) or provenance.get(
        "plan_sha256"
    ) != checksum_lf(plan_file.read_bytes()):
        ctx.problems.append(f"{unit.id}: selected plan differs from its E selection/provenance")
        unit.status = UNVERIFIED
        return
    # A story that failed after planning keeps its verified plan; only prose is absent.
    unit.plan = plan
    if rec.status != "ok":
        return
    ctx.bind(folder / "story.md", folder / "story.raw.md")
    try:
        unit.prose = selected_text(run, rec)
    except (OSError, ValueError) as error:
        ctx.problems.append(f"{unit.id}: final story unavailable: {error}")
        unit.status, unit.plan = UNREADABLE, None
        return
    if rec.sha256 is not None and rec.sha256 != digest(unit.prose):
        ctx.problems.append(f"{unit.id}: final story differs from its recorded selection")


def _story_units(ctx: _Context, ref: StoryRunRef) -> tuple[_Run, list[_Unit]]:
    run = _run(ctx, ref.run, ref.label)
    stories = run.config.stories
    if stories is None:
        raise ValueError(f"run {run.path} is not a story run")
    spec = next((c for c in stories.conditions if c.name == ref.condition), None)
    if spec is None or spec.pipeline != "candidate_e":
        raise ValueError(f"run {run.path}: condition {ref.condition!r} is not a candidate_e run")
    try:
        params = ConditionalPlanningParams.model_validate(spec.params)
    except ValidationError as error:
        raise ValueError(f"run {run.path}: invalid E parameters: {error}") from error
    briefs = _briefs(ctx, run, stories.briefs, stories.briefs_dir)
    ctx.bind(*(run.path / name for name in ("stories.jsonl", "local-work.jsonl")))
    ctx.bind(run.path / "comparison.json")
    found: dict[tuple[str, int], StoryRecord] = {}
    for rec in read_stories(run.path):
        if rec.condition != ref.condition:
            continue
        if (rec.brief, rec.repetition) in found:
            ctx.problems.append(f"{run.label}: duplicate story record {rec.brief}/{rec.repetition}")
        found[rec.brief, rec.repetition] = rec
    units = [
        _story_unit(
            ctx,
            run,
            params,
            brief=brief,
            repetition=repetition,
            rec=found.get((brief.id, repetition)),
        )
        for brief in briefs
        for repetition in range(stories.repeat)
    ]
    return run, units


def _revision_unit(  # noqa: PLR0913 - one alternative's run, request, brief, slot and record
    ctx: _Context,
    run: _Run,
    params: RevisionPlanningParams,
    *,
    brief: Brief,
    repetition: int,
    record: Record | None,
    source_sha256: str,
) -> _Unit:
    backend = cast(ERevisionBackend, params.backend)
    unit = _Unit(
        id=f"revision/{run.label}/{brief.id}/r{repetition + 1}",
        kind="revision",
        label=run.label,
        run=run.path,
        brief=brief,
        repetition=repetition,
        status="missing",
        spec=params.revision,
        # The registered request, known even for an alternative that left no record.
        conditions=Conditions.model_validate(backend.conditions).model_dump(mode="json"),
        checkpoint_path=str((run.config_dir / backend.checkpoint).resolve()),
    )
    if record is None:
        return unit
    folder = run.path / str(record["path"])
    unit.path = str(record["path"])
    revision = folder / "revisions/revision"
    try:
        provenance = _read(revision / "provenance.json")
        recorded = RevisionSpec.model_validate(provenance["revision"])
        unit.before = load_plan(revision / "source-plan.json")
    except (OSError, ValueError, KeyError, PlanError) as error:
        ctx.problems.append(f"{unit.id}: revision provenance unavailable: {error}")
        unit.status = UNREADABLE
        return unit
    ctx.bind(revision / "provenance.json", revision / "source-plan.json")
    if {plan_sha256(unit.before), str(provenance.get("before_sha256"))} != {source_sha256}:
        # Locks and scope are judged against the source; a substituted one proves nothing.
        ctx.problems.append(f"{unit.id}: source plan differs from the recorded original")
        unit.status, unit.before = UNVERIFIED, None
        return unit
    if provenance.get("brief") != brief.model_dump(mode="json"):
        ctx.problems.append(f"{unit.id}: brief differs from the one the revision was made for")
    saved = _object(provenance.get("backend"))
    if recorded != unit.spec or saved.get("conditions", unit.conditions) != unit.conditions:
        ctx.problems.append(f"{unit.id}: recorded revision request differs from the run config")
    unit.status, unit.seed = str(provenance.get("status")), provenance.get("seed")
    unit.checkpoint_id = saved.get("checkpoint_id")
    unit.diagnostics = {
        "backend_status": saved.get("status"),
        "attempts": _attempts(saved.get("attempts")),
        "guidance_scale": saved.get("guidance_scale"),
        "local_work": saved.get("work"),
        "text_calls": len(_objects(provenance.get("calls"))),
        "unrepresented": saved.get("unrepresented"),
    }
    if unit.status not in {"accepted", "noop"}:
        unit.status = "failed"
        return unit
    ctx.bind(folder / "plan.json")
    try:
        unit.plan = load_plan(folder / "plan.json")
    except (OSError, PlanError) as error:
        ctx.problems.append(f"{unit.id}: selected plan unavailable: {error}")
        unit.status = UNREADABLE
        return unit
    if provenance.get("after_sha256") != unit.plan_sha256:
        ctx.problems.append(f"{unit.id}: selected plan differs from its revision provenance")
    return unit


def _revision_units(ctx: _Context, ref: RevisionRunRef) -> tuple[_Run, list[_Unit], Record]:
    run = _run(ctx, ref.run, ref.label)
    plans = run.config.plans
    if plans is None or "revision" not in plans.params:
        raise ValueError(f"run {run.path} is not a plan revision run")
    params = RevisionPlanningParams.model_validate(plans.params)
    if not isinstance(params.backend, ERevisionBackend):
        raise ValueError(f"run {run.path} does not use the E revision backend")
    briefs = _briefs(ctx, run, plans.briefs, plans.briefs_dir)
    ctx.bind(run.path / "plans.jsonl", run.path / "alternatives.json")
    try:
        alternatives = check_report(run.path)
    except AlternativesError as error:
        raise ValueError(str(error)) from error
    if not alternatives["consistent"]:
        ctx.problems.append(
            f"{run.label}: revision alternatives are inconsistent or stale: "
            f"{alternatives.get('stale', 'see alternatives diagnostics')}"
        )
    records = {(str(r["brief"]), int(r["repetition"])): r for r in alternatives["alternatives"]}
    units = [
        _revision_unit(
            ctx,
            run,
            params,
            brief=brief,
            repetition=repetition,
            record=records.get((brief.id, repetition)),
            source_sha256=str(alternatives["source"]["plan_sha256"]),
        )
        for brief in briefs
        for repetition in range(plans.repeat)
    ]
    summary = {
        "run": str(run.path),
        "label": run.label,
        "source": alternatives["source"],
        "counts": alternatives["counts"],
        "consistent": alternatives["consistent"],
        "revision": params.revision.model_dump(mode="json"),
    }
    return run, units, summary


def _selector_status(plan: NarrativePlan, selector: str, expected: Any) -> tuple[Status, str]:
    try:
        actual = selector_value(plan, selector)
    except ValueError as error:
        return "failure", f"lock selector absent from the selected plan: {error}"
    if canonical(actual) == canonical(expected):
        return "success", "locked value held exactly"
    return "failure", f"locked value changed to {canonical(actual)}"


UNEVALUABLE: dict[str, str] = {
    "missing": "expected output is missing",
    UNREADABLE: "the unit's saved evidence is unreadable",
    UNVERIFIED: "the unit's saved evidence does not bind its recorded inputs",
}


def _words(unit: _Unit, request: Any) -> tuple[Status, str]:
    if unit.prose is None:
        if unit.kind == "story":
            return NO_OUTPUT, "no final prose: an operational failure, not non-adherence"
        return "unknown", "plan-only output has no prose to measure"
    low, high = request
    words = word_count(unit.prose)
    lower, upper = low * (1 - unit.tolerance), high * (1 + unit.tolerance)
    status: Status = "success" if lower <= words <= upper else "failure"
    return status, f"{words} words in the final prose; allowed {lower:g}-{upper:g}"


def _evaluate(unit: _Unit, predicate: str, request: Any) -> tuple[Status, str]:  # noqa: PLR0911
    if unit.status in UNEVALUABLE:
        return "unknown", UNEVALUABLE[unit.status]
    plan = unit.plan
    if plan is None:
        return NO_OUTPUT, "no saved plan: an operational failure, not non-adherence"
    if predicate == "scene_range":
        low, high = request
        count = len(plan.scenes)
        status: Status = "success" if low <= count <= high else "failure"
        return status, f"{count} scenes; requested {low}-{high}"
    if predicate == "target_words":
        return _words(unit, request)
    if predicate == "lock":
        return _selector_status(plan, request["selector"], request["value"])
    if predicate == "revision_lock" and unit.before is not None:
        try:
            before = selector_value(unit.before, request)
        except ValueError as error:
            return "unknown", f"lock selector absent from the source plan: {error}"
        return _selector_status(plan, request, before)
    if predicate == "revision_scope" and unit.before is not None and unit.spec is not None:
        errors = boundary_errors(unit.before, plan, unit.spec)
        return ("failure", "; ".join(errors)) if errors else ("success", "within the mask")
    if predicate == "validity":
        diagnostics = validate_plan(plan).diagnostics
        errors = [d.message for d in diagnostics if d.severity == "error"]
        return ("failure", "; ".join(errors)) if errors else ("success", "complete plan valid")
    return "unknown", "the predicate's source evidence is unavailable"


def _requests(unit: _Unit) -> list[tuple[str, Any]]:
    requests: list[tuple[str, Any]] = []
    for name in ("scene_range", "target_words"):
        if unit.conditions.get(name) is not None:
            requests.append((name, list(unit.conditions[name])))
    requests += [
        ("lock", {"selector": selector, "value": value})
        for selector, value in sorted(unit.locks.items())
    ]
    if unit.spec is not None:
        requests += [("revision_lock", selector) for selector in unit.spec.locks]
        requests.append(("revision_scope", unit.spec.mask.model_dump(mode="json")))
    requests.append(("validity", None))
    return requests


def _counts(rows: list[Record], statuses: tuple[str, ...]) -> Record:
    counter = Counter(str(row["status"]) for row in rows)
    return {"expected": len(rows), **{status: counter[status] for status in statuses}}


def _deterministic(units: list[_Unit]) -> Record:
    rows: list[Record] = []
    for unit in units:
        for predicate, request in _requests(unit):
            status, reason = _evaluate(unit, predicate, request)
            rows.append(
                {
                    "unit": unit.id,
                    "predicate": predicate,
                    "request": request,
                    "status": status,
                    "reason": reason,
                }
            )
    statuses = ("success", "failure", "unknown", NO_OUTPUT)
    return {
        "predicates": PREDICATES,
        "rows": rows,
        "by_predicate": {
            name: _counts([r for r in rows if r["predicate"] == name], statuses)
            for name in sorted({r["predicate"] for r in rows})
        },
        "total": _counts(rows, statuses),
    }


def _judgement_request(unit: _Unit, condition: str) -> Any:
    value = unit.conditions.get(condition)
    return None if value in (None, []) else value


def _judgement_status(unit: _Unit) -> str:
    if unit.output_sha256 is not None:
        return "pending"
    return "missing" if unit.status in {"missing", UNREADABLE, UNVERIFIED} else "no-output"


def _judgements(ctx: _Context, units: list[_Unit]) -> Record:
    rows: dict[tuple[str, str], Record] = {}
    by_id = {unit.id: unit for unit in units}
    for unit in units:
        for condition in JUDGEMENT_CONDITIONS:
            request = _judgement_request(unit, condition)
            if request is None:
                continue
            rows[unit.id, condition] = {
                "unit": unit.id,
                "condition": condition,
                "request": request,
                "status": _judgement_status(unit),
                "evidence": None,
                "reviewer": None,
                "note": None,
                "stale_assessments": 0,
                "synthetic_test_only_assessments": 0,
                "diagnostic_estimates": 0,
            }
    for path in ctx.request.judgements:
        source = ctx.resolve(path)
        review = JudgementReview.model_validate_json(source.read_text(encoding="utf-8"))
        ctx.bind(source)
        for item in review.assessments:
            row = rows.get((item.unit, item.condition))
            unit = by_id.get(item.unit)
            if row is None or unit is None:
                ctx.problems.append(
                    f"{source.name}: {item.unit} has no requested {item.condition} to assess"
                )
                continue
            if item.output_sha256 != unit.output_sha256:
                # Assessed against an output that is no longer the final selected one.
                row["stale_assessments"] += 1
            elif review.diagnostic_provenance is not None:
                row["diagnostic_estimates"] += 1
            elif review.test_only:
                row["synthetic_test_only_assessments"] += 1
            elif row["status"] in {"pending", "adheres", "does_not_adhere", "unknown"}:
                row.update(
                    status=item.rating, evidence="owner", reviewer=review.reviewer, note=item.note
                )
    listed = list(rows.values())
    statuses = ("pending", "adheres", "does_not_adhere", "unknown", "no-output", "missing")
    return {
        "rule": (
            "Genre, themes and natural-language endings need actual literary judgement. "
            "They stay pending until the owner assesses the current final output; model "
            "estimates are diagnostic and synthetic assessments are test-only, so neither "
            "can establish adherence. Assessments of a superseded output are stale."
        ),
        "rows": listed,
        "by_condition": {
            name: _counts([r for r in listed if r["condition"] == name], statuses)
            for name in JUDGEMENT_CONDITIONS
        },
        "total": _counts(listed, statuses),
    }


def retained_structure(plan: NarrativePlan, policy: dict[str, str]) -> Record | None:
    """The ID-independent structural view the report compares; None for an invalid plan."""
    try:
        encoded = encode_plan(plan, policy, {})
    except ValueError:
        return None
    return structural_content(cast(Record, encoded["view"]))


def _unknown_labels(value: Any) -> int:
    if isinstance(value, dict):
        row = cast(Record, value)
        own = int(row.get("status") in {"unknown", "uncertain", "ambiguous"})
        return own + sum(_unknown_labels(child) for child in row.values())
    if isinstance(value, list):
        return sum(_unknown_labels(child) for child in cast(list[Any], value))
    return 0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pair(left: _Unit, right: _Unit) -> Record:
    structure = (
        view_distance(left.view, right.view)
        if left.view is not None and right.view is not None
        else {"distance": None, "comparison": "unavailable"}
    )
    return {
        "left": left.id,
        "right": right.id,
        "structure_distance": structure["distance"],
        "structure_comparison": structure["comparison"],
        "uncertain": structure.get("uncertain", False),
        # Typed edges need resolvable references, so only valid plans have an edge distance.
        "edge_distance": multiset_distance(typed_edges(left.plan), typed_edges(right.plan))
        if left.valid and right.valid and left.plan is not None and right.plan is not None
        else None,
    }


def _diversity_group(name: str, units: list[_Unit]) -> Record:
    samples: list[Record] = []
    exact: dict[str, str] = {}
    structural: dict[str, str] = {}
    for unit in units:
        row: Record = {
            "unit": unit.id,
            "seed": unit.seed,
            "status": unit.status,
            "plan_sha256": unit.plan_sha256,
            "structure_sha256": None if unit.view is None else identity(unit.view),
            "unknown_labels": None if unit.view is None else _unknown_labels(unit.view),
            "exact_duplicate_of": None,
            "structural_duplicate_of": None,
        }
        if row["plan_sha256"] is not None:
            first = exact.setdefault(row["plan_sha256"], unit.id)
            row["exact_duplicate_of"] = None if first == unit.id else first
        if row["structure_sha256"] is not None:
            first = structural.setdefault(row["structure_sha256"], unit.id)
            row["structural_duplicate_of"] = None if first == unit.id else first
        samples.append(row)
    available = [unit for unit in units if unit.plan is not None]
    # Duplicates stay in the pairs: identical alternatives are zero-distance evidence.
    pairs = [_pair(a, b) for a, b in combinations(available, 2)]
    structures = [p["structure_distance"] for p in pairs if p["structure_distance"] is not None]
    edges = [p["edge_distance"] for p in pairs if p["edge_distance"] is not None]
    return {
        "group": name,
        "samples": samples,
        "pairs": pairs,
        "summary": {
            "expected": len(units),
            "with_output": len(available),
            "without_output": len(units) - len(available),
            "exact_distinct": len(exact),
            "exact_duplicates": len(available) - len(exact),
            "structurally_distinct": len(structural),
            "structural_duplicates": sum(row["structure_sha256"] is not None for row in samples)
            - len(structural),
            "pairs": len(pairs),
            "unavailable_pairs": len(pairs) - len(structures),
            "uncertain_pairs": sum(bool(p["uncertain"]) for p in pairs),
            "mean_structure_distance": _mean(structures),
            "mean_edge_distance": _mean(edges),
        },
    }


def _diversity(units: list[_Unit]) -> Record:
    groups: dict[str, list[_Unit]] = {}
    for unit in units:
        groups.setdefault(f"{unit.kind}/{unit.label}/{unit.brief.id}", []).append(unit)
    return {
        "structure_metric": {"name": STRUCTURE_METRIC, "definition": STRUCTURE},
        "edge_metric": {"name": EDGE_METRIC, "definition": EDGES},
        "groups": [_diversity_group(name, members) for name, members in sorted(groups.items())],
    }


@dataclass
class _Identity:
    checkpoint: Record | None
    summary: Record
    policy: dict[str, str]
    policy_status: str
    dataset: Path | None


def _identity(ctx: _Context, units: list[_Unit]) -> _Identity:
    ids = {unit.checkpoint_id for unit in units if unit.checkpoint_id is not None}
    if len(ids) > 1:
        raise ValueError(f"report covers one E checkpoint; found {sorted(ids)}")
    paths = sorted({unit.checkpoint_path for unit in units if unit.checkpoint_path})
    checkpoint: Record | None = None
    summary: Record = {"id": next(iter(ids), None), "paths": paths, "available": False}
    for name in paths:
        path = Path(name)
        try:
            candidate = inspect_checkpoint(path)
        except (OSError, ValueError, KeyError) as error:
            summary["reason"] = f"checkpoint unavailable: {error}"
            continue
        if ids and candidate["id"] not in ids:
            ctx.problems.append(f"checkpoint file {path} no longer holds the recorded checkpoint")
            continue
        ctx.bind(path)
        checkpoint = candidate
        binding = candidate["binding"]
        summary.update(
            id=candidate["id"],
            available=True,
            file_sha256=file_sha(path),
            test_only=binding["config"]["test_only"],
            representation=binding["representation"],
            condition_id=binding["condition_id"],
            guidance_policy=binding["guidance_policy"],
            dataset_id=binding["dataset_id"],
            base_id=binding["base_id"],
            step=candidate["step"],
            scientific_evidence={
                "status": binding["scientific_evidence"]["status"],
                "decision_id": binding["scientific_evidence"].get("decision_id"),
            },
        )
        break
    found = None if checkpoint is None else training_policy(checkpoint)
    if found is not None:
        policy, dataset = found
        return _Identity(checkpoint, summary, policy, "verified", Path(dataset["dataset"]))
    if ctx.request.target_policy is None:
        raise ValueError(
            "the checkpoint's training dataset is unavailable; supply target_policy to "
            "compare structure under a declared, unverified policy"
        )
    declared = {k: v for k, v in sorted(ctx.request.target_policy.items()) if v != "excluded"}
    return _Identity(checkpoint, summary, declared, "declared-unverified", None)


@dataclass
class _Population:
    """A known population's report row and its records, which never leave the report."""

    row: Record
    records: list[Record] | None = None


def _population(ctx: _Context, name: str, origin: str, path: Path, role: str) -> _Population:
    view = load_view(path)
    check_index(view)
    if view.get("codec_id") != CODEC_ID:
        raise ValueError(f"training population {name} uses another codec")
    rows = view_records(view)
    ctx.bind(path)
    sources = [_object(row.get("provenance")) for row in rows]
    return _Population(
        {
            "name": name,
            "origin": origin,
            "role": role,
            "path": str(path),
            "status": "verified",
            "id": view["id"],
            "kind": view.get("kind"),
            "split": view.get("split"),
            "test_only": bool(view.get("test_only", False)),
            "records": len(rows),
            "works": len({source.get("work_id") for source in sources} - {None}),
            "authors": len({source.get("author_id") for source in sources} - {None}),
        },
        rows,
    )


def _fine_tuning(ctx: _Context, dataset: Path | None) -> _Population:
    """The conditional fine-tuning examples the checkpoint is bound to, lineage rechecked."""
    row: Record = {"name": "conditional-fine-tuning", "origin": "checkpoint binding"}
    if dataset is None:
        row.update(
            status="unavailable",
            reason="the checkpoint's bound training dataset is unavailable or changed",
        )
        return _Population(row)
    try:
        return _population(ctx, row["name"], row["origin"], dataset, "other")
    except (OSError, ValueError, KeyError) as error:
        row.update(path=str(dataset), status="changed", reason=str(error))
        return _Population(row)


def _protected(ctx: _Context, ref: PopulationRef) -> _Population:
    path = ctx.resolve(ref.path)
    view = load_view(path)
    if view.get("role") != "evaluator-only" and view.get("split") not in {"validation", "test"}:
        raise ValueError(f"protected population {ref.name} is not evaluator-only material")
    ctx.bind(path)
    rows = view_records(view)
    row = {"name": ref.name, "path": str(path), "id": view["id"], "split": view.get("split")}
    return _Population({**row, "records": len(rows)}, rows)


def _pretraining(checkpoint: Record | None) -> Record:
    if checkpoint is None:
        return {"status": "unknown", "reason": "checkpoint unavailable"}
    binding = checkpoint["binding"]
    model = next((Path(n) for n in binding["inputs"] if Path(n).name == "model.json"), None)
    try:
        base = read_sealed(model) if model is not None else None
    except (OSError, ValueError):
        base = None
    if base is None or base.get("id") != binding["base_id"]:
        return {
            "status": "unknown",
            "base_id": binding["base_id"],
            "reason": "base model record is unavailable or changed",
        }
    if base.get("initialization") == "local-random-fixture":
        return {
            "status": "none",
            "base_id": base["id"],
            "initialization": base["initialization"],
            "reason": "random test-only initialization; no pretraining data",
        }
    pretraining = cast(Record, base.get("pretraining") or {})
    return {
        "status": "unknown",
        "base_id": base["id"],
        "initialization": base.get("initialization"),
        "model_id": pretraining.get("model_id"),
        "description": pretraining.get("pretraining_description"),
        "reason": "pretraining data is described, not enumerated; overlap with it is unknown",
    }


def _indexed(rows: list[Record]) -> list[Record]:
    return [
        {
            "id": row["id"],
            "view": structural_content(cast(Record, row["view"])),
            "provenance": row.get("provenance", {}),
        }
        for row in rows
    ]


def _nearest(rows: list[Record]) -> dict[str, Record]:
    return {
        row["query_id"]: {
            "matches": [
                {
                    "record_id": match["record_id"],
                    "work_id": _object(match.get("provenance")).get("work_id"),
                    "distance": match["distance"],
                    "uncertain": match.get("uncertain", False),
                }
                for match in row["matches"]
            ]
        }
        for row in rows
    }


def _similarity_summary(nearest: dict[str, Record]) -> Record:
    best = [
        row["matches"][0]["distance"]
        for row in nearest.values()
        if row["matches"] and row["matches"][0]["distance"] is not None
    ]
    return {
        "queries": len(nearest),
        "exact_structural_matches": sum(distance == 0 for distance in best),
        "min_distance": min(best) if best else None,
        "mean_nearest_distance": _mean(best),
    }


def _compare(population: _Population, queries: list[Record], neighbours: int) -> None:
    nearest = (
        {}
        if population.records is None
        else _nearest(neighbour_rows(_indexed(population.records), queries, neighbours)[0])
    )
    population.row.update(nearest=nearest, summary=_similarity_summary(nearest))


def _audit(training: _Population, guarded: _Population) -> Record:
    """Protected records queried against a training index; any shared source is overlap.

    Every training record is compared, so the overlap list is exhaustive, not only the
    structurally nearest neighbours.
    """
    index = training.records or []
    rows, unresolved = neighbour_rows(index, guarded.records or [], max(1, len(index)))
    return {
        "training": training.row["name"],
        "protected": guarded.row["name"],
        "protected_overlap": unresolved,
        "overlaps": [
            {"protected_record": row["query_id"], **match}
            for row in rows
            for match in row["matches"]
            if match["exact_source"] or match["exact_work"] or match["author_overlap"]
        ],
    }


def _g3(populations: list[_Population], evidence: Any) -> Record:
    supplied = [p.row["name"] for p in populations if p.row.get("role") == "g3"]
    if supplied:
        return {"status": "supplied", "populations": supplied}
    if evidence == "passed":
        return {"status": "missing", "reason": "checkpoint has G3 evidence; supply its population"}
    return {"status": "not-applicable", "reason": f"scientific evidence: {evidence}"}


def _training(ctx: _Context, identity_: _Identity, units: list[_Unit]) -> Record:
    populations = [_fine_tuning(ctx, identity_.dataset)]
    populations += [
        _population(ctx, ref.name, "supplied", ctx.resolve(ref.path), ref.role)
        for ref in ctx.request.training_populations
    ]
    protected = [_protected(ctx, ref) for ref in ctx.request.protected_populations]
    if {p.row.get("id") for p in populations} & {p.row["id"] for p in protected}:
        raise ValueError("a protected population cannot also be a training population")
    queries = [
        {"id": unit.id, "view": unit.view, "provenance": {}}
        for unit in units
        if unit.view is not None
    ]
    for population in [*populations, *protected]:
        _compare(population, queries, ctx.request.neighbours)
    audits = [
        _audit(training, guarded)
        for guarded in protected
        for training in populations
        if training.records is not None
    ]
    g3 = _g3(populations, _object(identity_.summary.get("scientific_evidence")).get("status"))
    pretraining = _pretraining(identity_.checkpoint)
    missing = [p.row["name"] for p in populations if p.row["status"] != "verified"]
    missing += ["g3-population"] if g3["status"] == "missing" else []
    limitations = list(SIMILARITY_LIMITATIONS)
    if pretraining["status"] != "none":
        limitations.append(
            f"Pretraining data overlap is unknown: {pretraining['reason']}. Similarity to "
            "pretraining material cannot be assessed."
        )
    return {
        "metric": {"name": STRUCTURE_METRIC, "definition": STRUCTURE},
        "populations": [p.row for p in populations],
        "g3_population": g3,
        "pretraining": pretraining,
        "lineage": {"complete": not missing, "missing": missing},
        "protected_populations": [p.row for p in protected],
        "protected_audits": audits,
        "protected_overlap": any(audit["protected_overlap"] for audit in audits),
        "limitations": limitations,
    }


def _gd(ctx: _Context) -> tuple[Literal["C", "D"], str, Record | None]:
    if ctx.request.reference.gd_decision is None:
        return "C", "no G-D decision supplied; the identified C baseline is the reference", None
    folder = ctx.resolve(ctx.request.reference.gd_decision)
    path = folder / "decision.json"
    try:
        saved = _read(path)
        ctx.bind(path, folder / "reference-baseline.json")
        replayed = gd_decision(Path(saved["batch"]), [])
    except (OSError, ValueError, KeyError) as error:
        return "C", f"G-D decision unreadable: {error}", None
    summary = {
        "path": str(folder),
        "id": saved.get("id"),
        "outcome": saved.get("outcome"),
        "criteria_outcome": saved.get("criteria_outcome"),
        "test_only": saved.get("test_only"),
        "reference_baseline": saved.get("reference_baseline"),
    }
    ignored = {"id", "realisation"}
    if {k: v for k, v in replayed.items() if k not in ignored} != {
        k: v for k, v in saved.items() if k not in ignored
    }:
        return "C", "G-D decision no longer replays from its bound evidence", summary
    named = _object(saved.get("reference_baseline")).get("condition")
    if saved.get("outcome") == "passed" and saved.get("test_only") is False and named == "D":
        return "D", "an actual G-D pass replays and names D as the reference", summary
    reason = f"G-D outcome {saved.get('outcome')}"
    reason += " (synthetic test-only)" if saved.get("test_only") else ""
    return "C", f"{reason} does not permit D; the identified C baseline is the reference", summary


def _reference_units(ctx: _Context, ref: ReferenceRunRef) -> tuple[Record, list[_Unit]]:
    run = _run(ctx, ref.run, None)
    stories = run.config.stories
    pipeline = {"C": "baseline_c", "D": "baseline_d"}[ref.condition]
    spec = (
        None
        if stories is None
        else next((c for c in stories.conditions if c.name == ref.condition), None)
    )
    if stories is None or spec is None or spec.pipeline != pipeline:
        raise ValueError(f"reference run {run.path} has no {ref.condition} {pipeline} condition")
    briefs = {brief.id: brief for brief in _briefs(ctx, run, stories.briefs, stories.briefs_dir)}
    ctx.bind(run.path / "stories.jsonl")
    units: list[_Unit] = []
    for rec in read_stories(run.path):
        if rec.condition != ref.condition or rec.brief not in briefs:
            continue
        unit = _Unit(
            id=f"reference/{run.label}/{ref.condition}/{rec.brief}/r{rec.repetition + 1}",
            kind="story",
            label=run.label,
            run=run.path,
            brief=briefs[rec.brief],
            repetition=rec.repetition,
            status=rec.status,
            seed=rec.seed,
        )
        if rec.status == "ok":
            plan_file = run.path / rec.path / "plan.json"
            try:
                unit.plan = load_plan(plan_file)
            except (OSError, PlanError) as error:
                ctx.problems.append(f"{unit.id}: reference plan unavailable: {error}")
            ctx.bind(plan_file)
        units.append(unit)
    identity_ = {
        "condition": ref.condition,
        "pipeline": pipeline,
        "params": spec.params,
        "run": str(run.path),
        "config_sha256": run.manifest.config_hash,
        "models": {name: role.model_dump(mode="json") for name, role in run.config.roles.items()},
    }
    return identity_, units


def _reference(ctx: _Context, policy: dict[str, str], units: list[_Unit]) -> Record:
    permitted, reason, gd = _gd(ctx)
    identities: list[Record] = []
    references: list[_Unit] = []
    problems: list[str] = []
    for ref in ctx.request.reference.runs:
        if ref.condition != permitted:
            problems.append(
                f"{ref.condition} outputs in {ref.run} are not used: {permitted} is the "
                "permitted reference"
            )
            continue
        identity_, rows = _reference_units(ctx, ref)
        identities.append(identity_)
        references.extend(rows)
    selection: Record | None = None
    if ctx.request.reference.c_selection is not None:
        path = ctx.resolve(ctx.request.reference.c_selection)
        chosen = load_selection(path)
        ctx.bind(path)
        selection = {"path": str(path), "id": chosen.id, "tier": chosen.tier}
        selection["test_only"] = chosen.test_only
        tiers = {str(i["params"].get("tier")) for i in identities if i["condition"] == "C"}
        if permitted == "C" and tiers - {chosen.tier}:
            problems.append("C reference runs differ from the selected C tier")
    for unit in references:
        unit.compare_with(policy)
    comparisons: list[Record] = []
    for unit in units:
        if unit.kind != "story":
            continue
        same = [
            r
            for r in references
            if r.brief.id == unit.brief.id and _brief_sha256(r.brief) == _brief_sha256(unit.brief)
        ]
        pairs = [_pair(unit, r) for r in same if r.plan is not None]
        distances = [p["structure_distance"] for p in pairs if p["structure_distance"] is not None]
        comparisons.append(
            {
                "unit": unit.id,
                "reference_outputs": [r.id for r in same],
                "reference_without_output": [r.id for r in same if r.plan is None],
                "pairs": pairs if unit.plan is not None else [],
                "nearest_structure_distance": min(distances) if distances and unit.plan else None,
            }
        )
    available = [c["nearest_structure_distance"] for c in comparisons]
    measured = [d for d in available if d is not None]
    return {
        "permitted": permitted,
        "reason": reason,
        "gd_decision": gd,
        "identities": identities,
        "c_selection": selection,
        "status": "compared" if measured else "unavailable",
        "problems": problems,
        "comparisons": comparisons,
        "summary": {
            "expected": len(comparisons),
            "compared": len(measured),
            "uncompared": len(comparisons) - len(measured),
            "mean_nearest_structure_distance": _mean(measured),
        },
    }


def _realisation(ctx: _Context, units: list[_Unit]) -> Record:
    stories = [unit for unit in units if unit.kind == "story" and unit.prose is not None]
    packets: dict[str, list[Record]] = {}
    for folder in ctx.request.realisation:
        path = ctx.resolve(folder)
        packet = realisation.read_packet(path)
        ctx.bind(path / "packet.json", *sorted((path / "history").glob("*.json")))
        match = next(
            (
                unit
                for unit in stories
                if unit.record is not None
                and Path(packet.run).resolve() == unit.run
                and (packet.story.condition, packet.story.brief, packet.story.repetition)
                == (unit.record.condition, unit.brief.id, unit.repetition)
            ),
            None,
        )
        if match is None:
            ctx.problems.append(f"realisation packet {path} binds no reported E story")
            continue
        summary = realisation.summarise(path)
        if summary["story_sha256"] != match.output_sha256:
            summary["stale"] = [*summary["stale"], "packet prose is not the final selected story"]
        packets.setdefault(match.id, []).append({"packet": str(path), **summary})
    rows: list[Record] = []
    for unit in stories:
        evidence = packets.get(unit.id, [])
        current = [row for row in evidence if not row["stale"]]
        status = "missing"
        if evidence:
            status = "stale" if not current else current[-1]["owner_status"]
        rows.append(
            {
                "unit": unit.id,
                "story_sha256": unit.output_sha256,
                "status": status,
                "packets": evidence,
            }
        )
    return {
        "rule": (
            "Realisation evidence counts only from packets bound to the current final selected "
            "prose and plan; changed outputs make earlier reviews stale."
        ),
        "rows": rows,
        "summary": _counts(rows, ("complete", "pending", "stale", "missing")),
    }


def build(manifest: Path) -> Record:
    """Recompute the complete report content from a request; raises ValueError or OSError."""
    manifest = manifest.resolve()
    try:
        request = ReportRequest.model_validate_json(manifest.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"invalid report request: {error}") from error
    ctx = _Context(manifest.parent, request, {manifest})
    units: list[_Unit] = []
    runs: list[Record] = []
    labels: list[str] = []
    for ref in request.story_runs:
        run, rows = _story_units(ctx, ref)
        labels.append(run.label)
        comparison = run.path / "comparison.json"
        policy = _read(comparison).get("resource_policy") if comparison.is_file() else None
        runs.append(
            {"kind": "story", "label": run.label, "run": str(run.path), "resource_policy": policy}
        )
        units.extend(rows)
    revisions: list[Record] = []
    for ref in request.revision_runs:
        run, rows, summary = _revision_units(ctx, ref)
        labels.append(run.label)
        revisions.append(summary)
        units.extend(rows)
    if len(labels) != len(set(labels)):
        raise ValueError("run labels must be distinct; set label for runs with the same name")
    identity_ = _identity(ctx, units)
    for unit in units:
        unit.compare_with(identity_.policy)
    content: Record = {
        "kind": KIND,
        "report_version": REPORT_VERSION,
        "scope": "development evidence; not a G-E decision",
        "request": {"path": str(manifest), **request.model_dump(mode="json")},
        "e_identity": {
            "checkpoint": identity_.summary,
            "target_policy": identity_.policy,
            "target_policy_status": identity_.policy_status,
        },
        "test_only": bool(identity_.summary.get("test_only", True)),
        "runs": runs,
        "revision_runs": revisions,
        "units": [_unit_summary(unit) for unit in units],
        "adherence": {
            "deterministic": _deterministic(units),
            "judgement": _judgements(ctx, units),
        },
        "model_diagnostics": {
            "rule": "Diagnostic only: model attempts and checks are not condition adherence.",
            "units": {unit.id: unit.diagnostics for unit in units},
        },
        "diversity": _diversity(units),
        "training_similarity": _training(ctx, identity_, units),
        "reference": _reference(ctx, identity_.policy, units),
        "realisation": _realisation(ctx, units),
    }
    content["pending"] = _pending(content)
    content["problems"] = ctx.problems
    content["consistent"] = not ctx.problems
    content["statement"] = NOT_A_DECISION
    content["inputs"] = snapshot(sorted(ctx.files))
    return content


def _unit_summary(unit: _Unit) -> Record:
    return {
        "unit": unit.id,
        "kind": unit.kind,
        "run": str(unit.run),
        "path": unit.path,
        "brief": unit.brief.id,
        "brief_sha256": _brief_sha256(unit.brief),
        "repetition": unit.repetition,
        "seed": unit.seed,
        "status": unit.status,
        "checkpoint_id": unit.checkpoint_id,
        "conditions": unit.conditions,
        "allowed_constraints": {
            "locks": unit.locks,
            "revision": None if unit.spec is None else unit.spec.model_dump(mode="json"),
        },
        "plan_sha256": unit.plan_sha256,
        "output_sha256": unit.output_sha256,
    }


def _pending(content: Record) -> list[str]:
    pending: list[str] = []
    judgement = content["adherence"]["judgement"]["total"]
    if judgement["pending"]:
        pending.append(f"{judgement['pending']} literary adherence judgements await the owner")
    realised = content["realisation"]["summary"]
    waiting = realised["pending"] + realised["stale"] + realised["missing"]
    if waiting:
        pending.append(f"{waiting} final stories lack current complete realisation review")
    lineage = content["training_similarity"]["lineage"]
    if not lineage["complete"]:
        pending.append(f"training lineage incomplete: {', '.join(lineage['missing'])}")
    if content["reference"]["status"] != "compared":
        pending.append(f"no {content['reference']['permitted']} reference outputs compared")
    if content["test_only"]:
        pending.append("test-only checkpoint: software evidence only")
    pending.append("G-E decision: separate registered comparison and owner review")
    return pending


def _markdown(report: Record) -> str:
    deterministic = report["adherence"]["deterministic"]
    judgement = report["adherence"]["judgement"]
    training = report["training_similarity"]
    reference = report["reference"]
    lines = [
        "# Candidate E condition adherence, diversity and training similarity",
        "",
        f"Report `{report['id']}`; checkpoint `{report['e_identity']['checkpoint']['id']}`"
        + (" (test-only)" if report["test_only"] else ""),
        "",
        report["statement"],
        "",
        "## Deterministic compliance",
        "",
        "| Predicate | Expected | Success | Failure | Unknown | No output |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {c['expected']} | {c['success']} | {c['failure']} | {c['unknown']} "
            f"| {c[NO_OUTPUT]} |"
            for name, c in deterministic["by_predicate"].items()
        ),
        "",
        *(f"- `{name}`: {text}" for name, text in deterministic["predicates"].items()),
        "",
        "## Literary judgement (pending until assessed)",
        "",
        judgement["rule"],
        "",
        "| Condition | Expected | Pending | Adheres | Does not | Unknown | No output | Missing |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {c['expected']} | {c['pending']} | {c['adheres']} | "
            f"{c['does_not_adhere']} | {c['unknown']} | {c['no-output']} | {c['missing']} |"
            for name, c in judgement["by_condition"].items()
        ),
        "",
        "## Structural diversity across seeds",
        "",
        report["diversity"]["structure_metric"]["definition"],
        "",
        "| Group | Expected | With output | Exact distinct | Structurally distinct | "
        "Mean structure distance | Mean edge distance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {g['group']} | {g['summary']['expected']} | {g['summary']['with_output']} | "
            f"{g['summary']['exact_distinct']} | {g['summary']['structurally_distinct']} | "
            f"{g['summary']['mean_structure_distance']} | {g['summary']['mean_edge_distance']} |"
            for g in report["diversity"]["groups"]
        ),
        "",
        "## Training similarity and lineage",
        "",
        "| Population | Origin | Status | Records | Exact structural matches | Min distance |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {p['name']} | {p['origin']} | {p['status']} | {p.get('records', '-')} | "
            f"{p['summary']['exact_structural_matches']} | {p['summary']['min_distance']} |"
            for p in training["populations"]
        ),
        "",
        f"- G3 population: {training['g3_population']['status']}",
        f"- Pretraining: {training['pretraining']['status']} ({training['pretraining']['reason']})",
        f"- Lineage complete: {training['lineage']['complete']}",
        f"- Protected overlap: {training['protected_overlap']}",
        *(f"- {text}" for text in training["limitations"]),
        "",
        "## Reference baseline",
        "",
        f"Reference: {reference['permitted']}. {reference['reason']}",
        f"Compared {reference['summary']['compared']} of {reference['summary']['expected']} "
        f"E stories; mean nearest structure distance "
        f"{reference['summary']['mean_nearest_structure_distance']}.",
        *(f"- {problem}" for problem in reference["problems"]),
        "",
        "## Realisation evidence",
        "",
        report["realisation"]["rule"],
        "",
        *(f"- {name}: {count}" for name, count in report["realisation"]["summary"].items()),
        "",
        "## Pending",
        "",
        *(f"- {item}" for item in report["pending"]),
        "",
        "## Problems",
        "",
        *(f"- {problem}" for problem in report["problems"] or ["None"]),
        "",
    ]
    return "\n".join(lines)


def write(manifest: Path, out: Path) -> Record:
    if out.exists():
        raise ValueError("report folder already exists; choose a new folder to keep history")
    report = create_record(out / REPORT, build(manifest))
    write_text_lf(out / "report.md", _markdown(report))
    return report


def check(folder: Path) -> tuple[Record, list[str]]:
    """Replay a saved report from its request; the reasons it no longer holds, if any."""
    saved = read_sealed(folder / REPORT)
    if saved.get("kind") != KIND:
        raise ValueError(f"not a {KIND}: {folder}")
    changed = [
        f"input changed or missing: {name}"
        for name, checksum in saved["inputs"].items()
        if not Path(name).is_file() or file_sha(Path(name)) != checksum
    ]
    try:
        current = build(Path(saved["request"]["path"]))
    except (OSError, ValueError) as error:
        return saved, [*changed, f"report can no longer be rebuilt: {error}"]
    expected = {key: value for key, value in saved.items() if key not in {"id", "version"}}
    changed += [
        f"section differs from current evidence: {key}"
        for key in sorted(expected.keys() | current.keys())
        if expected.get(key) != current.get(key)
    ]
    return saved, changed


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="conditional_report_command", required=True)
    create = commands.add_parser("build", help="write a new report from a request")
    create.add_argument("request", type=Path, help="JSON request; paths relative to this file")
    create.add_argument("--out", type=Path, required=True)
    replay = commands.add_parser("check", help="replay a saved report against current evidence")
    replay.add_argument("report", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.conditional_report_command == "build":
            report = write(args.request, args.out)
            print(f"conditional-report {report['id']}: {args.out}")
            return 0
        _, reasons = check(args.report)
    except (OSError, ValueError, KeyError) as error:
        print(f"nde conditional-report: {error}", file=sys.stderr)
        return 2
    for reason in reasons:
        print(reason, file=sys.stderr)
    print("conditional-report " + ("STALE" if reasons else "current") + f": {args.report}")
    return 1 if reasons else 0
