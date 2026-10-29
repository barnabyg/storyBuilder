"""Registered G-E development comparison: readiness, protocol, bound run and replay.

G-E asks whether Candidate E is competitive enough to justify Phase 8's owner
reading. This module prepares that comparison and never decides it:

- readiness is a no-spend check that the E checkpoint is authorised by a genuine
  passed G3 decision that still replays, with verified training lineage;
- registration freezes, before generation, E's checkpoint, data, conditioning,
  guidance and hybrid completion, the identified C/D reference, the exact dev
  cohort and seeds, the resource limits, the validator comparison, the
  structural-competitiveness criterion, the owner sample and review budget, and
  the one-development-iteration policy;
- a registered run executes the bound story workflow and records complete
  denominators, failures, costs, adherence/diversity/overlap evidence and every
  artifact identity, without applying the registered criteria.

The G-E decision is separate. Nothing here selects held-out (test) briefs or
launches another gate.
"""

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlparse

import yaml
from pydantic import Field, model_validator

from nde import briefs, gd_report
from nde.annotations.records import digest
from nde.cd_comparison import selected_c_problems
from nde.conditional import report as evidence_report
from nde.conditional.completion import PROMPT_ID
from nde.conditional.data import load_dataset
from nde.conditional.training import inspect_checkpoint
from nde.corpus.records_io import create_record, file_sha, read_json, read_sealed
from nde.eval.g1 import file_checksum
from nde.files import write_text_lf
from nde.gitinfo import git
from nde.llm.config import ModelBinding, Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import PlanError, load_plan, schema_json
from nde.plan.conditional import ConditionalPlanningParams
from nde.plan.revision import identity as plan_sha256
from nde.reconstruction import g3
from nde.reconstruction.overlap import load_view, view_path
from nde.runs.config import ConditionSpec, ConfigError, LoadedConfig, load_config
from nde.runs.prompts import PromptTemplate
from nde.runs.records import (
    CallRecord,
    LocalWorkRecord,
    Manifest,
    StoryRecord,
    read_calls,
    read_local_work,
    read_manifest,
    read_stories,
)
from nde.runs.runner import RunOptions, execute, prepare
from nde.schema.brief import Brief
from nde.stories.execute import story_conditions, story_templates
from nde.stories.resources import provider_categories
from nde.tiers import load_selection

type Record = dict[str, Any]

PROTOCOL_KIND = "ge-protocol"
PROTOCOL_VERSION = "1"
RUN_KIND = "ge-dev-run"
RUN_VERSION = "1"
RUN_RECORD = "ge-run.json"
CLAIM_KIND = "ge-run-claim"
ALREADY_RUN = (
    "this protocol has already been run (its run claim exists): a fixed cohort is run "
    "once, with no top-up or substituted rerun"
)
EVIDENCE = "ge-evidence"
CANDIDATE = "E"
CANDIDATE_LABEL = "e"
REFERENCE_PIPELINES = {"C": "baseline_c", "D": "baseline_d"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
VALIDITY = ("success", "failure", "no-output", "unknown")
# Errors that block an evaluation; every one is reported, never raised to the caller.
EVALUATION_ERRORS = (OSError, ValueError, KeyError, TypeError, ConfigError, briefs.BriefError)
SCOPE = (
    "G-E development comparison on frozen dev briefs. Not the Phase 8 held-out "
    "experiment and not a G-E decision."
)
FAIL_ACTION = (
    "If G-E fails, permit at most the one registered development iteration, then "
    "stop and keep the identified C/D reference planner. Never extend the cohort."
)
HELD_OUT = (
    "Only dev briefs are selected, generated or bound. The Phase 8 held-out cohort is "
    "never generated or read, and no later gate is launched."
)
NOT_A_DECISION = (
    "Registered G-E development evidence. Criteria are applied, and owner review "
    "is judged, only by the separate G-E decision; no later gate is launched."
)


class ValidatorComparison(Strict):
    """E's plan-validity pass rate against the reference's: a descriptive margin."""

    metric: Literal["plan_validity"] = "plan_validity"
    analysis: Literal["pass_rate_difference"]
    # E's rate minus the reference's rate must be at least minus this margin.
    margin: float = Field(ge=0, le=1, allow_inf_nan=False)
    # E's own rate must also reach this; register 0 for no absolute target.
    minimum_candidate_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    aggregation: Literal["pooled", "brief_macro"]
    denominator: Literal["all_expected_units"] = "all_expected_units"
    no_output: Literal["not_valid"] = "not_valid"
    unknown: Literal["pending"] = "pending"
    boundary: Literal["inclusive"] = "inclusive"


class StructuralCriterion(Strict):
    """The owner's blinded paired judgement of structure on the frozen dev sample."""

    analysis: Literal["owner_paired_competitive_share"]
    # Shown to the owner verbatim with each blinded pair of plans.
    question: str = Field(min_length=10)
    # Competitive pairs over the owner-reviewed denominator must reach this share.
    minimum_share: float = Field(gt=0, le=1, allow_inf_nan=False)
    ties: Literal["competitive"] = "competitive"
    unknown_rating: Literal["not_competitive"] = "not_competitive"
    candidate_without_output: Literal["not_competitive"] = "not_competitive"
    reference_without_output: Literal["excluded_and_reported"] = "excluded_and_reported"
    missing_evidence: Literal["pending"] = "pending"
    boundary: Literal["inclusive"] = "inclusive"


class OwnerReview(Strict):
    """Who reviews, how many frozen dev pairs, and the owner time they may take."""

    reviewer: str = Field(min_length=1)
    sample_size: int = Field(ge=1)
    sample_seed: int
    minutes_per_item: float = Field(gt=0, allow_inf_nan=False)
    max_minutes: float = Field(gt=0, allow_inf_nan=False)
    blinding: Literal["opaque_plan_labels"] = "opaque_plan_labels"

    @model_validator(mode="after")
    def _budgeted(self) -> Self:
        if self.sample_size * self.minutes_per_item > self.max_minutes:
            raise ValueError("owner sample exceeds its registered review time budget")
        return self


class Iteration(Strict):
    policy: Literal["at_most_one_development_iteration"] = "at_most_one_development_iteration"
    attempt: Literal[1, 2]
    # The first attempt's protocol, for the single permitted development iteration.
    previous_protocol: str | None = None
    change: str | None = None

    @model_validator(mode="after")
    def _bound(self) -> Self:
        if (self.attempt != 1) != (self.previous_protocol is not None and bool(self.change)):
            raise ValueError(
                "the development iteration (attempt 2) names its previous protocol and "
                "change; attempt 1 names neither"
            )
        return self


class Resources(Strict):
    """Hard limits, each equal to the bound configuration's value."""

    max_attempts: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    max_attention_cells: int = Field(ge=1)
    max_seconds: float = Field(gt=0, allow_inf_nan=False)
    completion_max_tokens: int = Field(ge=1)
    max_repairs: int = Field(ge=0)
    max_length_attempts: int = Field(ge=1)
    max_revision_passes: int = Field(ge=0)
    budget_tolerance: float = Field(gt=0, lt=1, allow_inf_nan=False)
    per_run_usd: float = Field(gt=0, allow_inf_nan=False)
    per_day_usd: float = Field(gt=0, allow_inf_nan=False)
    resource_policy: Literal["1"] = "1"


class Cohort(Strict):
    briefs: list[str] = Field(min_length=1)
    repeat: int = Field(ge=1)


class ReferenceRegistration(Strict):
    condition: Literal["C", "D"]
    # The owner's frozen C tier selection; identifies a C reference.
    c_selection: str | None = None
    # A `nde gd-report` folder, replayed from its closed batch; identifies a D reference.
    gd_decision: str | None = None
    gd_batch: str | None = None
    gd_realisation: list[str] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _identified(self) -> Self:
        if (self.gd_decision is None) != (self.gd_batch is None):
            raise ValueError("a G-D decision is replayed from its batch: give both or neither")
        if self.condition == "C" and self.c_selection is None:
            raise ValueError("a C reference is identified by the owner's C selection")
        if self.condition == "D" and self.gd_decision is None:
            raise ValueError("a D reference requires a replayed actual G-D pass naming D")
        return self


class PopulationRegistration(Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    path: str = Field(min_length=1)


class GERegistration(Strict):
    """Everything frozen before a G-E run; paths are relative to this file."""

    version: Literal["1"] = "1"
    id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    test_only: bool = False
    config: str = Field(min_length=1)
    reference: ReferenceRegistration
    cohort: Cohort
    resources: Resources
    # The G3 protocol's registered training population, for lineage.
    g3_population: str | None = None
    protected_populations: list[PopulationRegistration] = Field(
        default_factory=list[PopulationRegistration]
    )
    neighbours: int = Field(default=3, ge=1, le=50)
    validator_comparison: ValidatorComparison
    structural_criterion: StructuralCriterion
    owner_review: OwnerReview
    judges: Literal["diagnostic_only"] = "diagnostic_only"
    iteration: Iteration
    prior_exposure: str = Field(min_length=1)


class StaleEvidenceError(ValueError):
    """Bound evidence changed since it was recorded; every reason is kept."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class Evaluation:
    """Identities a registration resolves to, and every problem blocking it."""

    frozen: Record = field(default_factory=dict[str, Any])
    problems: list[str] = field(default_factory=list[str])
    files: set[Path] = field(default_factory=set[Path])


def relative(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        # Another drive cannot be expressed relative to the base.
        return path.as_posix()


def _rebased(registration: GERegistration, source: Path, target: Path) -> GERegistration:
    """The registration with its paths relative to `target` instead of `source`."""

    def move(name: str) -> str:
        return relative((source / name).resolve(), target)

    def maybe(name: str | None) -> str | None:
        return None if name is None else move(name)

    reference = registration.reference.model_copy(
        update={
            "c_selection": maybe(registration.reference.c_selection),
            "gd_decision": maybe(registration.reference.gd_decision),
            "gd_batch": maybe(registration.reference.gd_batch),
            "gd_realisation": [move(name) for name in registration.reference.gd_realisation],
        }
    )
    protected = [
        population.model_copy(update={"path": move(population.path)})
        for population in registration.protected_populations
    ]
    iteration = registration.iteration.model_copy(
        update={"previous_protocol": maybe(registration.iteration.previous_protocol)}
    )
    return registration.model_copy(
        update={
            "config": move(registration.config),
            "reference": reference,
            "g3_population": maybe(registration.g3_population),
            "protected_populations": protected,
            "iteration": iteration,
        }
    )


def load_registration(path: Path) -> GERegistration:
    return GERegistration.model_validate_json(path.read_text(encoding="utf-8"))


def g3_decision(checkpoint: Record) -> tuple[Record | None, str | None]:
    """The current replay of the checkpoint's G3 decision, or why there is none."""
    config = checkpoint["binding"]["config"]
    if config["test_only"]:
        return None, "test-only checkpoint"
    protocol = config.get("g3_protocol")
    if protocol is None:
        return None, "the checkpoint binds no G3 protocol"
    try:
        return g3.assess(Path(protocol)), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"G3 decision cannot be replayed: {error}"


def g3_authority(checkpoint: Record, decision: Record | None, reason: str | None = None) -> Record:
    """Whether a genuine passed G3 decision, replayed now, authorises real E experiments.

    Only the replayed decision counts: a closed issue, a synthetic (test-only)
    criteria pass or a checkpoint's own claim cannot authorise E.
    """
    binding = checkpoint["binding"]
    recorded = binding.get("scientific_evidence", {})
    if binding["config"]["test_only"]:
        return {
            "status": "test-only",
            "reason": "test-only checkpoint: software mechanics only",
            "decision_id": None,
        }
    blocked = {"status": "blocked", "decision_id": recorded.get("decision_id")}
    if decision is None:
        return blocked | {"reason": f"genuine passed G3 evidence is missing: {reason}"}
    if decision.get("test_only"):
        return blocked | {
            "reason": "synthetic (test-only) G3 evidence cannot authorise real E experiments"
        }
    if decision.get("outcome") != "passed":
        return blocked | {
            "reason": f"G3 outcome is {decision.get('outcome')}; only a genuine pass authorises E"
        }
    if recorded.get("status") != "passed" or recorded.get("decision_id") != decision.get("id"):
        return blocked | {"reason": "the checkpoint's recorded G3 decision no longer replays"}
    return {
        "status": "genuine",
        "reason": "a genuine passed G3 decision replays and matches the checkpoint",
        "decision_id": decision["id"],
        "g3_protocol_id": decision.get("protocol_id"),
    }


def d_reference_problems(
    spec: ConditionSpec,
    roles: dict[str, ModelBinding],
    baseline: Record,
    prompts: dict[str, str],
) -> list[str]:
    """The evidence report's planner contract applied to a configured D condition."""
    return evidence_report.planner_contract_problems(
        pipeline=spec.pipeline,
        params=spec.params,
        models={role: binding.model_dump(mode="json") for role, binding in roles.items()},
        prompts=prompts,
        baseline=baseline,
    )


def _workflow(
    registration: GERegistration, loaded: LoadedConfig
) -> tuple[ConditionSpec, ConditionSpec]:
    config, stories = loaded.config, loaded.config.stories
    if stories is None or loaded.briefs_dir is None:
        raise ValueError("G-E binds a story run configuration")
    if config.calls or config.judges is not None or config.plans is not None:
        raise ValueError("registered G-E generation cannot include other calls, judges or plans")
    reference = registration.reference.condition
    expected = {CANDIDATE: "candidate_e", reference: REFERENCE_PIPELINES[reference]}
    if {spec.name: spec.pipeline for spec in stories.conditions} != expected:
        raise ValueError(
            f"G-E requires exactly E candidate_e and {reference} {expected[reference]}"
        )
    matching = stories.equal_budget
    if (
        matching is None
        or set(matching.conditions) != set(expected)
        or matching.resource_policy is None
    ):
        raise ValueError(
            "G-E requires equal-budget matching of E and the reference under a "
            "registered resource policy"
        )
    if config.budget.allow_unpriced_models:
        raise ValueError("registered G-E generation cannot allow unpriced models")
    specs = {spec.name: spec for spec in stories.conditions}
    return specs[CANDIDATE], specs[reference]


def _checkpoint_path(params: ConditionalPlanningParams) -> Path:
    # Resolved exactly as the E pipeline resolves it at run time.
    return Path(params.checkpoint).resolve()


def _candidate(
    loaded: LoadedConfig,
    spec: ConditionSpec,
    params: ConditionalPlanningParams,
    checkpoint: Record,
    location: str,
) -> Record:
    binding = checkpoint["binding"]
    planner = loaded.config.roles.get("planner")
    if planner is None:
        raise ValueError("E's hybrid text completion requires a planner binding")
    return {
        "condition": CANDIDATE,
        "pipeline": spec.pipeline,
        "params": spec.params,
        "checkpoint": {
            "path": location,
            "id": checkpoint["id"],
            "file_sha256": file_sha(_checkpoint_path(params)),
            "parameter_sha256": checkpoint["parameter_sha256"],
            "step": checkpoint["step"],
            "test_only": binding["config"]["test_only"],
            "representation": binding["representation"],
            "layout_id": binding["layout_id"],
            "condition_id": binding["condition_id"],
            "dataset_id": binding["dataset_id"],
            "base_id": binding["base_id"],
            "training_config_sha256": hashlib.sha256(
                json.dumps(binding["config"], sort_keys=True).encode()
            ).hexdigest(),
        },
        "conditioning": {"condition_fields": params.condition_fields, "locks": params.locks},
        "guidance": {"scale": params.guidance_scale, "policy": binding["guidance_policy"]},
        "completion": {
            "role": "planner",
            "binding": planner.model_dump(mode="json"),
            "prompt_id": PROMPT_ID,
            "max_tokens": params.completion_max_tokens,
        },
    }


def _lineage(
    registration: GERegistration,
    base: Path,
    checkpoint: Record,
    authority: Record,
    result: Evaluation,
) -> Record:
    """The fine-tuning data and G3 population behind the checkpoint, rechecked now."""
    binding = checkpoint["binding"]
    dataset = Path(binding["config"]["dataset"])
    fine_tuning: Record
    try:
        loaded = load_dataset(dataset, optimizer=True)
        status = "verified" if loaded["id"] == binding["dataset_id"] else "changed"
        fine_tuning = {"status": status, "id": loaded["id"]}
        result.files.add(view_path(dataset))
    except (OSError, ValueError, KeyError) as error:
        fine_tuning = {"status": "unavailable", "reason": str(error)}
    if fine_tuning["status"] != "verified":
        result.problems.append(f"conditional fine-tuning data lineage is {fine_tuning['status']}")
    supplied: list[Record] = []
    location = None
    if registration.g3_population is not None:
        path = (base / registration.g3_population).resolve()
        supplied.append({"name": "g3", "id": load_view(path)["id"]})
        result.files.add(view_path(path))
        location = relative(path, base)
    population = evidence_report.g3_lineage(supplied, checkpoint) | {"path": location}
    if authority["status"] == "genuine" and population.get("status") != "verified":
        result.problems.append(
            f"the G3 training population is {population.get('status')}: "
            f"{population.get('reason', 'supply the G3 protocol training population')}"
        )
    return {"fine_tuning": fine_tuning, "g3_population": population}


def _brief(root: Path, brief_id: str) -> tuple[Path, Brief]:
    """A dev brief, read from the dev split only; held-out briefs are never opened."""
    path = root / "dev" / f"{brief_id}.yaml"
    if not path.is_file():
        raise ValueError(f"{brief_id} is not a dev brief; G-E never selects held-out briefs")
    brief = Brief.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if brief.id != brief_id or brief.split != "dev":
        raise ValueError(f"{path.name} does not hold dev brief {brief_id}")
    return path, brief


def _cohort(registration: GERegistration, loaded: LoadedConfig, result: Evaluation) -> Record:
    stories, root = loaded.config.stories, loaded.briefs_dir
    if stories is None or root is None:
        raise ValueError("G-E binds a story run configuration")
    selection = stories.briefs
    if selection.split not in {None, "dev"}:
        raise ValueError("G-E selects dev briefs only; the held-out cohort is never generated")
    if selection.split is None:
        # Without a split every split's briefs would be checked, the held-out ones included.
        raise ValueError("G-E configurations select `split: dev` (with optional ids)")
    ids = selection.ids or sorted(path.stem for path in (root / "dev").glob("*.yaml"))
    if (
        len(ids) != len(set(ids))
        or ids != registration.cohort.briefs
        or stories.repeat != registration.cohort.repeat
    ):
        raise ValueError("configuration must generate exactly the registered dev cohort and seeds")
    frozen = briefs.read_frozen(root).get("dev", {})
    checksums: dict[str, str] = {}
    for brief_id in ids:
        path, _ = _brief(root, brief_id)
        checksums[brief_id] = file_checksum(path)
        if frozen.get(brief_id) != checksums[brief_id]:
            raise ValueError(f"dev brief missing, changed or not frozen: {brief_id}")
        result.files.add(path)
    result.files.add(root / briefs.FROZEN)
    return {
        "split": "dev",
        "briefs": checksums,
        "split_sha256": briefs.split_checksum(root, "dev"),
        "repeat": stories.repeat,
        "run_seed": loaded.config.seed,
        "slots": [slot(brief_id, n) for brief_id in ids for n in range(stories.repeat)],
    }


def slot(brief: str, repetition: int) -> str:
    """A cohort slot: a brief and a one-based repetition, shared by E and the reference."""
    return f"{brief}/r{repetition + 1}"


def _resources(
    registration: GERegistration, loaded: LoadedConfig, params: ConditionalPlanningParams
) -> Record:
    config, stories = loaded.config, loaded.config.stories
    if stories is None or stories.equal_budget is None:
        raise ValueError("G-E requires equal-budget matching")
    current: Record = {
        "max_attempts": params.max_attempts,
        "max_tokens": params.max_tokens,
        "max_attention_cells": params.max_attention_cells,
        "max_seconds": params.max_seconds,
        "completion_max_tokens": params.completion_max_tokens,
        "max_repairs": config.structured.max_repairs,
        "max_length_attempts": stories.length.max_attempts,
        "max_revision_passes": stories.equal_budget.max_revision_passes,
        "budget_tolerance": stories.equal_budget.tolerance,
        "per_run_usd": config.budget.per_run_usd,
        "per_day_usd": config.budget.per_day_usd,
        "resource_policy": None
        if stories.equal_budget.resource_policy is None
        else stories.equal_budget.resource_policy.version,
    }
    registered = registration.resources.model_dump(mode="json")
    differ = [name for name, value in registered.items() if current[name] != value]
    if differ:
        raise ValueError(
            "registered resource limits differ from the configuration: " + ", ".join(differ)
        )
    return registered


def _replayed_gd(
    reference: ReferenceRegistration, base: Path, result: Evaluation
) -> tuple[Record, Record | None]:
    if reference.gd_decision is None or reference.gd_batch is None:
        raise ValueError("no G-D decision registered")
    folder = (base / reference.gd_decision).resolve()
    batch = (base / reference.gd_batch).resolve()
    packets = [(base / name).resolve() for name in reference.gd_realisation]
    saved = read_json(folder / "decision.json")
    # Rebuilt exactly as `nde gd-report` built it, and compared as saved JSON.
    replayed = json.loads(json.dumps(gd_report.decision(batch, packets)))
    baseline_file = folder / "reference-baseline.json"
    separate = read_json(baseline_file) if baseline_file.is_file() else None
    if replayed != saved or separate != saved.get("reference_baseline"):
        raise ValueError("the registered G-D decision does not replay from its closed batch")
    result.files.update(
        path for path in (folder / "decision.json", baseline_file) if path.is_file()
    )
    return saved, separate


def _reference(  # noqa: PLR0913 - one registration, its folder, config and prompt identities
    registration: GERegistration,
    base: Path,
    loaded: LoadedConfig,
    spec: ConditionSpec,
    *,
    prompts: dict[str, str],
    result: Evaluation,
) -> Record:
    reference = registration.reference
    record: Record = {
        "condition": reference.condition,
        "pipeline": spec.pipeline,
        "params": spec.params,
        "gd_decision": None,
        "c_selection": None,
    }
    permitted, reason = (
        "C",
        "no G-D decision registered; the identified C baseline is the reference",
    )
    baseline: Record | None = None
    if reference.gd_decision is not None:
        decision, baseline = _replayed_gd(reference, base, result)
        permitted, reason = evidence_report.gd_permission(decision)
        record["gd_decision"] = {
            "path": relative((base / reference.gd_decision).resolve(), base),
            "id": decision.get("id"),
            "outcome": decision.get("outcome"),
            "test_only": decision.get("test_only"),
        }
    if permitted != reference.condition:
        raise ValueError(f"the reference must be {permitted}: {reason}")
    record["reason"] = reason
    if reference.condition == "D":
        roles = {str(name): binding for name, binding in loaded.config.roles.items()}
        problems = d_reference_problems(spec, roles, baseline or {}, prompts)
        record["identification"] = "replayed actual G-D pass"
    else:
        path = (base / str(reference.c_selection)).resolve()
        selection = load_selection(path)
        result.files.add(path)
        problems = selected_c_problems(selection, loaded.path)
        record["identification"] = "owner C selection"
        record["c_selection"] = {
            "path": relative(path, base),
            "sha256": file_checksum(path),
            "id": selection.id,
            "tier": selection.tier,
            "test_only": selection.test_only,
        }
        if selection.test_only and not registration.test_only:
            result.problems.append("a test-only C selection cannot identify a real reference")
    if problems:
        raise ValueError("reference identity mismatch: " + "; ".join(problems))
    return record


def _protected(registration: GERegistration, base: Path, result: Evaluation) -> list[Record]:
    rows: list[Record] = []
    for population in registration.protected_populations:
        path = (base / population.path).resolve()
        view = load_view(path)
        if view.get("role") != "evaluator-only" and view.get("split") not in {"validation", "test"}:
            raise ValueError(f"protected population {population.name} is not evaluator-only")
        result.files.add(view_path(path))
        rows.append(
            {
                "name": population.name,
                "path": relative(path, base),
                "id": view["id"],
                "split": view.get("split"),
            }
        )
    return rows


def _identities(loaded: LoadedConfig) -> Record:
    templates: dict[str, PromptTemplate] = {}
    # Resolved without selecting briefs, so G-E never opens the held-out split.
    story_templates(loaded, story_conditions(loaded), templates)
    models = {
        name: binding.model_dump(mode="json") for name, binding in loaded.config.roles.items()
    }
    return {
        "config_sha256": loaded.hash,
        "models_sha256": digest(json.dumps(models, sort_keys=True)),
        "prompt_sha256": {label: template.sha256 for label, template in templates.items()},
        "schema_sha256": digest(schema_json()),
        "completion_prompt_id": PROMPT_ID,
    }


def owner_sample(slots: list[str], owner: OwnerReview) -> list[str]:
    """The frozen owner sample: a seeded order of the cohort slots, fixed before any output."""
    ranked = sorted(
        slots, key=lambda name: hashlib.sha256(f"{owner.sample_seed}:{name}".encode()).hexdigest()
    )
    return ranked[: owner.sample_size]


def _located(candidate: Record) -> Record:
    """E's identity without where its checkpoint file is found from a protocol."""
    return candidate | {"checkpoint": candidate["checkpoint"] | {"path": None}}


def _unlocated(reference: Record) -> Record:
    """The reference identity without where its evidence is found from a protocol."""
    return reference | {
        name: None if reference[name] is None else reference[name] | {"path": None}
        for name in ("c_selection", "gd_decision")
    }


def _iteration(
    registration: GERegistration, base: Path, frozen: Record, result: Evaluation
) -> Record:
    iteration = registration.iteration
    record: Record = {
        "policy": iteration.policy,
        "attempt": iteration.attempt,
        "change": iteration.change,
        "previous": None,
    }
    if iteration.previous_protocol is None:
        return record
    path = (base / iteration.previous_protocol).resolve()
    previous = read_protocol(path)
    result.files.add(path)
    record["previous"] = {"path": relative(path, base), "id": previous["id"]}
    earlier = GERegistration.model_validate(previous["registration"])
    if earlier.iteration.attempt != 1:
        raise ValueError("the development iteration must follow the first registered attempt")
    claim = read_claim(path)
    if claim is None or claim["protocol_id"] != previous["id"]:
        raise ValueError("the first attempt was never run; there is nothing to iterate on")
    result.files.add(claim_path(path))
    record["previous"]["claim_id"] = claim["id"]
    unchanged = (
        earlier.test_only == registration.test_only
        and earlier.validator_comparison == registration.validator_comparison
        and earlier.structural_criterion == registration.structural_criterion
        and earlier.owner_review == registration.owner_review
        # The whole cohort: briefs and their checksums, split, repeat, run seed and slots.
        and previous["frozen"]["cohort"] == frozen["cohort"]
        and _unlocated(previous["frozen"]["reference"]) == _unlocated(frozen["reference"])
    )
    if not unchanged:
        raise ValueError(
            "the development iteration keeps the first attempt's cohort, reference, "
            "criteria and owner review"
        )
    if _located(previous["frozen"]["candidate"]) == _located(frozen["candidate"]):
        raise ValueError("the development iteration must change E; an identical rerun is not one")
    return record


def _evaluate(registration: GERegistration, base: Path, result: Evaluation) -> None:
    frozen = result.frozen
    config = (base / registration.config).resolve()
    loaded = load_config(config)
    result.files.add(config)
    spec, reference = _workflow(registration, loaded)
    params = ConditionalPlanningParams.model_validate(spec.params)
    checkpoint_file = _checkpoint_path(params)
    checkpoint = inspect_checkpoint(checkpoint_file)
    result.files.add(checkpoint_file)
    frozen["g3"] = authority = g3_authority(checkpoint, *g3_decision(checkpoint))
    if authority["status"] == "blocked":
        result.problems.append(authority["reason"])
    elif authority["status"] == "test-only" and not registration.test_only:
        result.problems.append("a test-only checkpoint cannot authorise a real G-E protocol")
    try:
        inspect_checkpoint(checkpoint_file, validate_inputs=True)
    except (OSError, ValueError, KeyError) as error:
        result.problems.append(f"checkpoint lineage changed: {error}")
    frozen["candidate"] = _candidate(
        loaded, spec, params, checkpoint, relative(checkpoint_file, base)
    )
    frozen["lineage"] = _lineage(registration, base, checkpoint, authority, result)
    frozen["cohort"] = _cohort(registration, loaded, result)
    frozen["resources"] = _resources(registration, loaded, params)
    frozen["identities"] = _identities(loaded)
    prompts = frozen["identities"]["prompt_sha256"]
    frozen["reference"] = _reference(
        registration, base, loaded, reference, prompts=prompts, result=result
    )
    frozen["protected_populations"] = _protected(registration, base, result)
    slots = frozen["cohort"]["slots"]
    if registration.owner_review.sample_size > len(slots):
        raise ValueError("owner sample is larger than the registered dev cohort")
    frozen["owner_sample"] = owner_sample(slots, registration.owner_review)
    frozen["iteration"] = _iteration(registration, base, frozen, result)
    if registration.test_only and any(
        urlparse(provider.base_url).hostname not in LOCAL_HOSTS
        for provider in loaded.config.providers.values()
    ):
        result.problems.append("test-only G-E is restricted to local fake providers")


def evaluate(registration: GERegistration, base: Path) -> Evaluation:
    """Resolve a registration's identities without spending; never raises for bad evidence."""
    result = Evaluation()
    try:
        _evaluate(registration, base, result)
    except EVALUATION_ERRORS as error:
        result.problems.append(str(error))
    result.frozen["inputs"] = {
        relative(path, base): file_sha(path) for path in sorted(result.files) if path.is_file()
    }
    return result


def readiness(path: Path) -> Record:
    """The no-spend readiness of a registration: G3 authority, lineage and identities."""
    registration = load_registration(path)
    result = evaluate(registration, path.parent.resolve())
    return {
        "kind": "ge-readiness",
        "readiness_version": "1",
        "status": "blocked" if result.problems else "ready",
        "test_only": registration.test_only,
        "authority": result.frozen.get("g3"),
        "lineage": result.frozen.get("lineage"),
        "problems": result.problems,
        "frozen": result.frozen,
    }


def register(path: Path, out: Path) -> Record:
    """Freeze a ready registration as an immutable protocol beside `out`; commit it next."""
    if out.exists():
        raise ValueError("protocol already exists; registrations are immutable")
    folder = out.parent.resolve()
    registration = _rebased(load_registration(path), path.parent.resolve(), folder)
    result = evaluate(registration, folder)
    if result.problems:
        raise ValueError("G-E registration blocked: " + "; ".join(result.problems))
    sample = result.frozen["owner_sample"]
    owner = registration.owner_review
    return create_record(
        out,
        {
            "kind": PROTOCOL_KIND,
            "protocol_version": PROTOCOL_VERSION,
            "name": registration.id,
            "test_only": registration.test_only,
            "scope": SCOPE,
            "registration": registration.model_dump(mode="json"),
            "frozen": result.frozen,
            "criteria": {
                "validator_comparison": registration.validator_comparison.model_dump(mode="json"),
                "structural_criterion": registration.structural_criterion.model_dump(mode="json"),
                "judges": registration.judges,
            },
            "owner_review": owner.model_dump(mode="json")
            | {"sample": sample, "estimated_minutes": len(sample) * owner.minutes_per_item},
            "stopping_rule": "fixed_dev_cohort_no_top_up",
            "fail_action": FAIL_ACTION,
            "held_out": HELD_OUT,
        },
    )


def read_protocol(path: Path) -> Record:
    protocol = read_sealed(path)
    if protocol.get("kind") != PROTOCOL_KIND or protocol.get("protocol_version") != (
        PROTOCOL_VERSION
    ):
        raise ValueError(f"not a version {PROTOCOL_VERSION} G-E protocol: {path}")
    GERegistration.model_validate(protocol["registration"])
    return protocol


def committed(path: Path) -> str:
    """The commit holding the protocol exactly as it is now."""
    root_name = git(["rev-parse", "--show-toplevel"], path.parent)
    if root_name is None:
        raise ValueError("G-E pre-registration must belong to a Git repository")
    root = Path(root_name).resolve()
    relative = path.resolve().relative_to(root).as_posix()
    if git(["ls-files", "--error-unmatch", "--", relative], root) is None:
        raise ValueError("G-E pre-registration is uncommitted")
    if git(["diff", "--quiet", "HEAD", "--", relative], root) is None:
        raise ValueError("G-E pre-registration differs from its committed contents")
    revision = git(["rev-parse", "HEAD"], root)
    if revision is None:
        raise ValueError("G-E pre-registration has no committed revision")
    return revision


def check(path: Path) -> Record:
    """Revalidate a committed protocol against current evidence, with no provider call.

    Raises ValueError only when the file is not a readable G-E protocol.
    """
    protocol = read_protocol(path)
    result: Record = {
        "kind": "ge-check",
        "check_version": "1",
        "status": "blocked",
        "test_only": protocol["test_only"],
        "protocol_id": protocol["id"],
        "protocol_sha256": file_sha(path),
        "protocol_commit": None,
        "authority": None,
        "problems": [],
    }
    problems: list[str] = result["problems"]
    try:
        result["protocol_commit"] = committed(path)
    except ValueError as error:
        problems.append(str(error))
    current = evaluate(
        GERegistration.model_validate(protocol["registration"]), path.parent.resolve()
    )
    result["authority"] = current.frozen.get("g3")
    problems += current.problems
    saved: Record = protocol["frozen"]
    problems += [
        f"registered input changed or missing: {name}"
        for name, checksum in sorted(saved["inputs"].items())
        if current.frozen["inputs"].get(name) != checksum
    ]
    problems += [
        f"registered {section} differs from current evidence"
        for section in sorted(saved.keys() - {"inputs"})
        if current.frozen.get(section) != saved[section]
    ]
    if not problems:
        result["status"] = "ready"
    return result


def require(path: Path) -> Record:
    checked = check(path)
    if checked["status"] != "ready":
        raise ValueError("G-E prerequisites blocked: " + "; ".join(checked["problems"]))
    return checked


def _markdown_lines(title: str, record: Record) -> list[str]:
    authority: Record = record.get("authority") or {}
    return [
        f"# {title}: {record['status']}",
        "",
        f"Test only: {record['test_only']}",
        "",
        f"G3 authority: {authority.get('status')} - {authority.get('reason')}",
        "",
        "## Problems",
        "",
        *(f"- {problem}" for problem in record["problems"] or ["None"]),
        "",
    ]


def write_status(record: Record, out: Path, name: str, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_text_lf(out / f"{name}.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    write_text_lf(out / f"{name}.md", "\n".join(_markdown_lines(title, record)))


# Registered run: execute the bound workflow and record its complete evidence.


def _evidence_request(protocol: Record, protocol_path: Path, run: Path) -> Path:
    """The #83 evidence report request over the run's E stories and reference outputs."""
    folder = run / EVIDENCE
    folder.mkdir(parents=True, exist_ok=True)
    base = protocol_path.parent.resolve()
    registration = GERegistration.model_validate(protocol["registration"])
    reference = registration.reference

    def here(name: str) -> str:
        return relative((base / name).resolve(), folder)

    request: Record = {
        "story_runs": [{"run": relative(run, folder), "label": CANDIDATE_LABEL}],
        "reference": {
            "runs": [{"run": relative(run, folder), "condition": reference.condition}],
            "c_selection": None if reference.c_selection is None else here(reference.c_selection),
            "gd_decision": None if reference.gd_decision is None else here(reference.gd_decision),
            "gd_batch": None if reference.gd_batch is None else here(reference.gd_batch),
            "gd_realisation": [here(name) for name in reference.gd_realisation],
        },
        "training_populations": []
        if registration.g3_population is None
        else [{"name": "g3", "path": here(registration.g3_population), "role": "g3"}],
        "protected_populations": [
            {"name": population.name, "path": here(population.path)}
            for population in registration.protected_populations
        ],
        "neighbours": registration.neighbours,
    }
    path = folder / "request.json"
    write_text_lf(path, json.dumps(request, indent=2, sort_keys=True) + "\n")
    return path


def _candidate_units(report: Record, slots: list[str]) -> list[Record]:
    units = {unit["unit"]: unit for unit in report["units"]}
    validity = {
        row["unit"]: row["status"]
        for row in report["adherence"]["deterministic"]["rows"]
        if row["predicate"] == "validity"
    }
    rows: list[Record] = []
    for name in slots:
        unit_id = f"story/{CANDIDATE_LABEL}/{name}"
        unit = units.get(unit_id, {})
        rows.append(
            {
                "condition": CANDIDATE,
                "slot": name,
                "evidence_unit": unit_id,
                "status": unit.get("status", "missing"),
                "seed": unit.get("seed"),
                "path": unit.get("path"),
                "plan_sha256": unit.get("plan_sha256"),
                "final_sha256": unit.get("output_sha256"),
                "validity": validity.get(unit_id, "unknown"),
            }
        )
    return rows


def _run_unit(run: Path, condition: str, name: str, record: StoryRecord | None) -> Record:
    row: Record = {
        "condition": condition,
        "slot": name,
        "evidence_unit": None,
        "status": "missing" if record is None else record.status,
        "seed": None if record is None else record.seed,
        "path": None if record is None else record.path,
        "plan_sha256": None,
        "final_sha256": None,
        "validity": "unknown",
    }
    if record is None:
        return row
    plan_file = run / record.path / "plan.json"
    if not plan_file.is_file():
        # Planning failed: no output, which is not an invalid plan; a success without
        # its plan is unreadable evidence.
        row["validity"] = "no-output" if record.status != "ok" else "unknown"
        row["status"] = record.status if record.status != "ok" else "unreadable"
        return row
    try:
        plan: NarrativePlan = load_plan(plan_file)
    except (OSError, PlanError) as error:
        row.update(status="unreadable", error=str(error))
        return row
    row.update(
        plan_sha256=plan_sha256(plan),
        final_sha256=record.sha256 if record.status == "ok" else None,
        validity="success" if validate_plan(plan).valid else "failure",
    )
    return row


def _side(rows: list[Record], aggregation: str) -> Record:
    counts = Counter(str(row["validity"]) for row in rows)
    by_brief: dict[str, Record] = {}
    for row in rows:
        brief = row["slot"].split("/")[0]
        entry = by_brief.setdefault(brief, {"expected": 0, "valid": 0})
        entry["expected"] += 1
        entry["valid"] += row["validity"] == "success"
    for entry in by_brief.values():
        entry["rate"] = entry["valid"] / entry["expected"]
    pooled = counts["success"] / len(rows) if rows else None
    macro = sum(entry["rate"] for entry in by_brief.values()) / len(by_brief) if by_brief else None
    return {
        "counts": {"expected": len(rows), **{status: counts[status] for status in VALIDITY}},
        "by_brief": by_brief,
        "pooled_rate": pooled,
        "brief_macro_rate": macro,
        "rate": pooled if aggregation == "pooled" else macro,
    }


def _validator(units: list[Record], protocol: Record) -> Record:
    rule = protocol["criteria"]["validator_comparison"]
    reference = protocol["frozen"]["reference"]["condition"]
    candidate = _side([u for u in units if u["condition"] == CANDIDATE], rule["aggregation"])
    baseline = _side([u for u in units if u["condition"] == reference], rule["aggregation"])
    unknown = [f"{u['condition']}/{u['slot']}" for u in units if u["validity"] == "unknown"]
    difference = (
        None
        if candidate["rate"] is None or baseline["rate"] is None
        else candidate["rate"] - baseline["rate"]
    )
    return {
        "metric": rule["metric"],
        "aggregation": rule["aggregation"],
        "denominator": "every expected slot; no-output counts as not valid; unknown is pending",
        "candidate": candidate,
        "reference": baseline,
        "difference": difference,
        "complete": not unknown,
        "unknown_units": unknown,
    }


def _local(work: list[LocalWorkRecord]) -> Record:
    return {
        "operations": len(work),
        "passes": sum(item.passes for item in work),
        "tokens": sum(item.tokens for item in work),
        "attention_cells": sum(item.attention_cells for item in work),
        "elapsed_seconds": sum(item.elapsed_seconds for item in work),
    }


def _costs(
    manifest: Manifest, calls: list[CallRecord], work: list[LocalWorkRecord], names: list[str]
) -> Record:
    matching = None if manifest.stories is None else manifest.stories.equal_budget
    matches = [] if matching is None else matching.matches
    return {
        "rule": "Provider calls and local model work keep their own units; never converted or "
        "summed. Matched pairs are matched on provider generation tokens only.",
        "run": manifest.cost.model_dump(mode="json"),
        "conditions": {
            name: {
                "provider": provider_categories([c for c in calls if c.condition == name]),
                "local": _local([w for w in work if w.condition == name]),
            }
            for name in names
        },
        "unattributed_calls": sum(call.condition not in names for call in calls),
        "equal_budget": {
            "pairs": len(matches),
            "matched": sum(match.matched for match in matches),
            "unresolved": [
                f"{match.brief}/r{match.repetition + 1}: {match.reason}"
                for match in matches
                if not match.matched
            ],
        },
    }


def _evidence(report: Record) -> Record:
    training = report["training_similarity"]
    reference = report["reference"]
    return {
        "status": "available",
        "report": {
            "path": f"{EVIDENCE}/report",
            "id": report["id"],
            "report_version": report["report_version"],
            "test_only": report["test_only"],
            "consistent": report["consistent"],
            "problems": report["problems"],
        },
        "adherence": {
            "deterministic": {
                "by_predicate": report["adherence"]["deterministic"]["by_predicate"],
                "total": report["adherence"]["deterministic"]["total"],
            },
            "judgement": {
                "by_condition": report["adherence"]["judgement"]["by_condition"],
                "total": report["adherence"]["judgement"]["total"],
            },
        },
        "diversity": [
            {"group": group["group"], "summary": group["summary"]}
            for group in report["diversity"]["groups"]
        ],
        "overlap": {
            "lineage": training["lineage"],
            "g3_population": training["g3_population"]["status"],
            "populations": [
                {"name": row["name"], "status": row["status"], "summary": row.get("summary")}
                for row in training["populations"]
            ],
            "protected_overlap": training["protected_overlap"],
            "limitations": training["limitations"],
        },
        "reference": {
            "permitted": reference["permitted"],
            "identified": reference["identified"],
            "status": reference["status"],
            "identity_problems": reference["identity_problems"],
            "summary": reference["summary"],
        },
        "realisation": report["realisation"]["summary"],
        "pending": report["pending"],
    }


def _unverified(row: Record) -> Record:
    """An E unit without its evidence report: a saved plan cannot be verified, so unknown."""
    return row | {"validity": "unknown"} if row["validity"] in {"success", "failure"} else row


def measure(protocol: Record, run: Path, report: Record | None, error: str | None = None) -> Record:
    """Everything a registered run measured, recomputable from the saved run and report.

    Without a report (`error` says why) the run's costs, failures and denominators are
    still recorded, E's plans stay unverified and the evidence section is unavailable.
    """
    manifest = read_manifest(run)
    records = read_stories(run)
    reference = protocol["frozen"]["reference"]["condition"]
    slots: list[str] = protocol["frozen"]["cohort"]["slots"]
    saved = {(r.condition, slot(r.brief, r.repetition)): r for r in records}
    if report is None:
        units = [
            _unverified(_run_unit(run, CANDIDATE, name, saved.get((CANDIDATE, name))))
            for name in slots
        ]
    else:
        units = _candidate_units(report, slots)
    units += [_run_unit(run, reference, name, saved.get((reference, name))) for name in slots]
    errors = {(r.condition, slot(r.brief, r.repetition)): r.error for r in records}
    problems = (
        [f"evidence report unavailable: {error}"] if report is None else list(report["problems"])
    )
    problems += [
        f"unregistered output: {r.condition}/{slot(r.brief, r.repetition)}"
        for r in records
        if r.condition not in {CANDIDATE, reference} or slot(r.brief, r.repetition) not in slots
    ]
    if manifest.config_hash != protocol["frozen"]["identities"]["config_sha256"]:
        problems.append("run configuration differs from registration")
    work = read_local_work(run) if (run / "local-work.jsonl").is_file() else []
    return {
        "run": {
            "status": manifest.status,
            "abort_reason": manifest.abort_reason,
            "config_sha256": manifest.config_hash,
            "manifest_sha256": file_sha(run / "manifest.json"),
        },
        "units": units,
        "validator": _validator(units, protocol),
        "failures": [
            {
                "condition": unit["condition"],
                "slot": unit["slot"],
                "status": unit["status"],
                "error": errors.get((unit["condition"], unit["slot"])),
            }
            for unit in units
            if unit["status"] != "ok"
        ],
        "costs": _costs(manifest, read_calls(run), work, [CANDIDATE, reference]),
        "evidence": {"status": "unavailable", "error": error}
        if report is None
        else _evidence(report),
        "owner_review": {
            "status": "pending",
            "sample": protocol["owner_review"]["sample"],
            "rule": "Assemble the owner's packet with `nde ge packet`; a missing review stays "
            "pending. Model judges are diagnostic only and never replace it.",
        },
        "problems": problems,
    }


def artifacts(run: Path) -> dict[str, str]:
    """Checksums of every run and evidence file the G-E record binds."""
    names = {
        "manifest.json",
        "config.json",
        "stories.jsonl",
        "calls.jsonl",
        "local-work.jsonl",
        "comparison.json",
    }
    names |= {
        (Path(record.path) / name).as_posix()
        for record in read_stories(run)
        for name in record.files
    }
    names |= {path.relative_to(run).as_posix() for path in (run / EVIDENCE).rglob("*.*")}
    return {name: file_sha(run / name) for name in sorted(names) if (run / name).is_file()}


def _evidence_lines(evidence: Record) -> list[str]:
    if evidence["status"] != "available":
        return [f"Evidence report unavailable: {evidence['error']}"]
    return [
        f"Evidence report `{evidence['report']['id']}`; reference "
        f"{evidence['reference']['permitted']} identified: "
        f"{evidence['reference']['identified']}; protected overlap: "
        f"{evidence['overlap']['protected_overlap']}.",
        "",
        *(f"- {item}" for item in evidence["pending"]),
    ]


def _run_markdown(record: Record) -> str:
    validator = record["validator"]
    lines = [
        f"# G-E development run: {record['name']}"
        + (" (test-only)" if record["test_only"] else ""),
        "",
        record["statement"],
        "",
        f"Protocol `{record['protocol']['id']}` at commit `{record['protocol']['commit']}`; "
        f"run status {record['run']['status']}.",
        "",
        "## Validator pass rate (measured, not judged)",
        "",
        f"Aggregation: {validator['aggregation']}. {validator['denominator']}.",
        "",
        "| Condition | Expected | Valid | Invalid | No output | Unknown | Rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {name} | {s['counts']['expected']} | {s['counts']['success']} | "
            f"{s['counts']['failure']} | {s['counts']['no-output']} | {s['counts']['unknown']} "
            f"| {s['rate']} |"
            for name, s in (("E", validator["candidate"]), ("reference", validator["reference"]))
        ),
        "",
        f"Difference (E minus reference): {validator['difference']}; complete: "
        f"{validator['complete']}.",
        "",
        "## Failures",
        "",
        *(
            f"- {f['condition']} {f['slot']}: {f['status']} {f['error'] or ''}".rstrip()
            for f in record["failures"]
        ),
        *([] if record["failures"] else ["None"]),
        "",
        "## Costs",
        "",
        record["costs"]["rule"],
        "",
        f"New spend ${record['costs']['run']['usd']:.6f}; list price "
        f"${record['costs']['run']['token_cost_usd']:.6f}.",
        *(
            f"- {name}: {sum(c['calls'] for c in cost['provider'].values())} provider calls, "
            f"{cost['local']['passes']} local passes, {cost['local']['tokens']} local tokens"
            for name, cost in record["costs"]["conditions"].items()
        ),
        "",
        "## Evidence",
        "",
        *_evidence_lines(record["evidence"]),
        "",
        "## Owner structural review",
        "",
        f"Status: {record['owner_review']['status']}. {record['owner_review']['rule']}",
        "",
        "## Problems",
        "",
        *(f"- {problem}" for problem in record["problems"] or ["None"]),
        "",
    ]
    return "\n".join(lines)


def claim_path(protocol: Path) -> Path:
    """Where a protocol's run claim is kept: beside the protocol, never in a runs folder."""
    return protocol.with_name(f"{protocol.stem}.run-claim.json")


def read_claim(protocol: Path) -> Record | None:
    """The protocol's sealed run claim, or None when it has never been run."""
    path = claim_path(protocol)
    if not path.exists():
        return None
    claim = read_sealed(path)
    if claim.get("kind") != CLAIM_KIND:
        raise ValueError(f"not a G-E run claim: {path}")
    return claim


def _claim(path: Path, protocol: Record, options: RunOptions) -> Record:
    """Claim the protocol's one run before generating; exclusive creation refuses reruns."""
    try:
        return create_record(
            claim_path(path),
            {
                "kind": CLAIM_KIND,
                "protocol_id": protocol["id"],
                "runs_dir": str(options.runs_dir.resolve()),
                "claimed_at": options.clock().isoformat(),
            },
        )
    except FileExistsError as error:
        raise ValueError(ALREADY_RUN) from error


def run(path: Path, options: RunOptions) -> Path:
    """Execute the bound dev workflow after a ready check and record its evidence.

    The run is claimed beside the protocol before any generation, so it happens at
    most once wherever its outputs go. Once generation has happened the G-E record
    is always written, with the evidence marked unavailable if its report fails.
    """
    checked = require(path)
    protocol = read_protocol(path)
    if read_claim(path) is not None:
        raise ValueError(ALREADY_RUN)
    config = (path.parent / protocol["registration"]["config"]).resolve()
    # Validated without spending before the claim, so a configuration error burns nothing.
    prepared = prepare(config, options.environ)
    claim = _claim(path, protocol, options)
    folder = execute(prepared, options)
    report: Record | None = None
    error: str | None = None
    try:
        request = _evidence_request(protocol, path, folder)
        report = evidence_report.write(request, folder / EVIDENCE / "report")
    except (OSError, ValueError, KeyError, TypeError) as failure:
        error = str(failure) or type(failure).__name__
    record = create_record(
        folder / RUN_RECORD,
        {
            "kind": RUN_KIND,
            "run_version": RUN_VERSION,
            "name": protocol["name"],
            "test_only": bool(protocol["test_only"] or report is None or report["test_only"]),
            "statement": NOT_A_DECISION,
            "protocol": {
                "path": relative(path.resolve(), folder),
                "id": protocol["id"],
                "sha256": checked["protocol_sha256"],
                "commit": checked["protocol_commit"],
            },
            "claim": {"path": relative(claim_path(path).resolve(), folder), "id": claim["id"]},
            "judges": "diagnostic_only",
            **measure(protocol, folder, report, error),
            "artifacts": artifacts(folder),
        },
    )
    write_text_lf(folder / "ge-run.md", _run_markdown(record))
    return folder


def read_run(run: Path) -> Record:
    record = read_sealed(run / RUN_RECORD)
    if record.get("kind") != RUN_KIND or record.get("run_version") != RUN_VERSION:
        raise ValueError(f"not a version {RUN_VERSION} G-E run record: {run}")
    return record


def artifact_problems(run: Path, record: Record) -> list[str]:
    """Bound run artifacts that are missing or changed since the run was recorded."""
    current = artifacts(run)
    return [
        f"run artifact changed or missing: {name}"
        for name in sorted(record["artifacts"].keys() | current.keys())
        if record["artifacts"].get(name) != current.get(name)
    ]


def verify(path: Path, run: Path) -> list[str]:
    """Why a recorded run no longer holds against its protocol and evidence, if at all."""
    reasons = [f"protocol: {problem}" for problem in check(path)["problems"]]
    protocol = read_protocol(path)
    try:
        record = read_run(run)
    except (OSError, ValueError) as error:
        return [*reasons, f"G-E run record missing or changed: {error}"]
    if record["protocol"]["id"] != protocol["id"]:
        reasons.append("run is bound to another G-E protocol")
    claim = read_claim(path)
    if claim is None or claim["id"] != record["claim"]["id"]:
        reasons.append("run is not the protocol's claimed run")
    reasons += artifact_problems(run, record)
    folder = run / EVIDENCE / "report"
    try:
        if record["evidence"]["status"] != "available":
            # Never current: this run's evidence could not be recorded.
            error = record["evidence"]["error"]
            reasons.append(f"evidence report unavailable: {error}")
            current = measure(protocol, run, None, error)
        else:
            _, changed = evidence_report.check(folder)
            reasons += [f"evidence report: {reason}" for reason in changed]
            current = measure(protocol, run, read_sealed(folder / evidence_report.REPORT))
    except (OSError, ValueError, KeyError) as error:
        return [*reasons, f"run evidence can no longer be measured: {error}"]
    reasons += [
        f"recorded {section} differs from the current run"
        for section in sorted(current)
        if record.get(section) != current[section]
    ]
    return reasons
