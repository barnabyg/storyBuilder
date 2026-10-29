"""Candidate E plans: learned structure from a conditional checkpoint, then hybrid text.

E reuses the established workflows rather than re-implementing them: full
generation (`nde conditional-generate`) proposes the structure on the local
model, and restricted text completion (`nde conditional-complete`) fills the
proposal's text manifest on the run's own gateway, under its cache, prices and
caps. Both keep their durable records in the story's `conditional/` folder. The
local model's work is recorded as a `LocalWorkRecord`, never as provider calls.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from nde.conditional.completion import (
    PROMPT_ID,
    CompletionRequest,
    complete_with_gateway,
)
from nde.conditional.data import Conditions
from nde.conditional.generation import Generation, generate, scientific_authority
from nde.conditional.training import GUIDANCE_POLICY, candidate_cost, inspect_checkpoint
from nde.corpus.records_io import read_json, read_sealed
from nde.files import write_text_lf
from nde.gitinfo import git
from nde.llm.config import ModelBinding
from nde.narrative.base import Strict
from nde.narrative.ledger import plan_identity
from nde.narrative.models import NarrativePlan
from nde.narrative.workflow import PlanError, load_plan, rendered_files, schema_json
from nde.plan.revision import identity
from nde.runs.records import CallRecord, LocalAttempt, LocalWorkRecord
from nde.schema.brief import Brief
from nde.stories.pipeline import StepFailedError, StoryContext, derive_seed
from nde.stories.resources import COMPLETION_STEP

type ConditionField = Literal["genre", "themes", "ending_intent", "target_words", "scene_range"]

CONDITION_FIELDS: tuple[ConditionField, ...] = (
    "genre",
    "themes",
    "ending_intent",
    "target_words",
    "scene_range",
)
GENERATION_STEP = "conditional-generation"
# The story's folder for E's durable generation and completion records.
CONDITIONAL_DIR = "conditional"
PIPELINE = "candidate_e"
LITERARY_ADHERENCE = "unassessed; mechanical validation only, no literary or owner judgement"


class ConditionalPlanningParams(Strict):
    """E's registered planning settings: checkpoint, conditions, guidance, locks and limits."""

    checkpoint: str = Field(min_length=1)
    # The brief attributes passed to the model as conditions; the rest stay unknown.
    condition_fields: list[ConditionField] = Field(default_factory=lambda: list(CONDITION_FIELDS))
    guidance_scale: float = Field(default=1, ge=0, le=8, allow_inf_nan=False)
    locks: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    max_attempts: int = Field(default=3, ge=1, le=100)
    max_tokens: int = Field(default=1000000, ge=1)
    max_attention_cells: int = Field(default=100000000, ge=1)
    max_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    completion_max_tokens: int = Field(default=1024, ge=1, le=8192)

    @model_validator(mode="after")
    def _authority(self) -> Self:
        if len(set(self.condition_fields)) != len(self.condition_fields):
            raise ValueError("condition_fields must be distinct")
        # The generation workflow's own lock authority rules, checked before any run.
        Generation(checkpoint=Path(self.checkpoint), locks=self.locks)
        return self


class EPlanningProvenance(Strict):
    """Everything an E plan was made from and every unit of work it used."""

    version: Literal["1"]
    pipeline: Literal["candidate_e"]
    status: Literal["succeeded", "failed"]
    brief: Brief
    brief_sha256: str
    schema_sha256: str
    git_sha: str
    params: ConditionalPlanningParams
    # The planner binding that completed the learned proposal's text.
    model: ModelBinding
    seed: int
    max_repairs: int
    completion_prompt_id: str
    conditions: dict[str, JsonValue] | None
    checkpoint: dict[str, Any] | None
    generation_request: dict[str, Any] | None
    generation: dict[str, Any] | None
    proposal: dict[str, Any] | None
    completion: dict[str, Any] | None
    local_work: list[LocalWorkRecord]
    local_work_sha256: str
    calls: list[CallRecord]
    calls_sha256: str
    plan_sha256: str | None
    scientific_status: str
    literary_adherence: str


def brief_conditions(brief: Brief, fields: list[ConditionField]) -> Conditions:
    """The brief's requested attributes as model conditions; unselected ones stay unknown."""
    values: dict[str, object] = {
        "genre": brief.genre,
        "themes": brief.themes,
        "ending_intent": (
            f"{brief.ending_constraint.kind.value}: {brief.ending_constraint.detail}"
        ),
        "target_words": (brief.target_words, brief.target_words),
        "scene_range": (brief.min_scenes, brief.max_scenes),
    }
    return Conditions.model_validate({field: values[field] for field in fields})


def generation_seed(seed: int) -> int:
    return derive_seed(seed, "plan", GENERATION_STEP)


def completion_seed(seed: int) -> int:
    return derive_seed(seed, "plan", COMPLETION_STEP)


def checkpoint_summary(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint identities that bind training, conditioning and guidance."""
    binding = checkpoint["binding"]
    evidence = binding["scientific_evidence"]
    return {
        "id": checkpoint["id"],
        "path": str(path.resolve()),
        "test_only": binding["config"]["test_only"],
        "representation": binding["representation"],
        "layout_id": binding["layout_id"],
        "condition_id": binding["condition_id"],
        "guidance_policy": binding["guidance_policy"],
        "training_config_id": identity(binding["config"]),
        "dataset_id": binding["dataset_id"],
        "base_id": binding["base_id"],
        "step": checkpoint["step"],
        "conditioning_modes": sorted(set(checkpoint["conditioning_modes"])),
        "scientific_evidence": {
            "status": evidence["status"],
            "decision_id": evidence.get("decision_id"),
        },
    }


def checked_checkpoint(params: ConditionalPlanningParams) -> dict[str, Any]:
    """The checkpoint, if it can serve these settings; raises ValueError, OSError or KeyError."""
    checkpoint = inspect_checkpoint(Path(params.checkpoint))
    scientific_authority(checkpoint)
    candidate_cost(checkpoint, Conditions(), guidance_scale=params.guidance_scale)
    return checkpoint


def local_work(
    ctx: StoryContext, checkpoint: dict[str, Any], request: Generation, report: dict[str, Any]
) -> LocalWorkRecord:
    """The generation report's attempts and actual or reserved work, as one record."""
    attempts: list[LocalAttempt] = []
    passes = 0
    for row in report["attempts"]:
        candidate = row.get("candidate")
        if candidate is None:
            # An attempt interrupted inside the model keeps its full reservation.
            reserved = row["reserved_work"]
            attempts.append(
                LocalAttempt(
                    attempt=row["attempt"],
                    seed=row["seed"],
                    status=row["status"],
                    pass_modes=[],
                    tokens=reserved["tokens"],
                    attention_cells=reserved["attention_cells"],
                    elapsed_seconds=0.0,
                )
            )
            continue
        work = candidate["work"]
        passes += work["passes"]
        attempts.append(
            LocalAttempt(
                attempt=row["attempt"],
                seed=row["seed"],
                status=row["status"],
                pass_modes=sorted(candidate.get("pass_logit_sha256", {})),
                tokens=work["tokens"],
                attention_cells=work["attention_cells"],
                elapsed_seconds=work["elapsed_seconds"],
            )
        )
    return LocalWorkRecord(
        call=f"{ctx.condition}/{ctx.brief.id}/{GENERATION_STEP}",
        repetition=ctx.repetition,
        brief=ctx.brief.id,
        condition=ctx.condition,
        step=GENERATION_STEP,
        operation="full-generation",
        status=report["status"],
        checkpoint_id=checkpoint["id"],
        test_only=checkpoint["binding"]["config"]["test_only"],
        guidance_scale=request.guidance_scale,
        guidance_policy=GUIDANCE_POLICY,
        seed=request.seed,
        attempts=attempts,
        passes=passes,
        tokens=report["work"]["tokens"],
        attention_cells=report["work"]["attention_cells"],
        elapsed_seconds=report["work"]["elapsed_seconds"],
        proposal_id=report["proposal_id"],
    )


def keep_plan(ctx: StoryContext, plan: NarrativePlan, selection: dict[str, Any]) -> None:
    """Keep the selected plan's artifacts and the identities it was selected from."""
    _, files = rendered_files(plan)
    for name, content in files.items():
        ctx.keep(name, content)
    ctx.keep(
        "conditional-selection.json",
        json.dumps({**selection, "plan_sha256": plan_identity(plan)}, indent=2) + "\n",
    )


def retain(ctx: StoryContext) -> None:
    """List E's durable records with the story's files; save rewrites the same bytes."""
    if ctx.directory is None:
        return
    folder = ctx.directory / CONDITIONAL_DIR
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            name = f"{CONDITIONAL_DIR}/{path.relative_to(folder).as_posix()}"
            ctx.files[name] = path.read_bytes().decode("utf-8")


def _report(directory: Path) -> dict[str, Any] | None:
    path = directory / "report.json"
    return read_json(path) if path.is_file() else None


class _Progress:
    """What has happened so far, for the provenance written however planning ends."""

    def __init__(self) -> None:
        self.conditions: Conditions | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.request: Generation | None = None
        self.generation: dict[str, Any] | None = None
        self.proposal: dict[str, Any] | None = None
        self.completion: dict[str, Any] | None = None
        self.work: list[LocalWorkRecord] = []


def _generate(
    ctx: StoryContext, params: ConditionalPlanningParams, folder: Path, progress: _Progress
) -> tuple[Path, dict[str, Any]]:
    """Generate a learned structural proposal; returns its path and sealed record."""
    try:
        progress.conditions = brief_conditions(ctx.brief, params.condition_fields)
        checkpoint = checked_checkpoint(params)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailedError(GENERATION_STEP, str(error)) from error
    progress.checkpoint = checkpoint_summary(Path(params.checkpoint), checkpoint)
    request = Generation(
        checkpoint=Path(params.checkpoint).resolve(),
        conditions=progress.conditions,
        locks=params.locks,
        seed=generation_seed(ctx.seed),
        guidance_scale=params.guidance_scale,
        max_attempts=params.max_attempts,
        max_tokens=params.max_tokens,
        max_attention_cells=params.max_attention_cells,
        max_seconds=params.max_seconds,
    )
    progress.request = request
    folder.mkdir(parents=True, exist_ok=True)
    request_path = folder / "generation-request.json"
    write_text_lf(request_path, request.model_dump_json(indent=2) + "\n")
    out = folder / "generation"
    try:
        generate(request_path, out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailedError(GENERATION_STEP, str(error)) from error
    finally:
        # The report is written however generation ends; its work is never dropped.
        progress.generation = _report(out)
        if progress.generation is not None:
            record = local_work(ctx, checkpoint, request, progress.generation)
            progress.work.append(record)
            ctx.state.local_work.append(record)
    report = progress.generation
    if report is None or report["status"] != "structural-proposal":
        rows: list[dict[str, Any]] = [] if report is None else report["attempts"]
        diagnostics = [item for row in rows for item in row["diagnostics"]]
        status = "no report" if report is None else report["status"]
        raise StepFailedError(
            GENERATION_STEP,
            f"no accepted structural proposal ({status}): " + "; ".join(diagnostics[-3:]),
        )
    proposal_path = out / "proposal.json"
    progress.proposal = read_sealed(proposal_path)
    return proposal_path, progress.proposal


def _complete(
    ctx: StoryContext,
    params: ConditionalPlanningParams,
    folder: Path,
    proposal: tuple[Path, dict[str, Any]],
    progress: _Progress,
) -> tuple[NarrativePlan, dict[str, Any]]:
    """Complete the proposal's whole text manifest through the run's gateway."""
    proposal_path, record = proposal
    binding = ctx.roles["planner"]
    request = CompletionRequest(
        proposal=proposal_path,
        fields=list(record["completion_fields"]),
        model=binding,
        max_repairs=ctx.structured_config.max_repairs,
        seed=completion_seed(ctx.seed),
        max_tokens=params.completion_max_tokens,
    )
    request_path = folder / "completion-request.json"
    write_text_lf(request_path, request.model_dump_json(indent=2) + "\n")
    out = folder / "completion"
    provider = ctx.gateway.providers[binding.provider].resolved_base_url(ctx.gateway.environ)
    try:
        result = ctx.attributed(
            COMPLETION_STEP,
            lambda: complete_with_gateway(
                ctx.gateway, request_path, out, resolved_provider=provider
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailedError(COMPLETION_STEP, str(error)) from error
    finally:
        # The completion record is written however completion ends, budget stops included.
        if (out / "completion.json").is_file():
            progress.completion = read_sealed(out / "completion.json")
    if result["status"] != "complete" or progress.completion is None:
        diagnostics = [item for row in result["attempts"] for item in row["diagnostics"]]
        raise StepFailedError(
            COMPLETION_STEP,
            f"text completion {result['status']}: " + "; ".join(diagnostics[-3:]),
        )
    try:
        return load_plan(out / "plan.json"), progress.completion
    except PlanError as error:
        raise StepFailedError(COMPLETION_STEP, str(error)) from error


def _scientific_status(checkpoint: dict[str, Any] | None) -> str:
    if checkpoint is None:
        return "unavailable"
    if checkpoint["test_only"]:
        return "test-only mechanics; no scientific evidence"
    return f"passed G3 decision {checkpoint['scientific_evidence']['decision_id']}"


def provenance_record(
    ctx: StoryContext,
    params: ConditionalPlanningParams,
    progress: _Progress,
    status: Literal["succeeded", "failed"],
    calls: list[CallRecord],
) -> dict[str, Any]:
    call_rows = [call.model_dump(mode="json") for call in calls]
    work_rows = [record.model_dump(mode="json") for record in progress.work]
    return {
        "version": "1",
        "pipeline": PIPELINE,
        "status": status,
        "brief": ctx.brief.model_dump(mode="json"),
        "brief_sha256": hashlib.sha256(ctx.brief.model_dump_json().encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
        "git_sha": git(["rev-parse", "HEAD"], Path(__file__).parent) or "unknown",
        "params": params.model_dump(mode="json"),
        "model": ctx.roles["planner"].model_dump(mode="json"),
        "seed": ctx.seed,
        "max_repairs": ctx.structured_config.max_repairs,
        "completion_prompt_id": PROMPT_ID,
        "conditions": None
        if progress.conditions is None
        else progress.conditions.model_dump(mode="json"),
        "checkpoint": progress.checkpoint,
        "generation_request": None
        if progress.request is None
        else progress.request.model_dump(mode="json"),
        "generation": progress.generation,
        "proposal": progress.proposal,
        "completion": progress.completion,
        "local_work": work_rows,
        "local_work_sha256": identity(work_rows),
        "calls": call_rows,
        "calls_sha256": identity(call_rows),
        "plan_sha256": None
        if status != "succeeded"
        else hashlib.sha256(ctx.files["plan.json"].encode()).hexdigest(),
        "scientific_status": _scientific_status(progress.checkpoint),
        "literary_adherence": LITERARY_ADHERENCE,
    }


def generate_e_plan(ctx: StoryContext, params: ConditionalPlanningParams) -> NarrativePlan:
    """A completed E plan, with every attempt, call and unit of local work recorded."""
    if ctx.directory is None:
        raise StepFailedError(GENERATION_STEP, "E needs its story folder for durable records")
    first = len(ctx.state.calls)
    progress = _Progress()
    status: Literal["succeeded", "failed"] = "failed"
    folder = ctx.directory / CONDITIONAL_DIR
    try:
        proposal = _generate(ctx, params, folder, progress)
        plan, completion = _complete(ctx, params, folder, proposal, progress)
        keep_plan(
            ctx,
            plan,
            {
                "pipeline": PIPELINE,
                "source": "generated",
                "checkpoint_id": proposal[1]["checkpoint_id"],
                "proposal_id": proposal[1]["id"],
                "completion_id": completion["id"],
            },
        )
        status = "succeeded"
        return plan
    finally:
        retain(ctx)
        record = provenance_record(ctx, params, progress, status, ctx.state.calls[first:])
        ctx.keep("planning-provenance.json", json.dumps(record, indent=2) + "\n")
