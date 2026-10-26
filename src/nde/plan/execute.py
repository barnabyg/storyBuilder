"""Plan-only runs share experiment preparation, gateway and final run accounting."""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from nde.briefs import BriefError, load_briefs
from nde.files import write_text_lf
from nde.llm.errors import ProviderError
from nde.llm.gateway import Gateway
from nde.llm.ledger import BudgetExceededError
from nde.narrative.workflow import PlanError, load_plan
from nde.plan.narrative import NarrativePlanningParams, generate_plan, planning_templates
from nde.plan.revision import (
    RevisionPlanningParams,
    revise_plan,
    revision_templates,
    validate_selection,
)
from nde.runs.config import ConfigError, LoadedConfig
from nde.runs.prompts import PromptTemplate, require_template
from nde.runs.state import RunState
from nde.schema.brief import Brief
from nde.stories.pipeline import StepFailedError, StoryContext

PUBLISH_ATTEMPTS = 10
PUBLISH_RETRY_SECONDS = 0.05
WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})


def _publish(staging: Path, destination: Path) -> None:
    """Windows scanners may briefly deny renaming a just-written directory."""
    for attempt in range(PUBLISH_ATTEMPTS):
        try:
            staging.rename(destination)
        except PermissionError as error:
            if (
                getattr(error, "winerror", None) not in WINDOWS_SHARING_ERRORS
                or attempt == PUBLISH_ATTEMPTS - 1
            ):
                # Keep the complete staging bundle for inspection on permanent failure.
                raise
            time.sleep(PUBLISH_RETRY_SECONDS)
        else:
            return


@dataclass(frozen=True)
class PreparedPlans:
    briefs: list[Brief]
    repeat: int
    params: NarrativePlanningParams


def prepare_plans(
    loaded: LoadedConfig,
    templates: dict[str, PromptTemplate],
) -> PreparedPlans | None:
    config = loaded.config.plans
    if config is None:
        return None
    if "planner" not in loaded.config.roles:
        raise ConfigError("plan generation requires the planner role")
    try:
        params = (
            RevisionPlanningParams.model_validate(config.params)
            if "revision" in config.params
            else NarrativePlanningParams.model_validate(config.params)
        )
        if isinstance(params, RevisionPlanningParams):
            source = (loaded.directory / params.saved_plan).resolve()
            params.saved_plan = str(source)
            validate_selection(load_plan(source), params.revision)
        available = load_briefs((loaded.directory / config.briefs_dir).resolve())
    except (BriefError, PlanError, ValidationError) as error:
        raise ConfigError(str(error)) from error
    selection = config.briefs
    chosen = [
        brief
        for brief in available.values()
        if (
            (selection.split is None or brief.split == selection.split)
            and (selection.ids is None or brief.id in selection.ids)
        )
    ]
    if not chosen or (selection.ids is not None and set(selection.ids) != {b.id for b in chosen}):
        raise ConfigError("plan brief selection is empty or contains unknown IDs")
    if any(brief.split != "dev" for brief in chosen):
        raise ConfigError("plan generate is dev-only; held-out generation requires a gate workflow")
    needs = (
        revision_templates(params.revision)
        if isinstance(params, RevisionPlanningParams)
        else planning_templates(params)
    )
    for need in needs:
        require_template(
            templates,
            loaded.prompts_dir,
            need.prompt,
            dict.fromkeys(need.variables, "x"),
        )
    return PreparedPlans(sorted(chosen, key=lambda brief: brief.id), config.repeat, params)


@dataclass(frozen=True)
class PlanRun:
    loaded: LoadedConfig
    prepared: PreparedPlans
    templates: dict[str, PromptTemplate]
    gateway: Gateway
    state: RunState
    run_dir: Path


def run_plans(run: PlanRun) -> bool:
    loaded, prepared, templates = run.loaded, run.prepared, run.templates
    gateway, state, run_dir = run.gateway, run.state, run.run_dir
    records: list[dict[str, object]] = []
    passed = True
    for brief in prepared.briefs:
        for repetition in range(prepared.repeat):
            folder = Path("plans") / brief.id / f"r{repetition + 1}"
            ctx = StoryContext(
                gateway=gateway,
                state=state,
                templates=templates,
                roles=loaded.config.roles,
                structured_config=loaded.config.structured,
                brief=brief,
                condition="C",
                repetition=repetition,
                seed=loaded.config.seed + repetition,
            )
            record: dict[str, object] = {
                "brief": brief.id,
                "repetition": repetition,
                "path": folder.as_posix(),
                "status": "failed",
                "error": None,
            }
            try:
                if isinstance(prepared.params, RevisionPlanningParams):
                    revise_plan(
                        ctx, load_plan(Path(prepared.params.saved_plan)), prepared.params.revision
                    )
                else:
                    generate_plan(ctx, prepared.params)
                record["status"] = "succeeded"
            except StepFailedError as error:
                record["error"] = str(error)
                passed = False
            except (ProviderError, BudgetExceededError) as error:
                record["error"] = str(error)
                raise
            finally:
                # Publish the complete artifact directory in one rename. No accepted
                # plan becomes visible beside an incomplete diagnostic/card bundle.
                destination_folder = run_dir / folder
                destination_folder.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".plan-", dir=destination_folder.parent))
                for name, content in ctx.files.items():
                    destination = staging / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    write_text_lf(destination, content)
                _publish(staging, destination_folder)
                records.append(record)
                write_text_lf(
                    run_dir / "plans.jsonl",
                    "".join(json.dumps(r) + "\n" for r in records),
                )
    return passed
