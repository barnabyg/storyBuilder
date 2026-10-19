"""Run an experiment config and record it as a run directory."""

import secrets
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nde.eval.judges import (
    JUDGEMENTS,
    JudgeRun,
    PreparedJudges,
    prepare_judges,
    run_judges,
    write_records,
)
from nde.llm.cache import DEFAULT_CACHE_PATH, CacheMode, ResponseCache
from nde.llm.config import ModelBinding
from nde.llm.errors import ProviderError
from nde.llm.gateway import Gateway, Prompt
from nde.llm.ledger import DEFAULT_LEDGER_PATH, BudgetExceededError, CostLedger, Pricing
from nde.llm.structured import StructuredOutputError
from nde.runs.benchmark import benchmark_rows
from nde.runs.config import (
    CallSpec,
    ConfigError,
    ExperimentConfig,
    LoadedConfig,
    RoleName,
    load_config,
)
from nde.runs.prompts import PromptTemplate, load_template
from nde.runs.records import (
    CALLS,
    OUTPUTS,
    STORIES,
    BenchmarkSummary,
    CacheSummary,
    CallRecord,
    CostSummary,
    JudgesSummary,
    Manifest,
    ModelUse,
    OutputRecord,
    RunError,
    StoriesSummary,
    StructuredMetrics,
    TemplateUse,
    write_jsonl,
    write_manifest,
)
from nde.runs.report import write_report
from nde.runs.state import Attribution, RunState, record_calls_since
from nde.schema import RESPONSE_MODELS
from nde.stories.execute import PreparedStories, StoryRun, prepare_stories, run_stories

type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RunOptions:
    runs_dir: Path
    environ: Mapping[str, str]
    cache_path: Path = DEFAULT_CACHE_PATH
    cache_mode: CacheMode = "use"
    ledger_path: Path = DEFAULT_LEDGER_PATH
    clock: Clock = utc_now


@dataclass(frozen=True)
class PreparedRun:
    loaded: LoadedConfig
    templates: dict[str, PromptTemplate]
    stories: PreparedStories | None = None
    judges: PreparedJudges | None = None

    @property
    def story_roles(self) -> list[RoleName]:
        return [] if self.stories is None else self.stories.roles


def prepare(config_path: Path, environ: Mapping[str, str]) -> PreparedRun:
    """Load and fully validate a config; raises ConfigError before any network call."""
    loaded = load_config(config_path)
    config = loaded.config
    templates: dict[str, PromptTemplate] = {}
    for call in config.calls:
        ref = call.prompt
        if ref.label not in templates:
            templates[ref.label] = load_template(loaded.prompts_dir, ref.template, ref.version)
        templates[ref.label].render(call.variables)
    stories = prepare_stories(loaded, templates)
    story_roles = [] if stories is None else stories.roles
    for provider_name in sorted(
        {binding.provider for binding in config.used_bindings(story_roles)}
    ):
        provider = config.providers[provider_name]
        if provider.api_key_env and not environ.get(provider.api_key_env):
            raise ConfigError(
                f"provider {provider_name!r} needs the environment variable "
                f"{provider.api_key_env}; set it or add it to .env (see .env.example)"
            )
    judges = prepare_judges(loaded, templates)
    return PreparedRun(loaded=loaded, templates=templates, stories=stories, judges=judges)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        # S603/S607: fixed git arguments; git is resolved from PATH as a developer prerequisite.
        completed = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _new_run_dir(runs_dir: Path, name: str, started: datetime) -> Path:
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{stamp}-{name}-{secrets.token_hex(3)}"
    run_dir.mkdir(parents=True)
    return run_dir


def _manifest(prepared: PreparedRun, options: RunOptions, started: datetime) -> Manifest:
    loaded = prepared.loaded
    config = loaded.config
    manifest = Manifest(
        name=config.name,
        status="running",
        git_sha=_git(["rev-parse", "HEAD"], loaded.directory) or "unknown",
        git_dirty=bool(_git(["status", "--porcelain"], loaded.directory)),
        config_path=str(loaded.path),
        config_hash=loaded.hash,
        seed=config.seed,
        models=[
            ModelUse(
                role=role, provider=binding.provider, model=binding.model, family=binding.family
            )
            for role, binding in config.role_bindings(prepared.story_roles)
        ],
        prompt_templates=[
            TemplateUse(template=template.name, version=template.version, sha256=template.sha256)
            for template in prepared.templates.values()
        ],
        cache=CacheSummary(
            mode=options.cache_mode,
            path=None if options.cache_mode == "bypass" else str(options.cache_path),
        ),
        cost=CostSummary(
            per_run_cap_usd=config.budget.per_run_usd, per_day_cap_usd=config.budget.per_day_usd
        ),
        warnings=config.unpriced_warnings(prepared.story_roles),
        started_at=started.isoformat(),
        stories=None if prepared.stories is None else prepared.stories.summary(),
    )
    if prepared.stories is not None:
        manifest.files["stories"] = STORIES
    if prepared.judges is not None:
        judges = prepared.judges
        manifest.judges = JudgesSummary(
            batch=judges.batch.name,
            batch_dir=str(judges.batch),
            modes=list(judges.modes),
            panel=[
                ModelUse(role="judge", provider=j.provider, model=j.model, family=j.family)
                for j in judges.panel
            ],
        )
        manifest.files["judgements"] = JUDGEMENTS
    return manifest


@dataclass(frozen=True)
class CallRun:
    """One execution of a call: which call, on which model, which repetition."""

    call: CallSpec
    binding: ModelBinding
    repetition: int


def _run_call(gateway: Gateway, prepared: PreparedRun, run: CallRun, state: RunState) -> None:
    call, binding, repetition = run.call, run.binding, run.repetition
    config = prepared.loaded.config
    template = prepared.templates[call.prompt.label]
    prompt = Prompt(template.name, template.version, template.render(call.variables))
    seed = config.seed + repetition
    output = OutputRecord(
        call=call.id,
        repetition=repetition,
        role=call.role,
        provider=binding.provider,
        model=binding.model,
        seed=seed,
        text="",
        response_model=call.response_model,
    )
    first_attempt = len(gateway.history)
    try:
        if call.response_model is None:
            output.text = gateway.complete(binding, prompt, seed=seed).text
        else:
            metrics = state.structured.setdefault(binding.label, StructuredMetrics())
            try:
                result = gateway.structured(
                    binding,
                    prompt,
                    RESPONSE_MODELS[call.response_model],
                    seed=seed,
                    max_repairs=config.structured.max_repairs,
                )
            except StructuredOutputError as failure:
                output.text = failure.attempts[-1].text
                output.error = "; ".join(failure.errors)
                metrics.record(attempts=len(failure.attempts), valid=False)
            else:
                output.text = result.attempts[-1].text
                output.parsed = result.value.model_dump(mode="json")
                metrics.record(attempts=len(result.attempts), valid=True)
        state.outputs.append(output)
    finally:
        # Record every provider call made, even when a budget cap stopped a repair loop.
        record_calls_since(gateway, state, first_attempt, Attribution(call.id, repetition))


def _summarise_costs(summary: CostSummary, calls: list[CallRecord]) -> None:
    summary.input_tokens = sum(call.input_tokens for call in calls)
    summary.output_tokens = sum(call.output_tokens for call in calls)
    summary.usd = sum(call.cost_usd for call in calls)
    summary.token_cost_usd = sum(call.token_cost_usd for call in calls)


def _summarise_stories(summary: StoriesSummary, state: RunState) -> None:
    summary.stories = len(state.stories)
    summary.failed = sum(story.status == "failed" for story in state.stories)
    summary.out_of_range = [
        f"{story.condition}/{story.brief}#{story.repetition + 1}"
        for story in state.stories
        if story.status == "ok" and not story.in_range
    ]
    if summary.equal_budget is not None:
        summary.equal_budget.matches = state.budget_matches


def _gateway(config: ExperimentConfig, options: RunOptions) -> Gateway:
    return Gateway(
        config.providers,
        environ=options.environ,
        cache=None if options.cache_mode == "bypass" else ResponseCache(options.cache_path),
        cache_mode=options.cache_mode,
        pricing=Pricing(config.prices, config.budget.assumed_output_tokens),
        ledger=CostLedger(
            options.ledger_path,
            per_run_usd=config.budget.per_run_usd,
            per_day_usd=config.budget.per_day_usd,
            clock=options.clock,
        ),
    )


def _judge_run(prepared: PreparedRun, gateway: Gateway, state: RunState) -> JudgeRun | None:
    if prepared.judges is None:
        return None
    config = prepared.loaded.config
    return JudgeRun(
        gateway=gateway,
        prepared=prepared.judges,
        templates=prepared.templates,
        state=state,
        seed=config.seed,
        max_repairs=config.structured.max_repairs,
        records=[],
    )


@dataclass(frozen=True)
class _Finish:
    """What execute() hands to _finish(), however the run ended."""

    run_dir: Path
    manifest: Manifest
    state: RunState
    gateway: Gateway
    judge_run: JudgeRun | None


def _finish(prepared: PreparedRun, options: RunOptions, end: _Finish) -> None:
    """Summarise and write every run file; runs whether the run succeeded or not."""
    config = prepared.loaded.config
    manifest, state, gateway, run_dir = end.manifest, end.state, end.gateway, end.run_dir
    if gateway.ledger is not None:
        manifest.cost.day_spent_usd = gateway.ledger.day_spent()
    gateway.close()
    _summarise_costs(manifest.cost, state.calls)
    manifest.structured = state.structured
    if config.benchmark is not None:
        manifest.benchmark = BenchmarkSummary(
            role=config.benchmark.role,
            threshold=config.benchmark.threshold,
            rows=benchmark_rows(
                state.calls,
                state.structured,
                threshold=config.benchmark.threshold,
                models=[candidate.label for candidate in config.benchmark.candidates],
            ),
        )
    manifest.cache.hits = gateway.cache_stats.hits
    manifest.cache.misses = gateway.cache_stats.misses
    if manifest.status == "running":
        manifest.status = "failed"
    manifest.finished_at = options.clock().isoformat()
    write_jsonl(run_dir / OUTPUTS, state.outputs)
    write_jsonl(run_dir / CALLS, state.calls)
    if manifest.stories is not None:
        _summarise_stories(manifest.stories, state)
        write_jsonl(run_dir / STORIES, state.stories)
    if end.judge_run is not None and manifest.judges is not None:
        records = end.judge_run.records
        manifest.judges.records = len(records)
        manifest.judges.errors = sum(record.error is not None for record in records)
        write_records(run_dir, end.judge_run.prepared.batch, records)
    write_manifest(run_dir, manifest)
    write_report(run_dir)


def execute(prepared: PreparedRun, options: RunOptions) -> Path:
    """Run a prepared config. The run directory is always left with a manifest and report."""
    config = prepared.loaded.config
    started = options.clock()
    run_dir = _new_run_dir(options.runs_dir, config.name, started)
    manifest = _manifest(prepared, options, started)
    write_manifest(run_dir, manifest)
    state = RunState()
    gateway = _gateway(config, options)
    judge_run = _judge_run(prepared, gateway, state)
    try:
        for call in config.calls:
            for repetition in range(call.repeat):
                # Benchmark candidates take turns, so a budget stop leaves comparable samples.
                for binding in config.bindings_for(call):
                    state.current_call = call.id
                    _run_call(gateway, prepared, CallRun(call, binding, repetition), state)
        if prepared.stories is not None:
            run_stories(
                StoryRun(
                    gateway, prepared.loaded, prepared.templates, prepared.stories, run_dir, state
                )
            )
        if judge_run is not None:
            run_judges(judge_run)
        manifest.status = "succeeded"
    except BudgetExceededError as error:
        manifest.status = "aborted"
        manifest.abort_reason = "budget"
        manifest.abort_message = str(error)
    except ProviderError as error:
        manifest.status = "failed"
        manifest.error = RunError(
            type=error.kind, message=str(error), call=state.current_call, status=error.status
        )
    except BaseException as error:
        # A bug, or an interrupt: still leave a manifest and report behind, then re-raise.
        manifest.error = RunError(type="internal", message=repr(error), call=state.current_call)
        raise
    finally:
        _finish(prepared, options, _Finish(run_dir, manifest, state, gateway, judge_run))
    return run_dir


def run_experiment(config_path: Path, options: RunOptions) -> Path:
    return execute(prepare(config_path, options.environ), options)
