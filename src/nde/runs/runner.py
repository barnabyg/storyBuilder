"""Run an experiment config and record it as a run directory."""

import secrets
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from nde.llm.cache import DEFAULT_CACHE_PATH, CacheMode, ResponseCache
from nde.llm.gateway import Gateway, Prompt
from nde.runs.config import CallSpec, ConfigError, LoadedConfig, RoleName, load_config
from nde.runs.prompts import PromptTemplate, load_template
from nde.runs.records import (
    CALLS,
    OUTPUTS,
    CacheSummary,
    CallRecord,
    Manifest,
    ModelUse,
    OutputRecord,
    TemplateUse,
    write_jsonl,
    write_manifest,
)
from nde.runs.report import write_report

type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RunOptions:
    runs_dir: Path
    environ: Mapping[str, str]
    cache_path: Path = DEFAULT_CACHE_PATH
    cache_mode: CacheMode = "use"
    clock: Clock = utc_now


@dataclass(frozen=True)
class PreparedRun:
    loaded: LoadedConfig
    templates: dict[str, PromptTemplate]


@dataclass
class RunState:
    outputs: list[OutputRecord] = field(default_factory=list[OutputRecord])
    calls: list[CallRecord] = field(default_factory=list[CallRecord])


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
    for provider_name in sorted({config.roles[call.role].provider for call in config.calls}):
        provider = config.providers[provider_name]
        if provider.api_key_env and not environ.get(provider.api_key_env):
            raise ConfigError(
                f"provider {provider_name!r} needs the environment variable "
                f"{provider.api_key_env}; set it or add it to .env (see .env.example)"
            )
    return PreparedRun(loaded=loaded, templates=templates)


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
    roles: list[RoleName] = list(dict.fromkeys(call.role for call in config.calls))
    return Manifest(
        name=config.name,
        status="running",
        git_sha=_git(["rev-parse", "HEAD"], loaded.directory) or "unknown",
        git_dirty=bool(_git(["status", "--porcelain"], loaded.directory)),
        config_path=str(loaded.path),
        config_hash=loaded.hash,
        seed=config.seed,
        models=[
            ModelUse(
                role=role, provider=config.roles[role].provider, model=config.roles[role].model
            )
            for role in roles
        ],
        prompt_templates=[
            TemplateUse(template=template.name, version=template.version, sha256=template.sha256)
            for template in prepared.templates.values()
        ],
        cache=CacheSummary(
            mode=options.cache_mode,
            path=None if options.cache_mode == "bypass" else str(options.cache_path),
        ),
        started_at=started.isoformat(),
    )


def _run_call(
    gateway: Gateway, prepared: PreparedRun, call: CallSpec, repetition: int, state: RunState
) -> None:
    config = prepared.loaded.config
    binding = config.roles[call.role]
    template = prepared.templates[call.prompt.label]
    prompt = Prompt(template.name, template.version, template.render(call.variables))
    seed = config.seed + repetition
    completion = gateway.complete(binding, prompt, seed=seed)
    state.outputs.append(
        OutputRecord(
            call=call.id,
            repetition=repetition,
            role=call.role,
            provider=binding.provider,
            model=binding.model,
            seed=seed,
            text=completion.text,
        )
    )
    result = completion.result
    state.calls.append(
        CallRecord(
            call=call.id,
            repetition=repetition,
            provider=binding.provider,
            model=binding.model,
            response_model=result.model,
            template=template.name,
            template_version=template.version,
            seed=seed,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_s=result.latency_s,
            cached=completion.cached,
        )
    )


def execute(prepared: PreparedRun, options: RunOptions) -> Path:
    """Run a prepared config. The run directory is always left with a manifest and report."""
    config = prepared.loaded.config
    started = options.clock()
    run_dir = _new_run_dir(options.runs_dir, config.name, started)
    manifest = _manifest(prepared, options, started)
    write_manifest(run_dir, manifest)
    state = RunState()
    gateway = Gateway(
        config.providers,
        environ=options.environ,
        cache=None if options.cache_mode == "bypass" else ResponseCache(options.cache_path),
        cache_mode=options.cache_mode,
    )
    try:
        for call in config.calls:
            for repetition in range(call.repeat):
                _run_call(gateway, prepared, call, repetition, state)
        manifest.status = "succeeded"
    finally:
        gateway.close()
        manifest.cache.hits = gateway.cache_stats.hits
        manifest.cache.misses = gateway.cache_stats.misses
        if manifest.status == "running":
            manifest.status = "failed"
        manifest.finished_at = options.clock().isoformat()
        write_jsonl(run_dir / OUTPUTS, state.outputs)
        write_jsonl(run_dir / CALLS, state.calls)
        write_manifest(run_dir, manifest)
        write_report(run_dir)
    return run_dir


def run_experiment(config_path: Path, options: RunOptions) -> Path:
    return execute(prepare(config_path, options.environ), options)
