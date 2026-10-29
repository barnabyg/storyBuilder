"""Experiment configs: YAML files that map roles to providers and models.

Code refers to roles, never to concrete models. A config is fully validated,
including its prompt templates and API-key variables, before any network call.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, JsonValue, ValidationError, field_validator, model_validator

from nde.llm.config import ModelBinding, ProviderConfig, Strict
from nde.llm.ledger import DEFAULT_ASSUMED_OUTPUT_TOKENS, Price
from nde.llm.structured import DEFAULT_MAX_REPAIRS
from nde.schema import RESPONSE_MODELS
from nde.schema.brief import Split

type RoleName = Literal["annotator", "planner", "prose", "critic", "judge"]


class ConfigError(Exception):
    """The experiment config cannot be run as written."""


class PromptRef(Strict):
    template: str
    version: int = Field(ge=1)

    @property
    def label(self) -> str:
        return f"{self.template}@{self.version}"


class CallSpec(Strict):
    id: str
    role: RoleName
    prompt: PromptRef
    variables: dict[str, str] = Field(default_factory=dict[str, str])
    repeat: int = Field(default=1, ge=1)
    # Name of a registered Pydantic response model (see nde.schema); None for free text.
    response_model: str | None = None

    @field_validator("response_model")
    @classmethod
    def _response_model_is_registered(cls, name: str | None) -> str | None:
        if name is not None and name not in RESPONSE_MODELS:
            known = ", ".join(sorted(RESPONSE_MODELS))
            raise ValueError(f"unknown response model {name!r}; known models: {known}")
        return name


class StructuredConfig(Strict):
    # Repair requests allowed after an invalid first response.
    max_repairs: int = Field(default=DEFAULT_MAX_REPAIRS, ge=0)


class BudgetConfig(Strict):
    per_run_usd: float | None = Field(default=None, ge=0)
    per_day_usd: float | None = Field(default=None, ge=0)
    # Unpriced models are rejected unless allowed; their calls then cost $0 and
    # are invisible to the caps, and the run records a warning.
    allow_unpriced_models: bool = False
    # Output tokens assumed when estimating a call that sets no max_tokens.
    assumed_output_tokens: int = Field(default=DEFAULT_ASSUMED_OUTPUT_TOKENS, ge=1)


class BenchmarkConfig(Strict):
    """Run the calls of one role once per candidate model and compare them (gate G0)."""

    role: RoleName
    candidates: list[ModelBinding] = Field(min_length=1)
    # Validity after repair a model must reach to pass gate G0.
    threshold: float = Field(default=0.95, gt=0, le=1)

    @model_validator(mode="after")
    def _candidates_are_distinct(self) -> Self:
        labels = [candidate.label for candidate in self.candidates]
        if len(set(labels)) != len(labels):
            raise ValueError("benchmark candidates must be distinct provider/model pairs")
        return self


class BriefSelection(Strict):
    """Which briefs a story run uses: a whole split, listed IDs, or listed IDs within a split."""

    split: Split | None = None
    ids: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _names_something(self) -> Self:
        if self.split is None and self.ids is None:
            raise ValueError("name a split or brief ids")
        return self


class ConditionSpec(Strict):
    """A condition: a registered generation pipeline and its parameters."""

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    pipeline: str
    params: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class PlansConfig(Strict):
    """Plan-only development runs, recorded without manufacturing a story."""

    briefs_dir: str = "../benchmark"
    briefs: BriefSelection
    repeat: int = Field(default=1, ge=1)
    params: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class LengthConfig(Strict):
    """Length control: stories must be within `tolerance` of their brief's target length."""

    # Allowed deviation from the target word count, as a fraction (0.10 is ±10%).
    tolerance: float = Field(default=0.10, gt=0, lt=1)
    # Prose generations per story, the first included, before an out-of-range story is kept.
    max_attempts: int = Field(default=3, ge=1)


class ResourcePolicy(Strict):
    """How a comparison treats work measured in units other than provider tokens.

    Required when a compared condition runs a local conditional model. Matching
    still uses provider generation tokens; local model work is reported beside
    them and never converted, summed or called an equal total budget.
    """

    version: Literal["1"] = "1"
    provider_measure: Literal["generation_tokens"] = "generation_tokens"
    local_compute: Literal["report-separately"] = "report-separately"


def _unset(value: object) -> bool:
    return value is None


class EqualBudgetConfig(Strict):
    """Compare two conditions at equal generation-token budget, per brief and repetition."""

    conditions: list[str] = Field(min_length=2, max_length=2)
    # Budgets match when they are within this fraction of the more expensive one.
    tolerance: float = Field(default=0.10, gt=0, lt=1)
    # Revision passes the cheaper condition may get, per story, before matching fails.
    max_revision_passes: int = Field(default=4, ge=0)
    # The shared prose contract's revision prompt.
    revision_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="story_revision", version=1)
    )
    # Omitted from the config identity while unset, so existing config hashes are unchanged.
    resource_policy: ResourcePolicy | None = Field(default=None, exclude_if=_unset)

    @model_validator(mode="after")
    def _two_different_conditions(self) -> Self:
        if self.conditions[0] == self.conditions[1]:
            raise ValueError("equal_budget compares two different conditions")
        return self


class StoriesConfig(Strict):
    """Generate one story per brief, per condition, per repetition."""

    # Benchmark folder with the briefs, relative to the config file.
    briefs_dir: str = "../benchmark"
    briefs: BriefSelection
    conditions: list[ConditionSpec] = Field(min_length=1)
    repeat: int = Field(default=1, ge=1)
    length: LengthConfig = Field(default_factory=LengthConfig)
    equal_budget: EqualBudgetConfig | None = None

    @model_validator(mode="after")
    def _condition_names_are_unique(self) -> Self:
        names = [condition.name for condition in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("condition names must be unique")
        if self.equal_budget is not None:
            for name in self.equal_budget.conditions:
                if name not in names:
                    raise ValueError(f"equal_budget names unknown condition {name!r}")
        return self


type JudgeMode = Literal["pairwise", "rubric"]


class JudgesConfig(Strict):
    """An LLM judge panel for a blind batch: the `judge` role, one model per panel member."""

    batch: str
    # Where batches live, relative to the config file.
    batches_dir: str = "../batches"
    panel: list[ModelBinding] = Field(min_length=1)
    modes: list[JudgeMode] = Field(default_factory=lambda: ["pairwise", "rubric"], min_length=1)
    pairwise_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="judge_pairwise", version=1)
    )
    rubric_prompt: PromptRef = Field(
        default_factory=lambda: PromptRef(template="judge_rubric", version=1)
    )
    # The plan's family rules (§3.1, §4): at least two judge families, none shared
    # with the models that generated the batch. A config may relax them only
    # explicitly; run manifests, judge verdicts and batch reports then say so.
    min_families: int = Field(default=2, ge=1)
    allow_generator_families: bool = False

    @model_validator(mode="after")
    def _panel_spans_enough_families(self) -> Self:
        labels = [judge.label for judge in self.panel]
        if len(set(labels)) != len(labels):
            raise ValueError("judges must be distinct provider/model pairs")
        for judge in self.panel:
            if judge.family is None:
                raise ValueError(f"judge {judge.label} has no family; set `family` on it")
        if len({judge.family for judge in self.panel}) < self.min_families:
            raise ValueError(
                f"the judge panel needs at least {self.min_families} model families "
                "(judges.min_families)"
            )
        return self


class ExperimentConfig(Strict):
    name: str
    seed: int
    # Directory of prompt templates, relative to the config file.
    prompts_dir: str = "../prompts"
    providers: dict[str, ProviderConfig]
    roles: dict[RoleName, ModelBinding] = Field(default_factory=dict[RoleName, ModelBinding])
    calls: list[CallSpec] = Field(default_factory=list[CallSpec])
    # USD per million tokens, keyed by "provider/model".
    prices: dict[str, Price] = Field(default_factory=dict[str, Price])
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    benchmark: BenchmarkConfig | None = None
    stories: StoriesConfig | None = None
    plans: PlansConfig | None = None
    judges: JudgesConfig | None = None

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        if not self.calls and self.stories is None and self.judges is None and self.plans is None:
            raise ValueError("a config needs `calls`, `stories`, `plans` or `judges`")
        if self.plans is not None and (self.stories is not None or self.judges is not None):
            raise ValueError("plan-only runs cannot also generate stories or judge batches")
        bindings = [(f"role {role!r}", binding) for role, binding in self.roles.items()]
        if self.judges is not None:
            bindings += [("judge", judge) for judge in self.judges.panel]
        if self.benchmark is not None:
            bindings += [("benchmark candidate", c) for c in self.benchmark.candidates]
            if all(call.role != self.benchmark.role for call in self.calls):
                raise ValueError(f"no call uses the benchmark role {self.benchmark.role!r}")
        for owner, binding in bindings:
            if binding.provider not in self.providers:
                raise ValueError(f"{owner} uses unknown provider {binding.provider!r}")
        for call in self.calls:
            if not self.bindings_for(call):
                raise ValueError(f"call {call.id!r} uses role {call.role!r}, which has no model")
        ids = [call.id for call in self.calls]
        if len(set(ids)) != len(ids):
            raise ValueError("call ids must be unique")
        unpriced = self.unpriced_warnings(binding for _, binding in self.role_bindings())
        if unpriced and not self.budget.allow_unpriced_models:
            raise ValueError("; ".join(unpriced))
        return self

    def bindings_for(self, call: CallSpec) -> list[ModelBinding]:
        """The models a call runs on: every candidate for the benchmark role, else its role's."""
        if self.benchmark is not None and call.role == self.benchmark.role:
            return list(self.benchmark.candidates)
        return [self.roles[call.role]] if call.role in self.roles else []

    def role_bindings(self, extra: Sequence[RoleName] = ()) -> list[tuple[RoleName, ModelBinding]]:
        """Every (role, model) the config uses, once per model, in a stable order.

        The models of calls, then of the `extra` roles (those a story run's
        pipelines use, which only preparing the config can tell), then the judges.
        """
        seen: dict[str, tuple[RoleName, ModelBinding]] = {}
        for call in self.calls:
            for binding in self.bindings_for(call):
                seen.setdefault(binding.label, (call.role, binding))
        for role in extra:
            if role in self.roles:
                seen.setdefault(self.roles[role].label, (role, self.roles[role]))
        if self.judges is not None:
            for judge in self.judges.panel:
                seen.setdefault(judge.label, ("judge", judge))
        return list(seen.values())

    def unpriced_warnings(self, bindings: Iterable[ModelBinding]) -> list[str]:
        """A warning for each of `bindings` that has no price."""
        return [
            f"model {binding.label} has no price; add it to `prices` or set "
            "budget.allow_unpriced_models (its calls then cost $0 and bypass the caps)"
            for binding in bindings
            if binding.label not in self.prices
        ]


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    config: ExperimentConfig

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def prompts_dir(self) -> Path:
        return (self.directory / self.config.prompts_dir).resolve()

    @property
    def batches_dir(self) -> Path | None:
        judges = self.config.judges
        return None if judges is None else (self.directory / judges.batches_dir).resolve()

    @property
    def briefs_dir(self) -> Path | None:
        stories = self.config.stories
        return None if stories is None else (self.directory / stories.briefs_dir).resolve()

    @property
    def hash(self) -> str:
        canonical = json.dumps(self.config.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_validation_error(error: ValidationError) -> str:
    lines: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<config>"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)


def load_config(path: Path) -> LoadedConfig:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"config {path} is not valid YAML: {error}") from error
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid config {path}:\n{_format_validation_error(error)}") from error
    return LoadedConfig(path=path.resolve(), config=config)
