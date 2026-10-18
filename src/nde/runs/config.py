"""Experiment configs: YAML files that map roles to providers and models.

Code refers to roles, never to concrete models. A config is fully validated,
including its prompt templates and API-key variables, before any network call.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from nde.llm.config import ModelBinding, ProviderConfig, Strict
from nde.llm.ledger import DEFAULT_ASSUMED_OUTPUT_TOKENS, Price
from nde.llm.structured import DEFAULT_MAX_REPAIRS
from nde.schema import RESPONSE_MODELS

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


class ExperimentConfig(Strict):
    name: str
    seed: int
    # Directory of prompt templates, relative to the config file.
    prompts_dir: str = "../prompts"
    providers: dict[str, ProviderConfig]
    roles: dict[RoleName, ModelBinding]
    calls: list[CallSpec] = Field(min_length=1)
    # USD per million tokens, keyed by "provider/model".
    prices: dict[str, Price] = Field(default_factory=dict[str, Price])
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    structured: StructuredConfig = Field(default_factory=StructuredConfig)

    @model_validator(mode="after")
    def _references_resolve(self) -> Self:
        for role, binding in self.roles.items():
            if binding.provider not in self.providers:
                raise ValueError(f"role {role!r} uses unknown provider {binding.provider!r}")
        for call in self.calls:
            if call.role not in self.roles:
                raise ValueError(f"call {call.id!r} uses role {call.role!r}, which has no model")
        ids = [call.id for call in self.calls]
        if len(set(ids)) != len(ids):
            raise ValueError("call ids must be unique")
        if not self.budget.allow_unpriced_models and self.unpriced_models():
            raise ValueError("; ".join(self.unpriced_warnings()))
        return self

    def used_bindings(self) -> list[ModelBinding]:
        """Every model binding some call uses, once each, in call order."""
        bindings = {self.roles[call.role].label: self.roles[call.role] for call in self.calls}
        return list(bindings.values())

    def unpriced_models(self) -> list[str]:
        return [b.label for b in self.used_bindings() if b.label not in self.prices]

    def unpriced_warnings(self) -> list[str]:
        return [
            f"model {label} has no price; add it to `prices` or set "
            "budget.allow_unpriced_models (its calls then cost $0 and bypass the caps)"
            for label in self.unpriced_models()
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
