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
from pydantic import Field, ValidationError, model_validator

from nde.llm.config import ModelBinding, ProviderConfig, Strict

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


class ExperimentConfig(Strict):
    name: str
    seed: int
    # Directory of prompt templates, relative to the config file.
    prompts_dir: str = "../prompts"
    providers: dict[str, ProviderConfig]
    roles: dict[RoleName, ModelBinding]
    calls: list[CallSpec] = Field(min_length=1)

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
        return self


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
