"""Generation pipelines: Python code that turns a brief into a story through roles.

A pipeline is registered by name (see `nde.stories.PIPELINES`) and bound to a
condition's parameters when the config is prepared. It writes a story in two
phases:

- `plan` makes the calls that are never regenerated, such as an outline. It
  returns whatever the prose phase needs (or None).
- `write` makes the prose calls and returns the raw story text. Length control
  calls it again with a new seed when the story is out of range.

A pipeline may also work on the chosen story afterwards, in `finish`.

Every call goes through a `StoryContext`, which records the brief, condition
and step it served, and keeps intermediate outputs beside the story.
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, JsonValue

from nde.llm.config import ModelBinding
from nde.llm.gateway import Gateway, Prompt
from nde.llm.structured import StructuredOutputError
from nde.runs.config import ConfigError, PromptRef, RoleName, StructuredConfig
from nde.runs.prompts import PromptTemplate
from nde.runs.records import StructuredMetrics
from nde.runs.state import Attribution, RunState, record_calls_since
from nde.schema.brief import Brief

ATTEMPTS = "attempts"


def derive_seed(seed: int, *labels: object) -> int:
    """A deterministic seed for a sub-step, such as a regeneration, derived from `seed`."""
    material = ":".join(str(part) for part in (seed, *labels)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


@dataclass(frozen=True)
class TemplateNeed:
    """A prompt template a pipeline uses, and the variables it fills in."""

    prompt: PromptRef
    variables: frozenset[str]


@dataclass(frozen=True)
class Step:
    """One pipeline step's call: its name in the run record, role, template and variables."""

    name: str
    role: RoleName
    prompt: PromptRef
    variables: Mapping[str, str]


class StepFailedError(Exception):
    """A step could not produce a valid output, so the story cannot be written."""

    def __init__(self, step: str, message: str) -> None:
        self.step = step
        super().__init__(f"step {step!r}: {message}")


@dataclass
class StoryContext:
    """One story in progress: its brief and seed, and calls made through roles."""

    gateway: Gateway
    state: RunState
    templates: Mapping[str, PromptTemplate]
    roles: Mapping[RoleName, ModelBinding]
    structured_config: StructuredConfig
    brief: Brief
    condition: str
    repetition: int
    seed: int
    # The length-control attempt being written; None while planning.
    attempt: int | None = None
    # Folder for kept files, relative to the story's, such as "revisions/1/".
    prefix: str = ""
    # Intermediate outputs to keep beside the story, by path relative to its folder.
    files: dict[str, str] = field(default_factory=dict[str, str])
    # Approved development examples, supplied only to structural requests.
    planning_examples: str | None = None
    # The story's folder in the run directory, for pipelines whose local work keeps
    # durable records while it runs; None outside a story run.
    directory: Path | None = None

    def _prompt(self, ref: PromptRef, variables: Mapping[str, str]) -> Prompt:
        template = self.templates[ref.label]
        return Prompt(template.name, template.version, template.render(variables))

    def _attribution(self, step: str) -> Attribution:
        call = f"{self.condition}/{self.brief.id}/{step}"
        if self.attempt:
            call += f" (attempt {self.attempt + 1})"
        return Attribution(
            call=call,
            repetition=self.repetition,
            brief=self.brief.id,
            condition=self.condition,
            step=step,
            story_attempt=self.attempt,
        )

    def complete(self, step: Step, *, seed: int) -> str:
        """One free-text call."""
        attribution = self._attribution(step.name)
        self.state.current_call = attribution.call
        first = len(self.gateway.history)
        try:
            prompt = self._prompt(step.prompt, step.variables)
            return self.gateway.complete(self.roles[step.role], prompt, seed=seed).text
        finally:
            record_calls_since(self.gateway, self.state, first, attribution)

    def attributed[T](self, step: str, request: Callable[[], T]) -> T:
        """Calls an established workflow makes on the gateway, attributed to `step`.

        For requests that do not render a prompt template, such as conditional text
        completion; every provider call they make is recorded like any other.
        """
        attribution = self._attribution(step)
        self.state.current_call = attribution.call
        first = len(self.gateway.history)
        try:
            return request()
        finally:
            record_calls_since(self.gateway, self.state, first, attribution)

    def structured[T: BaseModel](self, step: Step, model: type[T], *, seed: int) -> T:
        """One structured call with bounded repair; raises StepFailedError when repairs run out."""
        attribution = self._attribution(step.name)
        self.state.current_call = attribution.call
        binding = self.roles[step.role]
        metrics = self.state.structured.setdefault(binding.label, StructuredMetrics())
        first = len(self.gateway.history)
        try:
            result = self.gateway.structured(
                binding,
                self._prompt(step.prompt, step.variables),
                model,
                seed=seed,
                max_repairs=self.structured_config.max_repairs,
            )
        except StructuredOutputError as failure:
            metrics.record(attempts=len(failure.attempts), valid=False)
            raise StepFailedError(step.name, "; ".join(failure.errors)) from failure
        finally:
            record_calls_since(self.gateway, self.state, first, attribution)
        metrics.record(attempts=len(result.attempts), valid=True)
        return result.value

    def keep(self, name: str, content: str) -> None:
        """Keep an intermediate output as a file beside the story.

        Files kept while writing go under `attempts/<n>/`, one folder per
        length-control attempt; files kept while planning sit beside the story.
        """
        folder = "" if self.attempt is None else f"{ATTEMPTS}/{self.attempt + 1}/"
        self.files[self.prefix + folder + name] = content


class Pipeline[PlanT](ABC):
    """A generation pipeline bound to one condition's parameters."""

    roles: ClassVar[tuple[RoleName, ...]]
    # Whether the pipeline runs a local conditional model, whose work is reported in
    # its own units and needs a registered resource policy in equal-budget comparisons.
    local_model: ClassVar[bool] = False
    # Whether the pipeline may only run on dev briefs; a config naming others is rejected.
    dev_only: ClassVar[bool] = False

    def required_roles(self) -> tuple[RoleName, ...]:
        """Roles needed by the configured pipeline, including optional stages."""
        return self.roles

    def check_roles(self, roles: Mapping[RoleName, ModelBinding]) -> None:
        """Validate role relationships during preparation, before any provider call."""
        for role in self.required_roles():
            if role not in roles:
                raise ConfigError(f"pipeline uses role {role!r}, which has no model in `roles`")

    def check_inputs(self) -> None:  # noqa: B027 - an optional hook; most pipelines have none
        """Validate local model inputs during preparation; raises ConfigError."""

    def check_structured(self, structured: StructuredConfig) -> None:  # noqa: B027 - optional
        """Validate the run's structured-output settings during preparation; raises ConfigError."""

    @abstractmethod
    def templates(self) -> list[TemplateNeed]:
        """Every prompt template the pipeline uses, so configs are checked before any call."""

    @abstractmethod
    def plan(self, ctx: StoryContext) -> PlanT:
        """The calls that length control never regenerates, such as an outline."""

    @abstractmethod
    def write(self, ctx: StoryContext, plan: PlanT, *, seed: int) -> str:
        """The prose calls; returns the raw story text."""

    def finish(
        self,
        ctx: StoryContext,  # noqa: ARG002 - an optional hook; most pipelines have none
        plan: PlanT,  # noqa: ARG002
        story: str,
        *,
        attempt: int,  # noqa: ARG002
    ) -> str:
        """Work on the chosen story after length control; returns its final text.

        `attempt` is the chosen draft's length-control attempt. Most pipelines have
        nothing to do here. Kept files sit beside the story.
        """
        return story


# Builds a pipeline from a condition's params; raises pydantic.ValidationError.
type PipelineFactory = Callable[[Mapping[str, JsonValue]], Pipeline[Any]]
