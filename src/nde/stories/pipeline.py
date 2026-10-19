"""Generation pipelines: Python code that turns a brief into a story through roles.

A pipeline is registered by name (see `nde.stories.PIPELINES`) and bound to a
condition's parameters when the config is prepared. It writes a story in two
phases:

- `plan` makes the calls that are never regenerated, such as an outline. It
  returns whatever the prose phase needs (or None).
- `write` makes the prose calls and returns the raw story text. Length control
  calls it again with a new seed when the story is out of range.

Every call goes through a `StoryContext`, which records the brief, condition
and step it served, and keeps intermediate outputs beside the story.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, JsonValue

from nde.llm.config import ModelBinding
from nde.llm.gateway import Gateway, Prompt
from nde.llm.structured import StructuredOutputError
from nde.runs.config import PromptRef, RoleName, StructuredConfig
from nde.runs.prompts import PromptTemplate
from nde.runs.records import StructuredMetrics
from nde.runs.state import Attribution, RunState, record_calls_since
from nde.schema.brief import Brief


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
    # Intermediate outputs to keep beside the story, by file name.
    files: dict[str, str] = field(default_factory=dict[str, str])

    def _prompt(self, ref: PromptRef, variables: Mapping[str, str]) -> Prompt:
        template = self.templates[ref.label]
        return Prompt(template.name, template.version, template.render(variables))

    def _attribution(self, step: str) -> Attribution:
        return Attribution(
            call=f"{self.condition}/{self.brief.id}/{step}",
            repetition=self.repetition,
            brief=self.brief.id,
            condition=self.condition,
            step=step,
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
        """Keep an intermediate output, such as an outline, as a file beside the story."""
        self.files[name] = content


class Pipeline[PlanT](ABC):
    """A generation pipeline bound to one condition's parameters."""

    roles: ClassVar[tuple[RoleName, ...]]

    @abstractmethod
    def templates(self) -> list[TemplateNeed]:
        """Every prompt template the pipeline uses, so configs are checked before any call."""

    @abstractmethod
    def plan(self, ctx: StoryContext) -> PlanT:
        """The calls that length control never regenerates, such as an outline."""

    @abstractmethod
    def write(self, ctx: StoryContext, plan: PlanT, *, seed: int) -> str:
        """The prose calls; returns the raw story text."""


# Builds a pipeline from a condition's params; raises pydantic.ValidationError.
type PipelineFactory = Callable[[Mapping[str, JsonValue]], Pipeline[Any]]
