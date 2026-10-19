"""The mutable state of a run in progress, shared by call-list and story execution."""

from dataclasses import dataclass, field

from nde.llm.gateway import Completion, Gateway
from nde.runs.records import CallRecord, OutputRecord, StoryRecord, StructuredMetrics


@dataclass
class RunState:
    outputs: list[OutputRecord] = field(default_factory=list[OutputRecord])
    calls: list[CallRecord] = field(default_factory=list[CallRecord])
    stories: list[StoryRecord] = field(default_factory=list[StoryRecord])
    structured: dict[str, StructuredMetrics] = field(default_factory=dict[str, StructuredMetrics])
    # The call in progress, for failure records.
    current_call: str | None = None


@dataclass(frozen=True)
class Attribution:
    """What a provider call served: a config call, or a step of a story."""

    call: str
    repetition: int
    brief: str | None = None
    condition: str | None = None
    step: str | None = None


def call_record(
    gateway: Gateway, attribution: Attribution, attempt: int, completion: Completion
) -> CallRecord:
    result = completion.result
    return CallRecord(
        call=attribution.call,
        repetition=attribution.repetition,
        brief=attribution.brief,
        condition=attribution.condition,
        step=attribution.step,
        attempt=attempt,
        provider=completion.binding.provider,
        model=completion.binding.model,
        reported_model=result.model,
        template=completion.prompt.template,
        template_version=completion.prompt.version,
        seed=completion.seed,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        cached=completion.cached,
        cost_usd=completion.cost_usd,
        retries=completion.retries,
        token_cost_usd=gateway.token_cost(completion),
    )


def record_calls_since(
    gateway: Gateway, state: RunState, first: int, attribution: Attribution
) -> None:
    """Record every provider call the gateway made since history index `first`."""
    for attempt, completion in enumerate(gateway.history[first:]):
        state.calls.append(call_record(gateway, attribution, attempt, completion))
