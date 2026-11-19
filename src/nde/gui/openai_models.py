"""OpenAI model choices for the GUI's text generation and cost ledger."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from nde.continuity.pipeline import Continuity
from nde.llm.client import OpenAICompatibleClient
from nde.llm.config import ProviderConfig, Strict
from nde.llm.errors import AuthenticationError, ProviderError
from nde.llm.ledger import Price
from nde.runs.config import ConfigError, ExperimentConfig, LoadedConfig

_MORE_EFFORT = ("low", "medium", "high", "xhigh", "max")
_ALL_EFFORT = ("none", *_MORE_EFFORT)
_THROUGH_XHIGH = _ALL_EFFORT[:-1]
_THROUGH_HIGH = _THROUGH_XHIGH[:-1]
_MINIMAL_THROUGH_HIGH = ("minimal", "low", "medium", "high")
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_LOOKUP_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class ModelSupport:
    efforts: tuple[str, ...]
    input_per_mtok: float
    output_per_mtok: float
    max_output_tokens: int = 128_000


# GET /models has IDs and availability but no capabilities or pricing. Only list
# general-purpose Chat Completions models whose parameters and rates are known.
# Standard short-context rates in USD per million tokens, checked 2026-11-19.
# See https://developers.openai.com/api/docs/models and the individual model pages.
# Use the regular GPT-5.6 Sol rate so a temporary discount cannot understate the cap.
_MODELS = {
    "gpt-6.2-astra": ModelSupport(_MORE_EFFORT, 8, 40),
    "gpt-6.2-sol": ModelSupport(_MORE_EFFORT, 2, 10),
    "gpt-6.1-sol": ModelSupport(_MORE_EFFORT, 2, 10),
    "gpt-6.1-terra": ModelSupport(_ALL_EFFORT, 1, 5),
    "gpt-6-astra": ModelSupport(_MORE_EFFORT, 10, 50),
    "gpt-6-sol": ModelSupport(_ALL_EFFORT, 2, 10),
    "gpt-6-luna": ModelSupport(_ALL_EFFORT, 0.1, 0.5),
    "gpt-5.6-sol": ModelSupport(_ALL_EFFORT, 5, 30),
    "gpt-5.6-terra": ModelSupport(_ALL_EFFORT, 2, 12),
    "gpt-5.6-luna": ModelSupport(_ALL_EFFORT, 0.2, 1.2),
    "gpt-5.5": ModelSupport(_THROUGH_XHIGH, 5, 30),
    "gpt-5.4": ModelSupport(_THROUGH_XHIGH, 2.5, 15),
    "gpt-5.4-mini": ModelSupport(_THROUGH_XHIGH, 0.75, 4.5),
    "gpt-5.4-nano": ModelSupport(_THROUGH_XHIGH, 0.2, 1.25),
    "gpt-5.2": ModelSupport(_THROUGH_XHIGH, 1.75, 14),
    "gpt-5.1": ModelSupport(_THROUGH_HIGH, 1.25, 10),
    "gpt-5": ModelSupport(_MINIMAL_THROUGH_HIGH, 1.25, 10),
    "gpt-5-mini": ModelSupport(_MINIMAL_THROUGH_HIGH, 0.25, 2),
    "gpt-5-nano": ModelSupport(_MINIMAL_THROUGH_HIGH, 0.05, 0.4),
    "gpt-4.1": ModelSupport((), 2, 8, 32_768),
    "gpt-4.1-mini": ModelSupport((), 0.4, 1.6, 32_768),
}


class ModelChoice(Strict):
    model: str
    effort: str = ""


def model_support(model: str) -> ModelSupport | None:
    return _MODELS.get(_SNAPSHOT_SUFFIX.sub("", model))


def story_provider(config: ExperimentConfig) -> ProviderConfig | None:
    if any(
        binding.provider == "openai"
        for role, binding in config.roles.items()
        if role in Continuity.roles
    ):
        return config.providers["openai"]
    return None


def available_models(
    provider: ProviderConfig, environ: Mapping[str, str]
) -> dict[str, ModelSupport]:
    key = environ.get(provider.api_key_env) if provider.api_key_env else None
    if provider.api_key_env and not key:
        raise ConfigError(f"Set {provider.api_key_env} to load the OpenAI model list.")
    client = OpenAICompatibleClient(
        provider.resolved_base_url(environ),
        api_key=key,
        timeout_s=min(_LOOKUP_TIMEOUT_S, provider.timeout_s),
    )
    try:
        listed = client.list_models()
    except AuthenticationError as error:
        raise ConfigError("OpenAI denied access to the model list. Check the API key.") from error
    except ProviderError as error:
        raise ConfigError("Could not load models from OpenAI. Try refreshing the list.") from error
    finally:
        client.close()
    today = datetime.now(UTC).date()
    return {
        entry.id: support
        for entry in sorted(listed, key=lambda entry: entry.id, reverse=True)
        if (entry.shutdown_date is None or entry.shutdown_date > today)
        and (support := model_support(entry.id)) is not None
    }


def apply_choice(loaded: LoadedConfig, choice: ModelChoice) -> LoadedConfig:
    config = loaded.config
    support = model_support(choice.model)
    if support is None or (choice.effort and choice.effort not in support.efforts):
        raise ConfigError("Choose a supported model and thinking level in Settings.")
    roles = config.roles.copy()
    for role in Continuity.roles:
        binding = roles.get(role)
        if binding is None or binding.provider != "openai":
            continue
        params = binding.params.copy()
        params.pop("reasoning_effort", None)
        if choice.effort:
            params["reasoning_effort"] = choice.effort
        if support.efforts and choice.effort != "none":
            for name in ("temperature", "top_p", "logprobs", "top_logprobs"):
                params.pop(name, None)
        limit = params.pop("max_tokens", None)
        limit = params.get("max_completion_tokens", limit)
        if isinstance(limit, int):
            params["max_completion_tokens"] = min(limit, support.max_output_tokens)
        roles[role] = binding.model_copy(update={"model": choice.model, "params": params})
    prices = config.prices.copy()
    prices.setdefault(
        f"openai/{choice.model}",
        Price(input_per_mtok=support.input_per_mtok, output_per_mtok=support.output_per_mtok),
    )
    return replace(loaded, config=config.model_copy(update={"roles": roles, "prices": prices}))
