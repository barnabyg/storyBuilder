"""The gateway: every LLM call in the project goes through `Gateway`."""

import random
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from nde.llm import structured
from nde.llm.cache import CacheMode, CacheStats, ResponseCache, cache_key
from nde.llm.client import ChatRequest, ChatResult, Message, OpenAICompatibleClient
from nde.llm.config import ModelBinding, ProviderConfig, RetryConfig
from nde.llm.errors import ProviderError, RateLimitedError
from nde.llm.ledger import CostLedger, Pricing
from nde.llm.structured import StructuredOutputError, StructuredResult


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt and the template version it came from."""

    template: str
    version: int
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Completion:
    binding: ModelBinding
    prompt: Prompt
    seed: int
    result: ChatResult
    cached: bool
    # New spend: zero for a cache hit.
    cost_usd: float
    # Retries after rate-limited or transient failures before the call succeeded.
    retries: int = 0

    @property
    def text(self) -> str:
        return self.result.text


@dataclass
class Gateway:
    providers: Mapping[str, ProviderConfig]
    environ: Mapping[str, str]
    cache: ResponseCache | None = None
    cache_mode: CacheMode = "use"
    cache_stats: CacheStats = field(default_factory=CacheStats)
    pricing: Pricing = field(default_factory=lambda: Pricing({}))
    ledger: CostLedger | None = None
    # Every completion this gateway has returned, in order, including repair attempts.
    history: list[Completion] = field(default_factory=list[Completion])
    sleep: Callable[[float], None] = time.sleep
    jitter: random.Random = field(default_factory=secrets.SystemRandom)
    _clients: dict[str, OpenAICompatibleClient] = field(
        default_factory=dict[str, OpenAICompatibleClient]
    )

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        if self.cache is not None:
            self.cache.close()
        if self.ledger is not None:
            self.ledger.close()

    def _client(self, provider: str) -> OpenAICompatibleClient:
        if provider not in self._clients:
            config = self.providers[provider]
            api_key = self.environ.get(config.api_key_env) if config.api_key_env else None
            self._clients[provider] = OpenAICompatibleClient(
                config.resolved_base_url(self.environ),
                api_key=api_key,
                timeout_s=config.timeout_s,
            )
        return self._clients[provider]

    def complete(
        self,
        binding: ModelBinding,
        prompt: Prompt,
        *,
        seed: int,
        response_format: Mapping[str, Any] | None = None,
    ) -> Completion:
        request = ChatRequest(
            model=binding.model,
            messages=prompt.messages,
            seed=seed,
            params=binding.params,
            response_format=response_format,
        )
        completion = self._cached(binding, prompt, seed, request) or self._call(
            binding, prompt, seed, request
        )
        self.history.append(completion)
        return completion

    def _cached(
        self, binding: ModelBinding, prompt: Prompt, seed: int, request: ChatRequest
    ) -> Completion | None:
        if self.cache is None or self.cache_mode != "use":
            return None
        stored = self.cache.get(
            cache_key(binding.provider, prompt.template, prompt.version, request)
        )
        if stored is None:
            return None
        self.cache_stats.hits += 1
        return Completion(binding, prompt, seed, stored, cached=True, cost_usd=0.0)

    def _call(
        self, binding: ModelBinding, prompt: Prompt, seed: int, request: ChatRequest
    ) -> Completion:
        self.cache_stats.misses += 1
        reservation = (
            self.ledger.reserve(self.pricing.estimate(binding.label, request))
            if self.ledger is not None
            else None
        )
        cost = 0.0
        try:
            result, retries = self._chat_with_retries(binding.provider, request)
            cost = self._token_cost(binding, result)
        finally:
            if self.ledger is not None and reservation is not None:
                self.ledger.settle(reservation, cost)
        if self.cache is not None and self.cache_mode != "bypass":
            key = cache_key(binding.provider, prompt.template, prompt.version, request)
            self.cache.put(key, request, result)
        return Completion(
            binding, prompt, seed, result, cached=False, cost_usd=cost, retries=retries
        )

    def _chat_with_retries(self, provider: str, request: ChatRequest) -> tuple[ChatResult, int]:
        """Call the provider, retrying retryable errors under its policy."""
        policy = self.providers[provider].retry
        client = self._client(provider)
        retry = 0
        while True:
            try:
                return client.chat(request), retry
            except ProviderError as error:
                if not error.retryable or retry >= policy.max_retries:
                    raise
                self.sleep(self._backoff(policy, retry, error))
            retry += 1

    def _backoff(self, policy: RetryConfig, retry: int, error: ProviderError) -> float:
        ceiling = min(policy.max_delay_s, policy.base_delay_s * 2**retry)
        delay = self.jitter.uniform(0, ceiling)
        if isinstance(error, RateLimitedError) and error.retry_after_s is not None:
            delay = max(delay, min(error.retry_after_s, policy.max_delay_s))
        return delay

    def _token_cost(self, binding: ModelBinding, result: ChatResult) -> float:
        return self.pricing.cost(binding.label, result.input_tokens, result.output_tokens)

    def token_cost(self, completion: Completion) -> float:
        """What a completion's tokens cost at list price, whether or not it was cached."""
        return self._token_cost(completion.binding, completion.result)

    def supports_json_schema(self, binding: ModelBinding) -> bool:
        if binding.json_schema is not None:
            return binding.json_schema
        return self.providers[binding.provider].json_schema

    def structured[T: BaseModel](
        self,
        binding: ModelBinding,
        prompt: Prompt,
        model: type[T],
        *,
        seed: int,
        max_repairs: int = structured.DEFAULT_MAX_REPAIRS,
    ) -> StructuredResult[T]:
        """Ask for a response conforming to `model`, repairing invalid ones.

        Raises StructuredOutputError once `max_repairs` repairs have failed.
        """
        response_format: dict[str, Any] | None = None
        if self.supports_json_schema(binding):
            response_format = structured.response_format(model)
            messages = prompt.messages
        else:
            messages = structured.with_schema_instruction(prompt.messages, model)
        attempts: list[Completion] = []
        errors: list[str] = []
        for _ in range(max_repairs + 1):
            attempt = Prompt(prompt.template, prompt.version, messages)
            completion = self.complete(binding, attempt, seed=seed, response_format=response_format)
            attempts.append(completion)
            try:
                value = structured.parse(completion.text, model)
            except ValidationError as error:
                errors = structured.describe_errors(error)
                messages = (*messages, *structured.repair_messages(completion.text, errors))
            else:
                return StructuredResult(value, attempts)
        raise StructuredOutputError(binding.label, errors, attempts)
