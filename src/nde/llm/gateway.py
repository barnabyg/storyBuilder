"""The gateway: every LLM call in the project goes through `Gateway`."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from nde.llm.cache import CacheMode, CacheStats, ResponseCache, cache_key
from nde.llm.client import ChatRequest, ChatResult, Message, OpenAICompatibleClient
from nde.llm.config import ModelBinding, ProviderConfig
from nde.llm.ledger import CostLedger, Pricing


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

    def complete(self, binding: ModelBinding, prompt: Prompt, *, seed: int) -> Completion:
        request = ChatRequest(
            model=binding.model, messages=prompt.messages, seed=seed, params=binding.params
        )
        key = cache_key(binding.provider, prompt.template, prompt.version, request)
        cache = self.cache if self.cache_mode != "bypass" else None
        if cache is not None and self.cache_mode == "use":
            stored = cache.get(key)
            if stored is not None:
                self.cache_stats.hits += 1
                return Completion(binding, prompt, seed, stored, cached=True, cost_usd=0.0)
        self.cache_stats.misses += 1
        reservation = (
            self.ledger.reserve(self.pricing.estimate(binding.label, request))
            if self.ledger is not None
            else None
        )
        cost = 0.0
        try:
            result = self._client(binding.provider).chat(request)
            cost = self._token_cost(binding, result)
        finally:
            if self.ledger is not None and reservation is not None:
                self.ledger.settle(reservation, cost)
        if cache is not None:
            cache.put(key, request, result)
        return Completion(binding, prompt, seed, result, cached=False, cost_usd=cost)

    def _token_cost(self, binding: ModelBinding, result: ChatResult) -> float:
        return self.pricing.cost(binding.label, result.input_tokens, result.output_tokens)

    def token_cost(self, completion: "Completion") -> float:
        """What a completion's tokens cost at list price, whether or not it was cached."""
        return self._token_cost(completion.binding, completion.result)
