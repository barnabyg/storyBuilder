"""The gateway: every LLM call in the project goes through `Gateway`."""

from collections.abc import Mapping
from dataclasses import dataclass

from nde.llm.client import ChatRequest, ChatResult, Message, OpenAICompatibleClient
from nde.llm.config import ModelBinding, ProviderConfig


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

    @property
    def text(self) -> str:
        return self.result.text


class Gateway:
    def __init__(
        self, providers: Mapping[str, ProviderConfig], *, environ: Mapping[str, str]
    ) -> None:
        self._providers = providers
        self._environ = environ
        self._clients: dict[str, OpenAICompatibleClient] = {}

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def _client(self, provider: str) -> OpenAICompatibleClient:
        if provider not in self._clients:
            config = self._providers[provider]
            api_key = self._environ.get(config.api_key_env) if config.api_key_env else None
            self._clients[provider] = OpenAICompatibleClient(
                config.resolved_base_url(self._environ),
                api_key=api_key,
                timeout_s=config.timeout_s,
            )
        return self._clients[provider]

    def complete(self, binding: ModelBinding, prompt: Prompt, *, seed: int) -> Completion:
        request = ChatRequest(
            model=binding.model, messages=prompt.messages, seed=seed, params=binding.params
        )
        result = self._client(binding.provider).chat(request)
        return Completion(binding=binding, prompt=prompt, seed=seed, result=result)
