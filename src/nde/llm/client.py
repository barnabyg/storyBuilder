"""A minimal client for OpenAI-compatible chat-completions endpoints."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
from pydantic import JsonValue


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    seed: int
    params: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    response_format: Mapping[str, JsonValue] | None = None

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            **self.params,
            "model": self.model,
            "messages": list(self.messages),
            "seed": self.seed,
        }
        if self.response_format is not None:
            body["response_format"] = dict(self.response_format)
        return body


@dataclass(frozen=True)
class ChatResult:
    text: str
    # The model ID the provider reports, which may be more specific than the requested one.
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    latency_s: float


class ProviderResponseError(Exception):
    """The provider answered, but not with a usable completion."""

    def __init__(self, message: str, *, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class OpenAICompatibleClient:
    def __init__(self, base_url: str, *, api_key: str | None, timeout_s: float) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout_s
        )

    def close(self) -> None:
        self._http.close()

    def chat(self, request: ChatRequest) -> ChatResult:
        started = time.monotonic()
        response = self._http.post("chat/completions", json=request.body())
        latency = time.monotonic() - started
        if not response.is_success:
            raise ProviderResponseError(
                f"HTTP {response.status_code} from provider",
                status=response.status_code,
                body=response.text,
            )
        return _parse_completion(response, request.model, latency)


def _parse_completion(response: httpx.Response, requested_model: str, latency: float) -> ChatResult:
    try:
        data: dict[str, Any] = response.json()
        choice: dict[str, Any] = data["choices"][0]
        content: object = choice["message"].get("content")
        usage: dict[str, Any] = data.get("usage") or {}
        return ChatResult(
            text=content if isinstance(content, str) else "",
            model=str(data.get("model") or requested_model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason"),
            latency_s=latency,
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise ProviderResponseError(
            f"malformed completion from provider: {error!r}",
            status=response.status_code,
            body=response.text,
        ) from error
