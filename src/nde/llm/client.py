"""A minimal client for OpenAI-compatible chat-completions endpoints."""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import httpx
from pydantic import JsonValue

from nde.llm.errors import (
    AuthenticationError,
    ContentRefusedError,
    InvalidRequestError,
    ProviderError,
    RateLimitedError,
    TransientError,
)


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


# Error codes providers use for content-policy refusals.
_CONTENT_REFUSAL_CODES = {"content_filter", "content_policy_violation"}
_RETRYABLE_CLIENT_STATUSES = {408, 409}
_TOO_MANY_REQUESTS = 429
_FIRST_SERVER_ERROR = 500
_AUTHENTICATION_STATUSES = {401, 403}


class OpenAICompatibleClient:
    """Makes one chat-completion request; every failure raises a common ProviderError."""

    def __init__(self, base_url: str, *, api_key: str | None, timeout_s: float) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout_s
        )

    def close(self) -> None:
        self._http.close()

    def chat(self, request: ChatRequest) -> ChatResult:
        started = time.monotonic()
        try:
            response = self._http.post("chat/completions", json=request.body())
        except httpx.TimeoutException as error:
            raise TransientError(f"request timed out: {error!r}") from error
        except (httpx.NetworkError, httpx.RemoteProtocolError) as error:
            raise TransientError(f"connection failed: {error!r}") from error
        except httpx.HTTPError as error:
            raise InvalidRequestError(f"request could not be sent: {error!r}") from error
        latency = time.monotonic() - started
        if not response.is_success:
            raise _status_error(response)
        result = _parse_completion(response, request.model, latency)
        if result.finish_reason == "content_filter":
            raise ContentRefusedError("the provider's content filter stopped the response")
        return result


def _error_details(response: httpx.Response) -> tuple[str, str | None]:
    try:
        error: object = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict):
        details = cast("dict[str, Any]", error)
        code = details.get("code")
        return str(details.get("message") or response.text), str(code) if code else None
    return response.text[:500], None


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _status_error(response: httpx.Response) -> ProviderError:
    status = response.status_code
    message, code = _error_details(response)
    text = f"HTTP {status}: {message}"
    if status == _TOO_MANY_REQUESTS:
        return RateLimitedError(text, status=status, retry_after_s=_retry_after(response))
    if status >= _FIRST_SERVER_ERROR or status in _RETRYABLE_CLIENT_STATUSES:
        return TransientError(text, status=status)
    if status in _AUTHENTICATION_STATUSES:
        return AuthenticationError(text, status=status)
    if code in _CONTENT_REFUSAL_CODES:
        return ContentRefusedError(text, status=status)
    return InvalidRequestError(text, status=status)


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
        # A garbled 200 is a provider glitch, so it is retried like other transient failures.
        raise TransientError(
            f"malformed completion from provider: {error!r}", status=response.status_code
        ) from error
