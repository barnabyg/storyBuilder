"""The common provider error types every provider failure is mapped to.

| Type              | Retried | Typical cause                                      |
| ----------------- | ------- | -------------------------------------------------- |
| `rate_limited`    | yes     | HTTP 429                                           |
| `transient`       | yes     | HTTP 408/409/5xx, timeouts, lost connections, garbled replies |
| `authentication`  | no      | HTTP 401/403                                       |
| `invalid_request` | no      | other HTTP 4xx, unknown model, bad parameters      |
| `content_refused` | no      | provider content filter                            |
"""

from typing import ClassVar, Literal

type ErrorKind = Literal[
    "rate_limited", "transient", "authentication", "invalid_request", "content_refused"
]


class ProviderError(Exception):
    kind: ClassVar[ErrorKind]
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimitedError(ProviderError):
    kind = "rate_limited"
    retryable = True

    def __init__(
        self, message: str, *, status: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message, status=status)
        self.retry_after_s = retry_after_s


class TransientError(ProviderError):
    kind = "transient"
    retryable = True


class AuthenticationError(ProviderError):
    kind = "authentication"


class InvalidRequestError(ProviderError):
    kind = "invalid_request"


class ContentRefusedError(ProviderError):
    kind = "content_refused"
