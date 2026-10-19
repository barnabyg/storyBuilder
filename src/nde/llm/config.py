"""Gateway configuration: providers and the model bound to each role."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetryConfig(Strict):
    """Retries for rate-limited and transient failures: exponential backoff, full jitter.

    Retry n (from 0) waits a random time up to min(max_delay_s, base_delay_s * 2**n),
    and never less than a rate limit's Retry-After (still capped at max_delay_s).
    """

    max_retries: int = Field(default=3, ge=0)
    base_delay_s: float = Field(default=1.0, ge=0)
    max_delay_s: float = Field(default=30.0, ge=0)


class ProviderConfig(Strict):
    base_url: str
    # Environment variable that, when set, overrides base_url.
    base_url_env: str | None = None
    # Environment variable holding the API key. None for servers that need no key.
    api_key_env: str | None = None
    timeout_s: float = Field(default=60.0, gt=0)
    # Whether the provider supports JSON-schema response mode (`response_format`).
    json_schema: bool = False
    retry: RetryConfig = Field(default_factory=RetryConfig)

    def resolved_base_url(self, environ: Mapping[str, str]) -> str:
        if self.base_url_env and environ.get(self.base_url_env):
            return environ[self.base_url_env]
        return self.base_url


class ModelBinding(Strict):
    provider: str
    model: str
    params: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    # Overrides the provider's json_schema capability for this model.
    json_schema: bool | None = None
    # The model family (e.g. "openai", "deepseek"), recorded per model rather than inferred
    # from the provider, for the rule that judges come from other families than generators.
    family: str | None = Field(default=None, min_length=1)

    @field_validator("family")
    @classmethod
    def _family_is_case_insensitive(cls, family: str | None) -> str | None:
        """Families compare in lower case, so "OpenAI" and "openai" are one family."""
        return None if family is None else family.strip().lower()

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"
