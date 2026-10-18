"""Gateway configuration: providers and the model bound to each role."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(Strict):
    base_url: str
    # Environment variable that, when set, overrides base_url.
    base_url_env: str | None = None
    # Environment variable holding the API key. None for servers that need no key.
    api_key_env: str | None = None
    timeout_s: float = Field(default=60.0, gt=0)

    def resolved_base_url(self, environ: Mapping[str, str]) -> str:
        if self.base_url_env and environ.get(self.base_url_env):
            return environ[self.base_url_env]
        return self.base_url


class ModelBinding(Strict):
    provider: str
    model: str
    params: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"
