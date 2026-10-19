"""Versioned prompt templates.

A template lives at `<prompts_dir>/<name>/v<version>.yaml` with a `user`
message and an optional `system` message, using `$variable` placeholders.
`<prompts_dir>/checksums.json` records the SHA-256 of every released version.
Loading a template whose text no longer matches its checksum fails: changing a
template's text requires a new version. `nde prompts lock` records new versions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from nde import versioned
from nde.files import checksum_lf
from nde.llm.client import Message
from nde.runs.config import ConfigError, PromptRef

CHECKSUMS = versioned.CHECKSUMS


class _TemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str | None = None
    user: str


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str | None
    user: str
    sha256: str

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, variables: Mapping[str, str]) -> tuple[Message, ...]:
        try:
            messages: list[Message] = []
            if self.system is not None:
                messages.append({"role": "system", "content": _substitute(self.system, variables)})
            messages.append({"role": "user", "content": _substitute(self.user, variables)})
        except KeyError as error:
            raise ConfigError(f"template {self.label} needs variable {error.args[0]!r}") from error
        return tuple(messages)


def _substitute(text: str, variables: Mapping[str, str]) -> str:
    return Template(text).substitute(variables)


def template_path(prompts_dir: Path, name: str, version: int) -> Path:
    return prompts_dir / name / f"v{version}.yaml"


def load_template(prompts_dir: Path, name: str, version: int) -> PromptTemplate:
    label = f"{name}@{version}"
    path = template_path(prompts_dir, name, version)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"prompt template {label} not found at {path}") from error
    released = versioned.status(prompts_dir, label, data)
    if released == "unregistered":
        raise ConfigError(
            f"prompt template {label} is not registered; run `nde prompts lock` to record it"
        )
    if released == "changed":
        raise ConfigError(
            f"prompt template {label} changed after release; save the new text as a new version"
        )
    try:
        parsed = _TemplateFile.model_validate(yaml.safe_load(data))
    except (yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"prompt template {label} is malformed: {error}") from error
    return PromptTemplate(name, version, parsed.system, parsed.user, checksum_lf(data))


def require_template(
    templates: dict[str, PromptTemplate],
    prompts_dir: Path,
    ref: PromptRef,
    variables: Mapping[str, str],
) -> PromptTemplate:
    """Load `ref` into `templates` once and check it renders with `variables`.

    Raises ConfigError, so configs fail before any call. Callers that only know
    variable names pass placeholder values: `dict.fromkeys(names, "x")`.
    """
    if ref.label not in templates:
        templates[ref.label] = load_template(prompts_dir, ref.template, ref.version)
    templates[ref.label].render(variables)
    return templates[ref.label]


def check(prompts_dir: Path) -> tuple[list[str], list[str]]:
    """Return (unregistered labels, changed labels) without writing anything."""
    return versioned.check(prompts_dir)


def lock(prompts_dir: Path) -> tuple[list[str], list[str]]:
    """Record checksums for new template versions; see `nde.versioned.lock`."""
    return versioned.lock(prompts_dir)
