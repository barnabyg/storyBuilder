"""Versioned prompt templates.

A template lives at `<prompts_dir>/<name>/v<version>.yaml` with a `user`
message and an optional `system` message, using `$variable` placeholders.
`<prompts_dir>/checksums.json` records the SHA-256 of every released version.
Loading a template whose text no longer matches its checksum fails: changing a
template's text requires a new version. `nde prompts lock` records new versions.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from nde.files import checksum_lf, write_text_lf
from nde.llm.client import Message
from nde.runs.config import ConfigError

CHECKSUMS = "checksums.json"


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


def read_checksums(prompts_dir: Path) -> dict[str, str]:
    path = prompts_dir / CHECKSUMS
    if not path.is_file():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_template(prompts_dir: Path, name: str, version: int) -> PromptTemplate:
    label = f"{name}@{version}"
    path = template_path(prompts_dir, name, version)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"prompt template {label} not found at {path}") from error
    digest = checksum_lf(data)
    recorded = read_checksums(prompts_dir).get(label)
    if recorded is None:
        raise ConfigError(
            f"prompt template {label} is not registered; run `nde prompts lock` to record it"
        )
    if recorded != digest:
        raise ConfigError(
            f"prompt template {label} changed after release; save the new text as a new version"
        )
    try:
        parsed = _TemplateFile.model_validate(yaml.safe_load(data))
    except (yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"prompt template {label} is malformed: {error}") from error
    return PromptTemplate(name, version, parsed.system, parsed.user, digest)


def _versions(prompts_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(prompts_dir.glob("*/v*.yaml")):
        version = path.stem.removeprefix("v")
        if version.isdigit():
            found[f"{path.parent.name}@{int(version)}"] = checksum_lf(path.read_bytes())
    return found


def check(prompts_dir: Path) -> tuple[list[str], list[str]]:
    """Return (unregistered labels, changed labels) without writing anything."""
    checksums = read_checksums(prompts_dir)
    found = _versions(prompts_dir)
    unregistered = [label for label in found if label not in checksums]
    changed = [label for label, digest in found.items() if checksums.get(label, digest) != digest]
    return unregistered, changed


def lock(prompts_dir: Path) -> tuple[list[str], list[str]]:
    """Record checksums for new template versions.

    Returns (added labels, changed labels). A changed version is never
    re-recorded: restore its text, or save the new text as a new version.
    """
    added, changed = check(prompts_dir)
    if added:
        found = _versions(prompts_dir)
        checksums = read_checksums(prompts_dir) | {label: found[label] for label in added}
        write_text_lf(
            prompts_dir / CHECKSUMS, json.dumps(dict(sorted(checksums.items())), indent=2) + "\n"
        )
    return added, changed
