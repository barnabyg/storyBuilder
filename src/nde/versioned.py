"""Versioned files whose released text is fixed by recorded checksums.

Prompt templates (`prompts/<name>/v<n>.yaml`) and the rubric
(`benchmark/rubric/v<n>.yaml`) are released this way. A directory's
`checksums.json` records the SHA-256 of every released version, keyed
`<name>@<n>`. Changing a released version's text is rejected: save the new
text as a new version and record it with `lock`.
"""

import json
from pathlib import Path
from typing import Literal

from nde.files import checksum_lf, write_text_lf

CHECKSUMS = "checksums.json"

type Status = Literal["released", "unregistered", "changed"]


def read_checksums(directory: Path) -> dict[str, str]:
    path = directory / CHECKSUMS
    if not path.is_file():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def status(directory: Path, label: str, data: bytes) -> Status:
    """Whether `data` is the released text of `label`."""
    recorded = read_checksums(directory).get(label)
    if recorded is None:
        return "unregistered"
    return "released" if recorded == checksum_lf(data) else "changed"


def _versions(directory: Path, name: str | None) -> dict[str, str]:
    """Checksums of every version file: `<name>/v<n>.yaml`, or `v<n>.yaml` labelled `name`."""
    found: dict[str, str] = {}
    pattern = "*/v*.yaml" if name is None else "v*.yaml"
    for path in sorted(directory.glob(pattern)):
        version = path.stem.removeprefix("v")
        if version.isdigit():
            label = f"{name or path.parent.name}@{int(version)}"
            found[label] = checksum_lf(path.read_bytes())
    return found


def check(directory: Path, name: str | None = None) -> tuple[list[str], list[str]]:
    """Return (unregistered labels, changed labels) without writing anything."""
    checksums = read_checksums(directory)
    found = _versions(directory, name)
    unregistered = [label for label in found if label not in checksums]
    changed = [label for label, digest in found.items() if checksums.get(label, digest) != digest]
    return unregistered, changed


def lock(directory: Path, name: str | None = None) -> tuple[list[str], list[str]]:
    """Record checksums for new versions.

    Returns (added labels, changed labels). A changed version is never
    re-recorded: restore its text, or save the new text as a new version.
    """
    added, changed = check(directory, name)
    if added:
        found = _versions(directory, name)
        checksums = read_checksums(directory) | {label: found[label] for label in added}
        write_text_lf(
            directory / CHECKSUMS, json.dumps(dict(sorted(checksums.items())), indent=2) + "\n"
        )
    return added, changed
