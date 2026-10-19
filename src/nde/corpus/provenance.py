"""Human-written stories with provenance records, for calibration (P1.9).

The provenance manifest (`data/manifest.calibration.yaml`) is committed; the
texts it describes live under the gitignored `data/calibration/`, following
the data manifest convention (`data/README.md`). Each record names the source,
edition, licence basis, author and the SHA-256 of the source file.

`fetch` downloads each source (or reuses the copy already downloaded), checks
it against the recorded checksum, or records the checksum on first fetch, and
writes the story text: extracted from between the source's start and end
markers, unwrapped into paragraphs, stripped of title and credit lines, and
normalised exactly like a generated story.
"""

import hashlib
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nde.files import write_text_lf
from nde.stories.normalise import normalise

DEFAULT_MANIFEST = Path("data") / "manifest.calibration.yaml"
TEXTS_DIR = "calibration"

# Project Gutenberg plain-text markers around the work itself.
GUTENBERG_START = "*** START OF"
GUTENBERG_END = "*** END OF"

_HEADER = """\
# Provenance of the human-written calibration stories (docs/evaluation.md).
#
# Only this manifest is committed; the texts are fetched into data/calibration/
# (gitignored) with `uv run nde calibration fetch`. The first fetch records each
# source's SHA-256 here; later fetches verify it. Commit the recorded checksums.
#
# Each story's text is taken from between the line containing `start` and the
# line containing `end` (Project Gutenberg's markers by default).
"""

# Short leading lines that are front matter, not story: titles, bylines, credits.
_CREDIT_PREFIXES = ("by ", "produced by", "transcribed by", "transcriber", "e-text prepared")
_MAX_FRONT_MATTER_WORDS = 12
_TIMEOUT_S = 60.0


class ProvenanceError(Exception):
    """A calibration story cannot be fetched, verified or extracted."""


class HumanStory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    author: str = Field(min_length=1)
    source: str = Field(min_length=1, description="where the source file is downloaded from")
    edition: str = Field(min_length=1)
    licence_basis: str = Field(min_length=1)
    # SHA-256 of the source file; recorded by the first fetch.
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    start: str = GUTENBERG_START
    end: str = GUTENBERG_END


class ProvenanceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stories: list[HumanStory] = Field(min_length=1)


def read_manifest(path: Path) -> ProvenanceManifest:
    try:
        return ProvenanceManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ProvenanceError(f"cannot read the provenance manifest {path}: {error}") from error


def write_manifest(path: Path, manifest: ProvenanceManifest) -> None:
    data = manifest.model_dump(mode="json")
    write_text_lf(path, _HEADER + "\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def texts_dir(manifest_path: Path) -> Path:
    return manifest_path.parent / TEXTS_DIR


def source_path(manifest_path: Path, story_id: str) -> Path:
    return texts_dir(manifest_path) / f"{story_id}.source.txt"


def story_path(manifest_path: Path, story_id: str) -> Path:
    return texts_dir(manifest_path) / f"{story_id}.md"


def _is_front_matter(paragraph: str) -> bool:
    lower = paragraph.lower()
    if len(paragraph.split()) > _MAX_FRONT_MATTER_WORDS:
        return False
    return lower.startswith(_CREDIT_PREFIXES) or not paragraph.rstrip().endswith(tuple(".!?\"'"))


def extract(text: str, story: HumanStory) -> str:
    """The story's prose from a source file, normalised like a generated story."""
    lines = text.replace("\r\n", "\n").split("\n")
    starts = [i for i, line in enumerate(lines) if story.start in line]
    ends = [i for i, line in enumerate(lines) if story.end in line]
    if not starts or not ends or ends[-1] <= starts[0]:
        raise ProvenanceError(
            f"{story.id}: the source has no text between {story.start!r} and {story.end!r}"
        )
    body = "\n".join(lines[starts[0] + 1 : ends[-1]])
    # Sources are hard-wrapped: a paragraph is a block of lines between blank lines.
    paragraphs = [" ".join(block.split()) for block in body.split("\n\n") if block.strip()]
    while paragraphs and _is_front_matter(paragraphs[0]):
        paragraphs.pop(0)
    if not paragraphs:
        raise ProvenanceError(f"{story.id}: nothing is left after the front matter")
    return normalise("\n\n".join(paragraphs))


def _checked(story: HumanStory, data: bytes) -> HumanStory:
    digest = hashlib.sha256(data).hexdigest()
    if story.sha256 is not None and story.sha256 != digest:
        raise ProvenanceError(
            f"{story.id}: checksum mismatch: the source's SHA-256 is {digest}, but the manifest "
            f"records {story.sha256}. The source changed: check it, then update the record."
        )
    return story.model_copy(update={"sha256": digest})


def _download(url: str) -> bytes:
    try:
        response = httpx.get(
            url, follow_redirects=True, timeout=_TIMEOUT_S, headers={"User-Agent": "nde/0.1"}
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ProvenanceError(f"cannot download {url}: {error}") from error
    return response.content


def _store(manifest_path: Path, story: HumanStory, data: bytes) -> HumanStory:
    checked = _checked(story, data)
    text = extract(data.decode("utf-8-sig"), checked)
    texts_dir(manifest_path).mkdir(parents=True, exist_ok=True)
    source_path(manifest_path, story.id).write_bytes(data)
    write_text_lf(story_path(manifest_path, story.id), text)
    return checked


def fetch(manifest_path: Path) -> list[HumanStory]:
    """Fetch (or reuse and verify) every story; records new checksums in the manifest."""
    manifest = read_manifest(manifest_path)
    stored: list[HumanStory] = []
    try:
        for story in manifest.stories:
            local = source_path(manifest_path, story.id)
            data = local.read_bytes() if local.is_file() else _download(story.source)
            stored.append(_store(manifest_path, story, data))
    finally:
        done = {story.id: story for story in stored}
        manifest.stories = [done.get(story.id, story) for story in manifest.stories]
        write_manifest(manifest_path, manifest)
    return stored


def import_story(manifest_path: Path, story_id: str, file: Path) -> HumanStory:
    """Use a local copy of a story's source instead of downloading it."""
    manifest = read_manifest(manifest_path)
    matches = [story for story in manifest.stories if story.id == story_id]
    if not matches:
        raise ProvenanceError(f"no story {story_id!r} in {manifest_path}")
    stored = _store(manifest_path, matches[0], file.read_bytes())
    manifest.stories = [stored if s.id == story_id else s for s in manifest.stories]
    write_manifest(manifest_path, manifest)
    return stored


def read_texts(manifest_path: Path) -> dict[str, tuple[HumanStory, str]]:
    """Every fetched story's record and normalised text, by ID."""
    manifest = read_manifest(manifest_path)
    texts: dict[str, tuple[HumanStory, str]] = {}
    for story in manifest.stories:
        path = story_path(manifest_path, story.id)
        if not path.is_file() or story.sha256 is None:
            raise ProvenanceError(
                f"{story.id} has not been fetched; run `uv run nde calibration fetch`"
            )
        texts[story.id] = (story, path.read_text(encoding="utf-8"))
    return texts
