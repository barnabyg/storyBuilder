"""Individually acquired works; raw bytes and canonical LF text are immutable evidence."""

import hashlib
import html
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree

import httpx
import yaml
from pydantic import Field

from nde.files import write_text_lf
from nde.llm.config import Strict


class CorpusError(ValueError):
    """Invalid, changed, or incomplete corpus evidence."""


class WorkMetadata(Strict):
    version: Literal["1"] = "1"
    work_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    source_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    title: str = Field(min_length=1)
    author_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    source: str = Field(min_length=1)
    edition: str = Field(min_length=1)
    licence_basis: str = Field(min_length=1)
    licence_evidence: str = Field(min_length=1)
    period: str = "unknown"
    genre: str = "unknown"
    format: Literal["text", "xhtml"] = "text"
    start: str | None = "*** START OF"
    end: str | None = "*** END OF"
    section_id: str | None = None


class WorkSpec(WorkMetadata):
    raw_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class Selection(Strict):
    basis: Literal["decoded-source-codepoints", "xhtml-id"]
    start: int | None = None
    end: int | None = None
    section_id: str | None = None


class WorkRecord(WorkMetadata):
    raw_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_version: Literal["corpus-text@1"] = "corpus-text@1"
    selection: Selection


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_digest(value: Strict) -> str:
    return sha256(json.dumps(value.model_dump(mode="json"), sort_keys=True).encode())


def _text(raw: str, spec: WorkMetadata) -> tuple[str, Selection]:
    if not spec.start or not spec.end:
        raise CorpusError("plain text requires explicit start and end markers")
    starts = list(re.finditer(re.escape(spec.start), raw))
    ends = list(re.finditer(re.escape(spec.end), raw))
    if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
        raise CorpusError("selection markers must occur exactly once in source order")
    left = raw.find("\n", starts[0].end())
    right = raw.rfind("\n", 0, ends[0].start())
    if left < 0 or right < left:
        raise CorpusError("selection markers contain no work")
    body = raw[left + 1 : right].replace("\r\n", "\n").replace("\r", "\n")
    blocks = [" ".join(block.split()) for block in re.split(r"\n\s*\n", body) if block.strip()]
    while blocks and (
        blocks[0].casefold() in {spec.title.casefold(), f"by {spec.author.casefold()}"}
        or blocks[0].casefold().startswith(("produced by ", "transcribed by "))
    ):
        blocks.pop(0)
    return "\n\n".join(blocks) + "\n", Selection(
        basis="decoded-source-codepoints", start=left + 1, end=right
    )


def _xhtml(raw: str, spec: WorkMetadata) -> tuple[str, Selection]:
    # ElementTree does not load external entities; reject declarations as well to
    # exclude internal expansion. Revisit if declared-DTD sources become supported.
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", raw, re.IGNORECASE):
        raise CorpusError("XHTML entity/DTD declarations are unsupported")
    try:
        root = ElementTree.fromstring(raw)  # noqa: S314 - declarations rejected above
    except ElementTree.ParseError as error:
        raise CorpusError(f"malformed XHTML: {error}") from error
    if spec.section_id is None:
        raise CorpusError("XHTML requires section_id to select the work, excluding front matter")
    matches = [element for element in root.iter() if element.get("id") == spec.section_id]
    if len(matches) != 1:
        raise CorpusError("XHTML selection id must identify exactly one work")
    blocks: list[str] = []
    for element in matches[0].iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "hr":
            blocks.append("***")
        elif tag in {"p", "h1", "h2", "h3", "h4"}:
            blocks.append(" ".join("".join(element.itertext()).split()))
    return "\n\n".join(block for block in blocks if block) + "\n", Selection(
        basis="xhtml-id", section_id=spec.section_id
    )


def extract(data: bytes, spec: WorkMetadata) -> tuple[str, Selection]:
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CorpusError("source must be valid UTF-8") from error
    if "\x00" in raw:
        raise CorpusError("source contains NUL bytes")
    text, selection = _text(raw, spec) if spec.format == "text" else _xhtml(raw, spec)
    if not text.strip():
        raise CorpusError("selected work is empty")
    return text, selection


def load_work(directory: Path) -> tuple[WorkRecord, str]:
    record = WorkRecord.model_validate_json((directory / "work.json").read_text("utf-8"))
    data = (directory / "source.raw").read_bytes()
    saved = (directory / "text.txt").read_bytes()
    if sha256(data) != record.raw_sha256 or sha256(saved) != record.text_sha256:
        raise CorpusError("changed recorded source or canonical text checksum")
    text, selection = extract(data, record)
    if text.encode() != saved or selection != record.selection:
        raise CorpusError("recorded extraction or selection differs from verified source")
    return record, text


def _report(record: WorkRecord, text: str) -> str:
    details = html.escape(record.model_dump_json(indent=2))
    return (
        "<!doctype html><meta charset='utf-8'><title>Corpus provenance</title>"
        f"<h1>{html.escape(record.title)}</h1><p>Recorded provenance does not establish legal "
        f"eligibility.</p><pre>{details}</pre><h2>Canonical text</h2><pre>{html.escape(text)}</pre>"
    )


def acquire(spec_file: Path, out: Path, local_file: Path | None = None) -> WorkRecord:
    spec = WorkSpec.model_validate(yaml.safe_load(spec_file.read_text("utf-8")))
    if out.exists():
        record, _ = load_work(out)
        expected = spec.model_dump(exclude={"raw_sha256"})
        if any(getattr(record, key) != value for key, value in expected.items()):
            raise CorpusError("duplicate work/source identity: existing record differs")
        if spec.raw_sha256 is not None and spec.raw_sha256 != record.raw_sha256:
            raise CorpusError("changed expected source checksum")
        if local_file is not None and sha256(local_file.read_bytes()) != record.raw_sha256:
            raise CorpusError("changed imported bytes cannot overwrite verified source")
        return record
    if local_file is not None:
        data = local_file.read_bytes()
    else:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get(spec.source)
            response.raise_for_status()
            data = response.content
    raw_digest = sha256(data)
    if spec.raw_sha256 is not None and spec.raw_sha256 != raw_digest:
        raise CorpusError("source checksum differs from recorded expected bytes")
    text, selection = extract(data, spec)
    record = WorkRecord.model_validate(
        spec.model_dump()
        | {"raw_sha256": raw_digest, "text_sha256": sha256(text.encode()), "selection": selection}
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (staging / "source.raw").write_bytes(data)
        write_text_lf(staging / "text.txt", text)
        write_text_lf(staging / "work.json", record.model_dump_json(indent=2) + "\n")
        write_text_lf(staging / "report.html", _report(record, text))
        load_work(staging)
        staging.rename(out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return record
