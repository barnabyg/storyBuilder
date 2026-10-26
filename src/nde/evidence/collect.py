"""Capture local inputs only; report generation never downloads or calls a model."""

import base64
from pathlib import Path

from nde.annotations.corrections import correction_seconds
from nde.annotations.records import Session, digest
from nde.corpus import provenance
from nde.gitinfo import git
from nde.narrative.workflow import schema_json

from .mutations import load_suite
from .records import AnnotationInput, Snapshot, StoryInput


def _annotation(directory: Path) -> AnnotationInput:
    if not (directory / "session.json").is_file():
        return AnnotationInput()
    captured = AnnotationInput(session=(directory / "session.json").read_text(encoding="utf-8"))
    try:
        session = Session.model_validate_json(captured.session or "")
        names = ["draft/plan.json", "original.txt"]
        for revision in session.revisions:
            expected = f"revisions/{revision.number:04d}/submitted.json"
            if revision.file != expected or revision.number < 1:
                raise ValueError("invalid correction revision path")
            names.append(expected)
        for name in names:
            file = directory / name
            if file.is_file():
                captured.files[name] = file.read_text(encoding="utf-8")
        if (directory / "timer.jsonl").is_file():
            captured.timer = (directory / "timer.jsonl").read_text(encoding="utf-8")
        captured.measured_seconds = correction_seconds(directory)
    except (OSError, ValueError) as error:
        captured.capture_errors.append(str(error))
    return captured


def collect(manifest: Path, annotations: Path, suite: Path) -> Snapshot:
    records = provenance.read_manifest(manifest).stories
    names = [story.id for story in records]
    if len(set(names)) != len(names):
        raise ValueError("provenance story IDs must be unique")
    inputs: list[StoryInput] = []
    for record in records:
        story = StoryInput(provenance=record)
        source = provenance.source_path(manifest, record.id)
        if source.is_file():
            story.source_base64 = base64.b64encode(source.read_bytes()).decode("ascii")
        else:
            story.source_error = "source not acquired locally"
        story.annotation = _annotation(annotations / record.id)
        inputs.append(story)
    return Snapshot(
        schema_text=schema_json(),
        git_sha=git(["rev-parse", "HEAD"], manifest.parent) or "unknown",
        manifest_sha256=digest(manifest.read_text(encoding="utf-8")),
        suite=load_suite(suite),
        stories=inputs,
    )
