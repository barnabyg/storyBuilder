"""Inspectable explicit boundaries on exact LF Unicode-codepoint source text."""

import html
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from nde.corpus.acquisition import CorpusError, json_digest, load_work, sha256
from nde.corpus.artifacts import publish
from nde.llm.config import Strict

_CHAPTER = re.compile(r"(?im)^(?:chapter|book|part)\b[^\n]*$")
_BREAK = re.compile(r"(?m)^[ \t]*(?:\*[ \t]*){3,}$|^[ \t]*[-_]{3,}[ \t]*$")


class Boundary(Strict):
    id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    parent_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str


class Segmentation(Strict):
    version: Literal["1"] = "1"
    work_id: str
    source_id: str
    raw_sha256: str
    text_sha256: str
    method: Literal["explicit-breaks@1"] = "explicit-breaks@1"
    offset_basis: Literal["unicode-codepoints-lf"] = "unicode-codepoints-lf"
    gap_policy: Literal["whitespace-and-explicit-break-lines"] = (
        "whitespace-and-explicit-break-lines"
    )
    chapters: list[Boundary]
    scenes: list[Boundary]
    sha256: str


def _gap(text: str) -> bool:
    return not _BREAK.sub("", text).strip()


def _validate_intervals(boundaries: list[Boundary], text: str) -> None:
    cursor = 0
    identities: set[str] = set()
    for boundary in boundaries:
        if boundary.id in identities or not cursor <= boundary.start < boundary.end <= len(text):
            raise CorpusError("invalid boundary identity, bounds, order or overlap")
        if not _gap(text[cursor : boundary.start]):
            raise CorpusError("uncovered source text between boundaries")
        cursor = boundary.end
        identities.add(boundary.id)
    if not boundaries or not _gap(text[cursor:]):
        raise CorpusError("boundaries do not cover source text")


def validate_segments(segments: Segmentation, text: str) -> None:
    _validate_intervals(segments.chapters, text)
    _validate_intervals(segments.scenes, text)
    parents = {chapter.id: chapter for chapter in segments.chapters}
    for scene in segments.scenes:
        parent = parents.get(scene.parent_id or "")
        if parent is None or not parent.start <= scene.start < scene.end <= parent.end:
            raise CorpusError("scene parent is missing or does not contain its scene")


def _segment(text: str, work_id: str) -> tuple[list[Boundary], list[Boundary]]:
    starts = sorted({0, *(match.start() for match in _CHAPTER.finditer(text))})
    chapters: list[Boundary] = []
    scenes: list[Boundary] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chapter = Boundary(
            id=f"{work_id}.chapter-{index + 1}",
            start=start,
            end=end,
            confidence=1 if _CHAPTER.match(text, start) else None,
            reason="explicit chapter heading"
            if _CHAPTER.match(text, start)
            else "unmarked chapter",
        )
        chapters.append(chapter)
        breaks = list(_BREAK.finditer(text, start, end))
        cursor = start
        for stop, following in [(m.start(), m.end()) for m in breaks] + [(end, end)]:
            if text[cursor:stop].strip():
                scenes.append(
                    Boundary(
                        id=f"{work_id}.scene-{len(scenes) + 1}",
                        start=cursor,
                        end=stop,
                        parent_id=chapter.id,
                        confidence=1 if breaks else None,
                        reason="explicit scene-break line"
                        if breaks
                        else "unmarked scene; transitions remain uncertain",
                    )
                )
            cursor = following
    return chapters, scenes


def segment_work(work_dir: Path, out: Path) -> Segmentation:
    work, text = load_work(work_dir)
    chapters, scenes = _segment(text, work.work_id)
    value = Segmentation(
        work_id=work.work_id,
        source_id=work.source_id,
        raw_sha256=work.raw_sha256,
        text_sha256=work.text_sha256,
        chapters=chapters,
        scenes=scenes,
        sha256="",
    )
    value = value.model_copy(update={"sha256": json_digest(value)})
    validate_segments(value, text)
    cards = "".join(
        f"<h2>{html.escape(s.id)} [{s.start}, {s.end})</h2>"
        f"<p>{html.escape(s.reason)}; confidence {s.confidence}</p>"
        f"<pre>{html.escape(text[s.start : s.end])}</pre>"
        for s in scenes
    )
    publish(
        out,
        {
            "segmentation.json": value.model_dump_json(indent=2) + "\n",
            "report.html": "<!doctype html><meta charset='utf-8'><h1>Source boundaries</h1>"
            "<p>Offsets: Unicode codepoints on exact canonical LF text. Whitespace and explicit "
            "break lines may be gaps. Unmarked narrative transitions are not inferred.</p>" + cards,
        },
    )
    return value


def load_segmentation(path: Path, work_dir: Path) -> Segmentation:
    file = path / "segmentation.json" if path.is_dir() else path
    value = Segmentation.model_validate_json(file.read_text("utf-8"))
    work, text = load_work(work_dir)
    if (value.work_id, value.source_id, value.raw_sha256, value.text_sha256) != (
        work.work_id,
        work.source_id,
        work.raw_sha256,
        work.text_sha256,
    ):
        raise CorpusError("segmentation source identity is stale")
    validate_segments(value, text)
    if json_digest(value.model_copy(update={"sha256": ""})) != value.sha256:
        raise CorpusError("changed segmentation checksum")
    return value


class GoldSegmentation(Strict):
    version: Literal["1"] = "1"
    work_id: str
    text_sha256: str
    segmentation_sha256: str
    approved: bool
    owner: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)
    synthetic: bool = False
    tolerance_codepoints: int = Field(default=0, ge=0)
    scene_starts: list[int]


class BoundaryComparison(Strict):
    version: Literal["1"] = "1"
    work_id: str
    text_sha256: str
    segmentation_sha256: str
    gold_sha256: str | None
    status: Literal["current-owner-reference", "pending"]
    reason: str
    tolerance_codepoints: int | None
    matching: Literal["one-to-one-nearest-excluding-work-start"] = (
        "one-to-one-nearest-excluding-work-start"
    )
    matched: int | None
    predicted: int
    reference: int | None
    precision: float | None
    recall: float | None


def compare_segmentation(
    work_dir: Path, segmentation_path: Path, gold_path: Path | None, out: Path
) -> BoundaryComparison:
    segments = load_segmentation(segmentation_path, work_dir)
    _, text = load_work(work_dir)
    predicted = [scene.start for scene in segments.scenes if scene.start]
    gold = (
        GoldSegmentation.model_validate_json(gold_path.read_text("utf-8"))
        if gold_path and gold_path.is_file()
        else None
    )
    matched: int | None = None
    reference: list[int] | None = None
    reason = "missing gold reference"
    status: Literal["current-owner-reference", "pending"] = "pending"
    if gold is not None:
        if gold.work_id != segments.work_id or gold.text_sha256 != segments.text_sha256:
            reason = "stale gold source identity"
        else:
            if (
                sorted(set(gold.scene_starts)) != gold.scene_starts
                or not gold.scene_starts
                or gold.scene_starts[0] != 0
                or gold.scene_starts[-1] >= len(text)
            ):
                raise CorpusError(
                    "gold boundary starts must be unique ordered in-bounds and include zero"
                )
            reference = [start for start in gold.scene_starts if start]
            available = set(reference)
            matched = 0
            for start in predicted:
                candidates = sorted(available, key=lambda end: (abs(start - end), end))
                if candidates and abs(start - candidates[0]) <= gold.tolerance_codepoints:
                    matched += 1
                    available.remove(candidates[0])
            reason = (
                "synthetic reference is test-only"
                if gold.synthetic
                else "unapproved draft gold reference"
                if not gold.approved
                else "current owner-approved source-linked reference"
            )
            if gold.approved and not gold.synthetic:
                status = "current-owner-reference"
    value = BoundaryComparison(
        work_id=segments.work_id,
        text_sha256=segments.text_sha256,
        segmentation_sha256=segments.sha256,
        gold_sha256=sha256(gold_path.read_bytes()) if gold_path and gold_path.is_file() else None,
        status=status,
        reason=reason,
        tolerance_codepoints=gold.tolerance_codepoints if gold else None,
        matched=matched,
        predicted=len(predicted),
        reference=len(reference) if reference is not None else None,
        precision=matched / len(predicted) if matched is not None and predicted else None,
        recall=matched / len(reference) if matched is not None and reference else None,
    )
    publish(
        out,
        {
            "comparison.json": value.model_dump_json(indent=2) + "\n",
            "report.html": "<!doctype html><meta charset='utf-8'><h1>Boundary comparison</h1><pre>"
            + html.escape(value.model_dump_json(indent=2))
            + "</pre>",
        },
    )
    return value
