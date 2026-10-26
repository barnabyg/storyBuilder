"""Owner realisation evidence bound to the final selected prose, never a draft."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from nde.annotations.records import digest
from nde.files import write_text_lf
from nde.narrative.base import Strict
from nde.narrative.models import NarrativePlan
from nde.narrative.validation import validate_plan
from nde.narrative.workflow import load_plan
from nde.plan.beats import BeatOutline
from nde.runs.records import STORY_FILE, StoryRecord, read_stories
from nde.stories.normalise import normalise


class Intent(Strict):
    id: str
    scene_ids: list[str]
    channel: str
    intended: dict[str, Any]


class Packet(Strict):
    version: Literal["1"] = "1"
    packet_sha256: str
    run: str
    story: StoryRecord
    plan_sha256: str
    plan_kind: Literal["narrative", "beat_outline"] = "narrative"
    story_sha256: str
    prose: str
    items: list[Intent]
    alignment: Literal["scene-aligned", "unresolved"]
    alignment_note: str


class Assessment(Strict):
    item_id: str
    rating: Literal["realised", "not_realised", "unknown"]
    note: str = Field(min_length=1)
    # Optional final-prose Unicode offsets; an exact quote is required with offsets.
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=1)
    quote: str | None = None


class Review(Strict):
    packet_sha256: str
    reviewer: str = Field(min_length=1)
    test_only: bool = False
    assessments: list[Assessment]
    # Estimates are kept separately and never enter the owner denominator.
    diagnostic_provenance: dict[str, str] | None = None


def intentions(plan: NarrativePlan) -> list[Intent]:
    rows: list[Intent] = []
    for channel, records in (
        ("belief", [s for s in plan.states if s.kind == "belief"]),
        ("relationship", plan.relationships),
        ("reveal", plan.reveal_links),
    ):
        rows.extend(
            Intent(
                id=f"{channel}:{r.id}",
                scene_ids=[r.scene_id],
                channel=channel,
                intended=r.model_dump(mode="json"),
            )
            for r in records
        )
    rows.extend(
        Intent(
            id=f"tension:{r.id}",
            scene_ids=r.scene_ids,
            channel="tension",
            intended=r.model_dump(mode="json"),
        )
        for r in plan.tension
    )
    return rows


def _record(run: Path, condition: str, brief: str, repetition: int) -> StoryRecord:
    rows = [
        r
        for r in read_stories(run)
        if (r.condition, r.brief, r.repetition) == (condition, brief, repetition)
    ]
    if len(rows) != 1 or rows[0].status != "ok":
        raise ValueError("realisation requires exactly one successful selected story")
    return rows[0]


def _planning(folder: Path, story: StoryRecord) -> tuple[str, list[Intent], int]:
    if story.pipeline == "baseline_b":
        outline = BeatOutline.model_validate_json(
            (folder / "outline.json").read_text(encoding="utf-8")
        )
        return digest(outline.model_dump_json()), [], len(outline.scenes)
    plan = load_plan(folder / "plan.json")
    if not validate_plan(plan).valid:
        raise ValueError("realisation requires a validated plan")
    return digest(plan.model_dump_json()), intentions(plan), len(plan.scenes)


def create(run: Path, condition: str, brief: str, repetition: int, out: Path) -> Packet:
    if out.exists():
        raise ValueError("packet already exists; create a new packet to preserve history")
    story = _record(run, condition, brief, repetition)
    folder = run / story.path
    plan_sha256, items, scene_count = _planning(folder, story)
    prose = (folder / STORY_FILE).read_text(encoding="utf-8")
    scenes = sorted((folder / f"attempts/{(story.chosen_attempt or 0) + 1}").glob("scene-*.md"))
    assembled = normalise(
        "\n\n".join(normalise(p.read_text(encoding="utf-8")).strip() for p in scenes)
    )
    aligned = not story.revision_passes and len(scenes) == scene_count and assembled == prose
    packet = Packet(
        run=str(run.resolve()),
        story=story,
        plan_sha256=plan_sha256,
        plan_kind="beat_outline" if story.pipeline == "baseline_b" else "narrative",
        story_sha256=digest(prose),
        packet_sha256="",
        prose=prose,
        items=items,
        alignment="scene-aligned" if aligned else "unresolved",
        alignment_note="Selected scene outputs match final prose."
        if aligned
        else "Final prose has no reliable scene alignment; owner assessment is required.",
    )
    packet.packet_sha256 = digest(packet.model_dump_json(exclude={"packet_sha256"}))
    out.mkdir(parents=True)
    write_text_lf(out / "packet.json", packet.model_dump_json(indent=2) + "\n")
    lines = [
        "# Realisation review",
        f"Final story SHA-256: {packet.story_sha256}",
        f"Plan SHA-256: {packet.plan_sha256}",
        packet.alignment_note,
        "Owner assessments pending. Planned intentions do not establish literary accuracy.",
    ]
    lines.extend(f"{i.id} scenes {i.scene_ids}: {json.dumps(i.intended)}" for i in packet.items)
    write_text_lf(out / "packet.md", "\n\n".join(lines) + "\n\n## Final prose\n\n" + prose)
    return packet


def read_packet(folder: Path) -> Packet:
    packet = Packet.model_validate_json((folder / "packet.json").read_text(encoding="utf-8"))
    if packet.packet_sha256 != digest(packet.model_dump_json(exclude={"packet_sha256"})) or (
        digest(packet.prose) != packet.story_sha256
    ):
        raise ValueError("packet identity/prose checksum mismatch")
    return packet


def stale_reasons(packet: Packet) -> list[str]:
    try:
        run = Path(packet.run)
        current = _record(run, packet.story.condition, packet.story.brief, packet.story.repetition)
        folder = run / current.path
        checks = {
            "story record changed": current != packet.story,
            "final selected prose changed": digest(
                (folder / STORY_FILE).read_text(encoding="utf-8")
            )
            != packet.story_sha256,
            "plan changed": _planning(folder, current)[0] != packet.plan_sha256,
        }
        return [name for name, changed in checks.items() if changed]
    except (OSError, ValueError) as error:
        return [f"source evidence unavailable: {error}"]


def assess(folder: Path, review: Review) -> None:
    packet = read_packet(folder)
    if review.packet_sha256 != packet.packet_sha256 or stale_reasons(packet):
        raise ValueError("assessment identity is stale or does not match the selected prose/plan")
    known = {item.id for item in packet.items}
    ids = [item.item_id for item in review.assessments]
    if len(set(ids)) != len(ids) or set(ids) - known:
        raise ValueError("assessment items must be distinct known intentions")
    for item in review.assessments:
        if any(value is not None for value in (item.start, item.end, item.quote)) and (
            item.start is None
            or item.end is None
            or item.quote is None
            or item.end <= item.start
            or item.end > len(packet.prose)
            or packet.prose[item.start : item.end] != item.quote
        ):
            raise ValueError("span/quote must exactly match the final selected prose")
    history = folder / "history"
    history.mkdir(exist_ok=True)
    # Sequence preserves successive owner decisions; digest makes accidental overwrite impossible.
    path = (
        history
        / f"{len(list(history.glob('*.json'))) + 1:06d}-{digest(review.model_dump_json())}.json"
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(review.model_dump_json(indent=2) + "\n")


def report(folder: Path) -> dict[str, Any]:
    packet = read_packet(folder)
    stale = stale_reasons(packet)
    actual: dict[str, Assessment] = {}
    synthetic: dict[str, Assessment] = {}
    diagnostics = 0
    for path in sorted((folder / "history").glob("*.json")):
        review = Review.model_validate_json(path.read_text(encoding="utf-8"))
        if review.packet_sha256 != packet.packet_sha256 or stale:
            continue
        if review.diagnostic_provenance is not None:
            diagnostics += len(review.assessments)
        else:
            target = synthetic if review.test_only else actual
            target.update({item.item_id: item for item in review.assessments})
    known = sum(item.rating != "unknown" for item in actual.values())
    result: dict[str, Any] = {
        "packet_sha256": packet.packet_sha256,
        "story_sha256": packet.story_sha256,
        "plan_sha256": packet.plan_sha256,
        "sample": [packet.story.brief],
        "denominator": len(packet.items),
        "assessed": known,
        "missing": len(packet.items) - len(actual),
        "unknown": sum(item.rating == "unknown" for item in actual.values()),
        "realised": sum(item.rating == "realised" for item in actual.values()),
        "owner_status": "complete" if packet.items and known == len(packet.items) else "pending",
        "alignment": packet.alignment,
        "plan_kind": packet.plan_kind,
        "channel_note": "No belief/relationship/reveal/tension channels are represented."
        if not packet.items
        else "Assess the listed represented intentions.",
        "stale": stale,
        "synthetic_test_only_assessments": len(synthetic),
        "diagnostic_only_estimates": diagnostics,
    }
    write_text_lf(folder / "report.json", json.dumps(result, indent=2) + "\n")
    write_text_lf(
        folder / "report.md",
        "# Realisation evidence\n\n"
        + "\n\n".join(f"{key}: {value}" for key, value in result.items())
        + "\n",
    )
    return result


def configure(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="realisation_command", required=True)
    build = commands.add_parser("create")
    build.add_argument("run", type=Path)
    build.add_argument("--condition", required=True)
    build.add_argument("--brief", required=True)
    build.add_argument("--repetition", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)
    submit = commands.add_parser("assess")
    submit.add_argument("packet", type=Path)
    submit.add_argument("review", type=Path)
    show = commands.add_parser("report")
    show.add_argument("packet", type=Path)


def run(args: argparse.Namespace) -> int:
    try:
        if args.realisation_command == "create":
            create(args.run, args.condition, args.brief, args.repetition, args.out)
        elif args.realisation_command == "assess":
            assess(args.packet, Review.model_validate_json(args.review.read_text(encoding="utf-8")))
        else:
            return 1 if report(args.packet)["stale"] else 0
        return 0
    except (OSError, ValueError) as error:
        print(f"nde realisation: {error}", file=sys.stderr)
        return 2
