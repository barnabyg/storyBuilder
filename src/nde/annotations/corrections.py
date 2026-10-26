"""Append-only correction versions, explicit review and active correction intervals."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from nde.files import append_line, write_text_lf
from nde.narrative.workflow import PlanError, load_plan, render

from .drafting import validate_annotation
from .records import (
    OwnerReview,
    ReviewDecision,
    Revision,
    TimerEvent,
    digest,
    read_session,
    save_session,
    timer_events,
)


def timer(directory: Path, action: Literal["start", "pause", "recover"]) -> None:
    read_session(directory)
    events = timer_events(directory)
    active = events[-1] if events and events[-1].action == "start" else None
    now = time.time()
    if action == "start" and active:
        raise PlanError("timer already active; pause or recover the interrupted interval")
    if action != "start" and active is None:
        raise PlanError("no active correction interval")
    seconds = max(0.0, now - active.at) if active and action == "pause" else 0
    append_line(
        directory / "timer.jsonl",
        TimerEvent(action=action, at=now, seconds=seconds).model_dump_json(),
    )


def correction_seconds(directory: Path) -> float | None:
    events = timer_events(directory)
    if not events or events[-1].action == "start":
        return None
    completed = [event.seconds for event in events if event.action == "pause"]
    return sum(completed) if completed else None


def revise(directory: Path, file: Path) -> Revision:
    session = read_session(directory)
    if session.status != "drafted":
        raise PlanError("a validated draft is required before owner correction")
    text = file.read_text(encoding="utf-8")
    number = len(session.revisions) + 1
    folder = directory / "revisions" / f"{number:04d}"
    if folder.exists():
        pending = folder / "submitted.json"
        if not pending.is_file() or pending.read_text(encoding="utf-8") != text:
            raise PlanError(f"unfinished revision: resume with {pending}")
    else:
        folder.mkdir(parents=True)
        write_text_lf(folder / "submitted.json", text)
    plan, errors = validate_annotation(text, session.source)
    valid = plan is not None and not errors
    if plan is not None:
        render(plan, folder)
    write_text_lf(folder / "errors.json", json.dumps(errors, indent=2) + "\n")
    revision = Revision(
        number=number,
        sha256=digest(text),
        valid=valid,
        file=f"revisions/{number:04d}/submitted.json",
    )
    session.revisions.append(revision)
    save_session(directory, session)
    return revision


def review(directory: Path, revision: int, decision: ReviewDecision) -> OwnerReview:
    session = read_session(directory)
    if not 1 <= revision <= len(session.revisions):
        raise PlanError("unknown correction revision")
    record = session.revisions[revision - 1]
    path = directory / record.file
    text = path.read_text(encoding="utf-8")
    if not record.valid or digest(text) != record.sha256:
        raise PlanError("review requires an unchanged valid correction version")
    plan = load_plan(path)
    _, errors = validate_annotation(plan.model_dump_json(), session.source)
    if errors:
        raise PlanError("review requires a semantically valid correction")
    if correction_seconds(directory) is None:
        raise PlanError("pause and record actual correction time before review")
    evidence = OwnerReview(
        revision=revision,
        sha256=record.sha256,
        owner=decision.owner,
        literary_notes=decision.notes,
        meaningful_trajectories=decision.meaningful,
        accepted=decision.accepted,
        recorded_at=datetime.now(UTC).isoformat(),
    )
    session.owner_reviews.append(evidence)
    save_session(directory, session)
    return evidence
