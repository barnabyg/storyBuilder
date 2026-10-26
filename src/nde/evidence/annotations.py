"""Assess saved annotations without promoting model drafts to owner evidence."""

import json
import math

from nde.annotations.drafting import validate_annotation
from nde.annotations.records import Session, TimerEvent, digest
from nde.narrative.models import NarrativePlan, Source
from nde.narrative.validation import validate_plan

from .records import AnnotationInput, StoryEvidence

CORRECTION_TARGET_SECONDS = 7200


def _timing(captured: AnnotationInput, row: StoryEvidence) -> None:
    events = [
        TimerEvent.model_validate_json(line) for line in captured.timer.splitlines() if line.strip()
    ]
    active: TimerEvent | None = None
    total = 0.0
    completed = False
    previous = -math.inf
    for event in events:
        if not math.isfinite(event.at) or not math.isfinite(event.seconds) or event.at < previous:
            raise ValueError("invalid correction timer timestamps")
        previous = event.at
        if event.action == "start":
            if active is not None or event.seconds != 0:
                raise ValueError("invalid correction timer start")
            active = event
        else:
            if active is None:
                raise ValueError("correction timer has no active interval")
            expected = event.at - active.at if event.action == "pause" else 0.0
            if not math.isclose(event.seconds, expected, abs_tol=0.001):
                raise ValueError("correction timer duration does not match its interval")
            if event.action == "pause":
                total += event.seconds
                completed = True
            active = None
    measured = total if completed and active is None else None
    if measured != captured.measured_seconds:
        raise ValueError("saved correction time disagrees with timer intervals")
    row.correction_seconds = measured
    if measured is None:
        row.pending.append("actual owner correction time missing or interval still active")
    elif measured >= CORRECTION_TARGET_SECONDS:
        row.failures.append(
            "owner correction time is at or above the approximately two-hour target"
        )


def _trajectories(plan: NarrativePlan, row: StoryEvidence) -> None:
    channels = (("agency arc", plan.arcs), ("mood", plan.reader_affect), ("tension", plan.tension))
    for channel, records in channels:
        for record in records:
            trajectory = record.trajectory
            if trajectory is None:
                continue
            values = [
                str(point.value) if point.value is not None else "unknown"
                for point in trajectory.points
            ]
            row.trajectories.append(
                f"{channel} {record.id}: {trajectory.shape.value}; "
                f"{' → '.join(values)}; scenes "
                f"{', '.join(point.scene_id for point in trajectory.points)}"
            )
            row.trajectory_signature.append(
                f"{channel}:{trajectory.shape.value}:{','.join(values)}"
            )
    row.trajectory_signature.sort()


def _review(session: Session, row: StoryEvidence) -> None:
    if not session.revisions:
        row.pending.append("owner-corrected annotation missing")
        row.pending.append("explicit owner literary review missing")
        return
    latest = session.revisions[-1]
    reviews = [review for review in session.owner_reviews if review.revision == latest.number]
    if not reviews:
        row.pending.append("explicit owner literary review missing for latest correction")
        return
    review = reviews[-1]
    if review.sha256 != latest.sha256:
        row.mechanical_failures.append("owner review checksum does not match latest correction")
        return
    if not review.owner.strip() or not review.literary_notes.strip():
        row.mechanical_failures.append("owner review needs a named owner and literary notes")
        return
    row.owner = review.owner
    row.literary_evidence = review.literary_notes
    row.owner_review = "accepted" if review.accepted else "rejected"
    row.owner_confirmed_trajectories = review.meaningful_trajectories
    if not review.accepted:
        row.failures.append("owner rejected the corrected annotation")
    if not review.meaningful_trajectories:
        row.failures.append("owner did not confirm meaningful trajectory differences")


def _saved_plan(
    captured: AnnotationInput, session: Session, row: StoryEvidence
) -> NarrativePlan | None:
    draft_text = captured.files.get("draft/plan.json")
    if draft_text is None:
        row.mechanical_failures.append("validated draft file missing")
        return None
    if not session.attempts or captured.files.get("original.txt") != session.attempts[0].text:
        row.mechanical_failures.append("original model draft missing or changed")
    draft, errors = validate_annotation(draft_text, session.source)
    if draft is None or errors:
        row.mechanical_failures.extend(f"draft: {error}" for error in errors)
        return None
    if session.attempts:
        accepted, errors = validate_annotation(session.attempts[-1].text, session.source)
        if errors or accepted != draft:
            row.mechanical_failures.append(
                "saved draft differs from the final successful generation attempt"
            )
    plan = draft
    for number, revision in enumerate(session.revisions, start=1):
        expected = f"revisions/{number:04d}/submitted.json"
        text = captured.files.get(expected)
        if revision.number != number or revision.file != expected or text is None:
            row.mechanical_failures.append("correction versions are missing or out of sequence")
            return None
        if digest(text) != revision.sha256:
            row.mechanical_failures.append(f"correction {number} checksum mismatch")
            return None
        if number == len(session.revisions):
            row.revision = number
            row.identifiers["revision_sha256"] = revision.sha256
            plan, errors = validate_annotation(text, session.source)
            if not revision.valid or errors:
                row.mechanical_failures.append("latest correction is mechanically invalid")
                row.mechanical_failures.extend(errors)
    return plan


def assess_annotation(
    captured: AnnotationInput, row: StoryEvidence, source: Source | None, schema_sha256: str
) -> None:
    row.mechanical_failures.extend(captured.capture_errors)
    if captured.session is None:
        row.pending.extend(
            [
                "tier-L annotation missing",
                "actual owner correction time missing",
                "explicit owner literary review missing",
            ]
        )
        return
    try:
        if json.loads(captured.session).get("pending_call") is True:
            row.annotation_status = "in-flight recovery required"
            row.pending.append("in-flight generation must be recovered before gate review")
            return
        session = Session.model_validate_json(captured.session)
        row.annotation_status = session.status
        row.generation_cost_usd = session.spent_usd
        row.draft_attempts = len(session.attempts)
        row.identifiers.update(
            session_id=session.id,
            schema_sha256=session.schema_sha256,
            config_sha256=session.config_sha256,
            prompt_sha256=session.prompt_sha256,
            git_sha=session.git_sha,
        )
        row.identifiers.update(
            model=session.config.annotator.label,
            prompt=f"{session.config.prompt.template}@{session.config.prompt.version}",
            span_basis=session.source.span_basis,
        )
        if session.story_id != row.id or session.provenance != row.provenance:
            raise ValueError("annotation provenance does not match the selected story")
        if source is not None and session.source != source:
            raise ValueError("annotation source differs from the verified manifest source")
        if session.config_sha256 != digest(session.config.model_dump_json()):
            raise ValueError("annotation configuration checksum mismatch")
        if session.schema_sha256 != schema_sha256:
            raise ValueError("annotation schema checksum differs from the evidence schema")
        _timing(captured, row)
        if session.status != "drafted":
            row.pending.append(f"annotation generation is {session.status}: {session.failure}")
            return
        plan = _saved_plan(captured, session, row)
        if plan is not None:
            row.diagnostics = validate_plan(plan).diagnostics
            _trajectories(plan, row)
            if any(item.severity == "human" for item in row.diagnostics):
                row.pending.append("unresolved human-assessment diagnostics remain")
        _review(session, row)
    except (ValueError, KeyError) as error:
        row.mechanical_failures.append(str(error))
