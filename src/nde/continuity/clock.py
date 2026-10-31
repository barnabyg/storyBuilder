"""The clock check: compares the time the prose gives events with how long they should take."""

from collections.abc import Iterable
from typing import override

from pydantic import Field

from nde.continuity.audit import AuditRecord
from nde.continuity.flags import Flag
from nde.continuity.plan import ContinuityPlan, format_minutes
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict

PROBLEM_TYPE = "impossible_timing"
# How far a scene's stated time may stray from the plan before it is flagged: a factor
# either way, plus some slack so that short scenes are not flagged for a few minutes.
SCENE_FACTOR = 2.0
SCENE_SLACK_MINUTES = 5.0


class TimedEvent(Strict):
    event: str = Field(min_length=1, description="what happens, in a few words")
    paragraph: int = Field(description="the paragraph where the prose shows it happening")
    prose_minutes: float = Field(ge=0, description="the minutes the prose gives it")
    key_event: str | None = Field(
        default=None, description="the plan's label for this event, such as K2, if it has one"
    )
    realistic_min_minutes: float = Field(ge=0, description="the least time it could take")
    realistic_max_minutes: float = Field(ge=0, description="the most time it could take")
    why: str = Field(min_length=1, description="one sentence on what the realistic time rests on")


class SceneReading(Strict):
    scene: int = Field(description="the scene's number")
    paragraph: int = Field(description="the paragraph that states the time")
    prose_minutes: float = Field(ge=0, description="the minutes the prose says have passed")


class ClockAudit(AuditRecord):
    events: list[TimedEvent] = Field(
        description="every event whose time the prose states or clearly implies"
    )
    scenes: list[SceneReading] = Field(
        description="each scene where the prose says how much time has passed"
    )

    @override
    def cited_paragraphs(self) -> Iterable[int]:
        yield from (event.paragraph for event in self.events)
        yield from (reading.paragraph for reading in self.scenes)


def _event_reason(event: TimedEvent, low: float, high: float) -> str | None:
    given = format_minutes(event.prose_minutes)
    if event.prose_minutes < low:
        return (
            f"{event.event} happens too quickly: the prose gives it {given}, but it should "
            f"take at least {format_minutes(low)}. {event.why}"
        )
    if event.prose_minutes > high:
        return (
            f"{event.event} happens too slowly: the prose gives it {given}, but it should "
            f"take at most {format_minutes(high)}. {event.why}"
        )
    return None


def check_clock(plan: ContinuityPlan, story: NumberedStory, audit: ClockAudit) -> list[Flag]:
    flags: list[Flag] = []
    key_events = plan.key_events()
    for event in audit.events:
        low, high = event.realistic_min_minutes, event.realistic_max_minutes
        planned = None if event.key_event is None else key_events.get(event.key_event)
        if planned is not None:
            # The plan's clock outranks the reader's estimate.
            low, high = planned.expected_min_minutes, planned.expected_max_minutes
        reason = _event_reason(event, low, high)
        if reason is not None:
            flags.append(
                Flag(
                    problem_type=PROBLEM_TYPE,
                    passage=story.paragraph(event.paragraph),
                    reason=reason,
                    paragraph=event.paragraph,
                )
            )
    for reading in audit.scenes:
        if not 1 <= reading.scene <= len(plan.scenes):
            continue
        planned_minutes = plan.scenes[reading.scene - 1].elapsed_minutes
        low = planned_minutes / SCENE_FACTOR - SCENE_SLACK_MINUTES
        high = planned_minutes * SCENE_FACTOR + SCENE_SLACK_MINUTES
        if not low <= reading.prose_minutes <= high:
            flags.append(
                Flag(
                    problem_type=PROBLEM_TYPE,
                    passage=story.paragraph(reading.paragraph),
                    paragraph=reading.paragraph,
                    reason=(
                        f"Scene {reading.scene} says {format_minutes(reading.prose_minutes)} "
                        f"have passed, but the plan's clock gives "
                        f"{format_minutes(planned_minutes)}."
                    ),
                )
            )
    return flags
