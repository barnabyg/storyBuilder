"""Readable values, uncertainty and source evidence for interpretive cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nde.narrative.interpretation import Arc, ReaderAffect, Tension, iter_annotations

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan


def interpretation_lines(plan: NarrativePlan) -> list[str]:
    lines: list[str] = []
    for channel, record in iter_annotations(plan):
        reading = record.reading
        value = reading.value if reading.value is not None else "(no value recorded)"
        confidence = "unrecorded" if reading.confidence is None else str(reading.confidence)
        location = ", ".join(record.scene_ids) or f"story {plan.id}"
        lines.append(
            f"{channel} {record.id} [{location}] {record.kind}: {value}; "
            f"status={reading.status}; confidence={confidence}"
        )
        if reading.alternatives:
            lines.append(f"Alternative readings: {'; '.join(reading.alternatives)}")
        lines.extend(f"Evidence {record.id}: {span.model_dump_json()}" for span in record.evidence)
        if isinstance(record, Arc) and record.character_id is not None:
            lines.append(f"Arc character: {record.character_id}")
        if isinstance(record, (Arc, ReaderAffect, Tension)) and record.trajectory is not None:
            label = (
                "agency"
                if isinstance(record, Arc)
                else ("mood" if isinstance(record, ReaderAffect) else "tension")
            )
            points = record.trajectory.points
            sequence = " → ".join(
                "unknown" if point.value is None else str(point.value) for point in points
            )
            scenes = " → ".join(point.scene_id for point in points)
            lines.append(
                f"{label} trajectory {record.id}: {sequence}; "
                f"shape={record.trajectory.shape}; scenes={scenes}; ordinal 0-4"
            )
    for echo in plan.motif_echoes:
        lines.append(
            f"Motif echo {echo.id}: {echo.from_motif_id} → {echo.to_motif_id} ({echo.kind})"
        )
        lines.extend(f"Evidence {echo.id}: {span.model_dump_json()}" for span in echo.evidence)
    return lines
