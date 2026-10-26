"""Mechanical invariants for interpretive channels; never a literary verdict."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nde.narrative.diagnostics import Diagnostic
from nde.narrative.interpretation import Annotation, Arc, ReaderAffect, Tension

if TYPE_CHECKING:
    from nde.narrative.models import NarrativePlan


def interpretation_diagnostics(plan: NarrativePlan) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    scenes = {scene.id: index for index, scene in enumerate(plan.scenes)}
    characters = {character.id for character in plan.characters}
    records: list[Annotation] = [
        *plan.arcs,
        *plan.reader_affect,
        *plan.tension,
        *plan.themes,
        *plan.motifs,
        *plan.discourse,
        *plan.subtext,
    ]
    for record in records:
        if any(scene not in scenes for scene in record.scene_ids):
            diagnostics.append(
                Diagnostic(
                    code="interpretive_scene_reference",
                    location=record.id,
                    message="annotation scene IDs must resolve",
                )
            )
        if (
            isinstance(record, Arc)
            and record.character_id is not None
            and record.character_id not in characters
        ):
            diagnostics.append(
                Diagnostic(
                    code="arc_character_reference",
                    location=record.id,
                    message=f"unknown arc character {record.character_id}",
                )
            )
        if isinstance(record, (Arc, ReaderAffect, Tension)) and record.trajectory is not None:
            positions = [scenes.get(point.scene_id) for point in record.trajectory.points]
            if any(position is None for position in positions):
                diagnostics.append(
                    Diagnostic(
                        code="trajectory_scene_reference",
                        location=record.id,
                        message="trajectory points must name scenes",
                    )
                )
            else:
                known = [position for position in positions if position is not None]
                if known != sorted(known):
                    diagnostics.append(
                        Diagnostic(
                            code="trajectory_order",
                            location=record.id,
                            message="trajectory points must follow scene order",
                        )
                    )
    motifs = {record.id for record in plan.motifs}
    for echo in plan.motif_echoes:
        if echo.from_motif_id not in motifs or echo.to_motif_id not in motifs:
            diagnostics.append(
                Diagnostic(
                    code="motif_endpoint",
                    location=echo.id,
                    message="motif echo endpoints must name motif IDs",
                )
            )
        if echo.from_motif_id == echo.to_motif_id:
            diagnostics.append(
                Diagnostic(
                    code="motif_self_link",
                    location=echo.id,
                    message="a motif echo must link distinct occurrences",
                )
            )
    return diagnostics
