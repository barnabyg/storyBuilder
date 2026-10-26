"""Named diagnostics used by inspection, repair and mutation evidence."""

from collections.abc import Iterable
from typing import Protocol

from nde.files import checksum_lf
from nde.narrative.diagnostics import Diagnostic, Validation
from nde.narrative.interpretation import Annotation
from nde.narrative.interpretation_validation import interpretation_diagnostics
from nde.narrative.models import NarrativePlan, Span
from nde.narrative.projection import tier_diagnostics
from nde.narrative.state_validation import state_diagnostics
from nde.narrative.structural_validation import validate_structure

__all__ = ["Diagnostic", "Validation", "check_ids", "check_spans", "validate_plan"]


class HasID(Protocol):
    @property
    def id(self) -> str: ...


def check_ids(records: Iterable[HasID]) -> list[Diagnostic]:
    seen: set[str] = set()
    diagnostics: list[Diagnostic] = []
    for record in records:
        if record.id in seen:
            diagnostics.append(
                Diagnostic(
                    code="duplicate_id", location=record.id, message=f"duplicate ID {record.id}"
                )
            )
        seen.add(record.id)
    return diagnostics


def check_spans(plan: NarrativePlan, spans: Iterable[tuple[str, Span]]) -> list[Diagnostic]:
    sources = {source.id: source for source in plan.sources}
    diagnostics: list[Diagnostic] = []
    for location, span in spans:
        source = sources.get(span.source_id)
        if source is None:
            code = "span_source"
        elif span.source_sha256 != source.sha256:
            code = "span_version"
        elif not 0 <= span.start < span.end <= len(source.text):
            code = "span_bounds"
        else:
            continue
        diagnostics.append(
            Diagnostic(
                code=code, location=location, message=f"invalid evidence span for {span.source_id}"
            )
        )
    return diagnostics


def validate_plan(plan: NarrativePlan) -> Validation:
    records: list[HasID] = [plan, *plan.characters, *plan.sources, *plan.scenes]
    records.extend(beat for scene in plan.scenes for beat in scene.beats)
    records.extend([*plan.states, *plan.relationships, *plan.constraints])
    records.extend(
        [
            *plan.causal_links,
            *plan.setups,
            *plan.payoff_links,
            *plan.knowledge_facts,
            *plan.reveal_links,
            *plan.knowledge_uses,
        ]
    )
    annotations: list[Annotation] = [
        *plan.arcs,
        *plan.reader_affect,
        *plan.tension,
        *plan.themes,
        *plan.motifs,
        *plan.discourse,
        *plan.subtext,
    ]
    records.extend(annotations)
    records.extend(plan.motif_echoes)
    diagnostics = check_ids(records)
    characters = {character.id for character in plan.characters}
    spans: list[tuple[str, Span]] = []
    for source in plan.sources:
        if checksum_lf(source.text.encode()) != source.sha256 or "\r" in source.text:
            diagnostics.append(
                Diagnostic(
                    code="source_checksum",
                    location=source.id,
                    message="source must match SHA-256 and use LF text",
                )
            )
    for scene in plan.scenes:
        if len(scene.character_ids) != len(set(scene.character_ids)):
            diagnostics.append(
                Diagnostic(
                    code="duplicate_presence",
                    location=scene.id,
                    message="duplicate character presence",
                )
            )
        for character in scene.character_ids:
            if character not in characters:
                diagnostics.append(
                    Diagnostic(
                        code="character_reference",
                        location=scene.id,
                        message=f"unknown character {character}",
                    )
                )
        spans.extend((scene.id, span) for span in scene.evidence)
        spans.extend((beat.id, span) for beat in scene.beats for span in beat.evidence)
    spans.extend((record.id, span) for record in annotations for span in record.evidence)
    spans.extend((echo.id, span) for echo in plan.motif_echoes for span in echo.evidence)
    diagnostics.extend(check_spans(plan, spans))
    diagnostics.extend(validate_structure(plan))
    diagnostics.extend(state_diagnostics(plan))
    diagnostics.extend(interpretation_diagnostics(plan))
    diagnostics.extend(tier_diagnostics(plan))
    return Validation(diagnostics=diagnostics)
