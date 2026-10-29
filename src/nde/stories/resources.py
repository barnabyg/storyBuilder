"""Separate accounts of a story's provider calls and local conditional-model work.

Provider calls are grouped by what they served: structural planning, hybrid text
completion of a learned plan, scene prose, or equal-budget prose revision.
Local model work has its own units (forward passes, model tokens, attention
cells and seconds). The two are reported side by side and never summed.
"""

import re
from collections.abc import Sequence
from typing import Any, Literal

from nde.runs.records import CallRecord, LocalTotals, LocalWorkRecord

type Category = Literal["planning", "completion", "prose", "revision"]

CATEGORIES: tuple[Category, ...] = ("planning", "completion", "prose", "revision")
# The step of a learned plan's hybrid text completion.
COMPLETION_STEP = "conditional-completion"
# Imported plans re-attribute their original calls under this prefix.
IMPORTED = "imported-"
# Baseline A writes its whole story in this one prose step.
WHOLE_STORY_STEP = "story"


def call_category(step: str | None) -> Category:
    """What a story's call served, from its step name; imported calls keep their origin."""
    name = (step or "").removeprefix(IMPORTED)
    if name == COMPLETION_STEP:
        return "completion"
    if re.fullmatch(r"scene-\d+", name) or name == WHOLE_STORY_STEP:
        return "prose"
    # Equal-budget story revisions; D's critic regenerations are structural planning.
    if re.fullmatch(r"revision-\d+", name):
        return "revision"
    return "planning"


def _mine[T: CallRecord | LocalWorkRecord](
    records: Sequence[T], condition: str, brief: str, repetition: int
) -> list[T]:
    return [
        record
        for record in records
        if (record.condition, record.brief, record.repetition) == (condition, brief, repetition)
    ]


def local_totals(
    work: Sequence[LocalWorkRecord], condition: str, brief: str, repetition: int
) -> LocalTotals:
    """One story's local model work, imported work included."""
    mine = _mine(work, condition, brief, repetition)
    return LocalTotals(
        operations=len(mine),
        passes=sum(record.passes for record in mine),
        tokens=sum(record.tokens for record in mine),
        attention_cells=sum(record.attention_cells for record in mine),
        elapsed_seconds=sum(record.elapsed_seconds for record in mine),
    )


def provider_categories(calls: Sequence[CallRecord]) -> dict[str, dict[str, Any]]:
    """Calls, tokens, list-price cost and new spend per category, every category present."""
    result: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        mine = [call for call in calls if call_category(call.step) == category]
        result[category] = {
            "calls": len(mine),
            "imported_calls": sum(call.imported_from is not None for call in mine),
            "input_tokens": sum(call.input_tokens for call in mine),
            "output_tokens": sum(call.output_tokens for call in mine),
            "token_cost_usd": sum(call.token_cost_usd for call in mine),
            "new_spend_usd": sum(call.cost_usd for call in mine),
        }
    return result


def story_resources(
    calls: Sequence[CallRecord],
    work: Sequence[LocalWorkRecord],
    condition: str,
    brief: str,
    repetition: int,
) -> dict[str, Any]:
    """A story's provider calls by category and its local model work, in separate units."""
    return {
        "provider": provider_categories(_mine(calls, condition, brief, repetition)),
        "local": local_totals(work, condition, brief, repetition).model_dump(mode="json"),
    }
