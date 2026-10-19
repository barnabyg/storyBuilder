"""Generation budgets: what each story cost, in provider calls and tokens.

Every call attributed to a story counts: structured-output repairs, length
regenerations and equal-budget revision passes included. Cache hits count at
their recorded token counts, so a re-run reports the same budgets.
"""

from collections.abc import Sequence

from nde.runs.records import Budget, CallRecord


def story_budget(
    calls: Sequence[CallRecord], condition: str, brief: str, repetition: int
) -> Budget:
    mine = [
        call
        for call in calls
        if (call.condition, call.brief, call.repetition) == (condition, brief, repetition)
    ]
    return Budget(
        calls=len(mine),
        input_tokens=sum(call.input_tokens for call in mine),
        output_tokens=sum(call.output_tokens for call in mine),
    )


def within(a: int, b: int, tolerance: float) -> bool:
    """Whether two budgets are within `tolerance` of the larger one."""
    return abs(a - b) <= tolerance * max(a, b)
