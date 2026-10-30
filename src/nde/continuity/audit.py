"""Running the checks: a model reads the numbered story, then plain code decides what to flag.

Each audit is a structured request whose answers cite paragraph numbers, and a
pure `check` that turns the answers into flags. The model extracts; it never
rates the story. Checks guide repair and never establish story quality.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import create_model, model_validator

from nde.continuity.flags import Flag
from nde.continuity.plan import ContinuityPlan, render_plan
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict
from nde.runs.config import PromptRef, RoleName
from nde.stories.pipeline import Step, StoryContext, derive_seed

AUDIT_VARIABLES = frozenset({"plan", "story"})
CRITIC_ROLE: RoleName = "critic"


class AuditRecord(Strict, ABC):
    """A check's structured reading of the story."""

    @abstractmethod
    def cited_paragraphs(self) -> Iterable[int]:
        """Every paragraph number the reading cites, so bad citations can be repaired."""


def bounded[R: AuditRecord](model: type[R], paragraphs: int) -> type[R]:
    """`model`, rejecting citations of paragraphs the story does not have."""

    def check(record: R) -> R:
        for number in record.cited_paragraphs():
            if not 1 <= number <= paragraphs:
                raise ValueError(
                    f"paragraph {number} does not exist; the story has paragraphs 1 to {paragraphs}"
                )
        return record

    validator = cast("Callable[..., Any]", model_validator(mode="after")(check))
    return create_model(
        model.__name__, __base__=model, __validators__={"_citations_exist": validator}
    )


@dataclass(frozen=True)
class Audit[R: AuditRecord]:
    name: str
    prompt: PromptRef
    model: type[R]
    check: Callable[[ContinuityPlan, NumberedStory, R], list[Flag]]
    # A file to keep beside the story, holding the latest reading in a form for people.
    record_file: str | None = None
    record: Callable[[NumberedStory, R], object] | None = None


def run_audit(
    ctx: StoryContext,
    audit: Audit[Any],
    plan: ContinuityPlan,
    story: NumberedStory,
    *,
    round_number: int,
) -> list[Flag]:
    """One audit of the story as it stands; keeps the reading and returns the flags."""
    step = Step(
        f"audit-{audit.name}-round-{round_number}",
        CRITIC_ROLE,
        audit.prompt,
        {"plan": render_plan(plan), "story": story.render()},
    )
    seed = derive_seed(ctx.seed, "audit", audit.name, round_number)
    record = ctx.structured(step, bounded(audit.model, len(story)), seed=seed)
    ctx.keep(
        f"audits/round-{round_number}/{audit.name}.json",
        json.dumps(record.model_dump(mode="json"), indent=2) + "\n",
    )
    if audit.record_file is not None and audit.record is not None:
        ctx.keep(audit.record_file, json.dumps(audit.record(story, record), indent=2) + "\n")
    return audit.check(plan, story, record)
