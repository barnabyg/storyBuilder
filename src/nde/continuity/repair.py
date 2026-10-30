"""Targeted repair: rewrite only the flagged passages and leave every other word alone."""

from collections.abc import Sequence
from typing import Self

from pydantic import Field, model_validator

from nde.continuity.flags import Flag
from nde.continuity.plan import KNOWLEDGE_RULE, PLAIN_STYLE_RULES, ContinuityPlan, render_plan
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict
from nde.runs.config import PromptRef
from nde.stories.normalise import normalise
from nde.stories.pipeline import Step, StoryContext, derive_seed

REPAIR_PROMPT = PromptRef(template="continuity_repair", version=1)
REPAIR_VARIABLES = frozenset({"plan", "story", "passage", "problems", "rules"})


class RepairedPassage(Strict):
    replacement: str = Field(
        min_length=1, description="the new text for the passage, as one paragraph"
    )

    @model_validator(mode="after")
    def _one_paragraph(self) -> Self:
        if "\n" in self.replacement.strip():
            raise ValueError("the replacement must be one paragraph, with no line breaks")
        return self


class Change(Strict):
    """One rewritten passage, with what it addressed and the text before and after."""

    attempt: int
    paragraph: int
    problems: list[Flag]
    before: str
    after: str


def group_by_passage(flags: Sequence[Flag]) -> list[list[Flag]]:
    """The flags to repair together: one group per distinct passage, in first-seen order."""
    groups: dict[str, list[Flag]] = {}
    for flag in flags:
        groups.setdefault(flag.passage, []).append(flag)
    return list(groups.values())


def _render_problems(flags: Sequence[Flag]) -> str:
    return "\n".join(f"- {flag.problem_type}: {flag.reason}" for flag in flags)


def repair_passage(
    ctx: StoryContext,
    plan: ContinuityPlan,
    story: NumberedStory,
    flags: Sequence[Flag],
    *,
    attempt: int,
) -> tuple[NumberedStory, Change] | None:
    """Rewrite the passage `flags` share; None if the story no longer contains it.

    Nothing outside the passage can change: the model only supplies the new text,
    and code splices it into the one paragraph that holds the passage.
    """
    passage = flags[0].passage
    number = next(
        (n for n in range(1, len(story) + 1) if passage in story.paragraph(n)),
        None,
    )
    if number is None:
        return None
    step = Step(
        f"repair-{attempt}-p{number:03d}",
        "prose",
        REPAIR_PROMPT,
        {
            "plan": render_plan(plan),
            "story": story.render(),
            "passage": passage,
            "problems": _render_problems(flags),
            "rules": f"{KNOWLEDGE_RULE}\n\n{PLAIN_STYLE_RULES}",
        },
    )
    seed = derive_seed(ctx.seed, "repair", attempt, number)
    repaired = ctx.structured(step, RepairedPassage, seed=seed)
    replacement = " ".join(normalise(repaired.replacement).split())
    if not replacement:
        return None
    before = story.paragraph(number)
    after = before.replace(passage, replacement, 1)
    change = Change(
        attempt=attempt, paragraph=number, problems=list(flags), before=before, after=after
    )
    return story.with_paragraph(number, after), change
