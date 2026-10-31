"""The knowledge check: nobody acts on something the story never let them learn."""

from collections.abc import Iterable
from typing import Literal, Self, override

from pydantic import Field, model_validator

from nde.continuity.audit import AuditRecord
from nde.continuity.flags import Flag
from nde.continuity.plan import ContinuityPlan
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict

PROBLEM_TYPE = "unexplained_knowledge"


class KnowledgeEntry(Strict):
    character: str = Field(min_length=1, description="who acts on the fact")
    fact: str = Field(min_length=1, description="what they know, in one short sentence")
    acted_paragraph: int = Field(description="the paragraph where they first act on it")
    how_learned: str = Field(
        min_length=1, description="how they learned it, or that the story never says"
    )
    source: Literal["ordinary", "on_page", "sheet", "none"] = Field(
        description="ordinary if a person in their place would simply know it; on_page if they "
        "see, hear or work it out in the story; sheet if their character sheet lists it; none "
        "if the story gives no way to know it"
    )
    learned_paragraph: int | None = Field(
        default=None, description="for on_page: the paragraph where they learn it"
    )
    sheet_item: int | None = Field(
        default=None, description="for sheet: the number of the item on their sheet"
    )

    @model_validator(mode="after")
    def _source_is_cited(self) -> Self:
        if self.source == "on_page" and self.learned_paragraph is None:
            raise ValueError(f"{self.character}: on_page needs learned_paragraph")
        if self.source == "sheet" and self.sheet_item is None:
            raise ValueError(f"{self.character}: sheet needs sheet_item")
        return self


class KnowledgeAudit(AuditRecord):
    entries: list[KnowledgeEntry] = Field(
        description="each specific fact a character acts on, in story order"
    )

    @override
    def cited_paragraphs(self) -> Iterable[int]:
        for entry in self.entries:
            yield entry.acted_paragraph
            if entry.learned_paragraph is not None:
                yield entry.learned_paragraph


def _problem(plan: ContinuityPlan, entry: KnowledgeEntry) -> str | None:
    """What is wrong with the entry's explanation, or None if it holds up."""
    if entry.source == "ordinary":
        return None
    if entry.source == "none":
        return f"The story gives no way to know it: {entry.how_learned}"
    if entry.source == "on_page":
        learned = entry.learned_paragraph
        if learned is not None and learned > entry.acted_paragraph:
            return (
                f"They act on it in paragraph {entry.acted_paragraph} but only learn it "
                f"in paragraph {learned}."
            )
        return None
    sheet = plan.sheet_for(entry.character)
    item = entry.sheet_item
    if sheet is None or item is None or not 1 <= item <= len(sheet.knows_at_start):
        return f"The sheet does not list it, though the reading cites item {item}."
    return None


def check_knowledge(
    plan: ContinuityPlan, story: NumberedStory, audit: KnowledgeAudit
) -> list[Flag]:
    flags: list[Flag] = []
    for entry in audit.entries:
        problem = _problem(plan, entry)
        if problem is not None:
            flags.append(
                Flag(
                    problem_type=PROBLEM_TYPE,
                    passage=story.paragraph(entry.acted_paragraph),
                    paragraph=entry.acted_paragraph,
                    reason=f"{entry.character} knows something unexplained: {entry.fact} {problem}",
                )
            )
    return flags


def knowledge_by_scene(story: NumberedStory, audit: KnowledgeAudit) -> list[dict[str, object]]:
    """The scene-by-scene record of what each character acted on and how they learned it."""
    scenes = sorted(set(story.scenes))
    return [
        {
            "scene": scene,
            "entries": [
                entry.model_dump(mode="json")
                for entry in audit.entries
                if story.scene(entry.acted_paragraph) == scene
            ],
        }
        for scene in scenes
    ]
