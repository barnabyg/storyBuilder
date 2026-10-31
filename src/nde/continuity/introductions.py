"""The introduction check: every named person has a sheet and enters as planned."""

from collections.abc import Iterable
from typing import override

from pydantic import Field

from nde.continuity.audit import AuditRecord
from nde.continuity.flags import Flag
from nde.continuity.plan import ContinuityPlan
from nde.continuity.story import NumberedStory
from nde.llm.config import Strict

PROBLEM_TYPE = "character_introduction"


class Appearance(Strict):
    name: str = Field(min_length=1, description="the person's name as the story gives it")
    first_paragraph: int = Field(description="the paragraph where they are first on the page")
    introduced: bool = Field(
        description="whether the reader is told who they are, and something of their appearance "
        "or history, where they first appear"
    )
    why: str = Field(min_length=1, description="one sentence on what the reader is told")


class IntroductionAudit(AuditRecord):
    people: list[Appearance] = Field(description="every named person who takes part in the story")

    @override
    def cited_paragraphs(self) -> Iterable[int]:
        return (person.first_paragraph for person in self.people)


def check_introductions(
    plan: ContinuityPlan, story: NumberedStory, audit: IntroductionAudit
) -> list[Flag]:
    flags: list[Flag] = []
    seen: set[str] = set()

    def flag(paragraph: int, reason: str) -> None:
        flags.append(
            Flag(
                problem_type=PROBLEM_TYPE,
                passage=story.paragraph(paragraph),
                reason=reason,
                paragraph=paragraph,
            )
        )

    for person in audit.people:
        paragraph = person.first_paragraph
        sheet = plan.sheet_for(person.name)
        if sheet is None:
            flag(paragraph, f"{person.name} is a named character with no character sheet.")
            continue
        seen.add(sheet.name)
        scene = story.scene(paragraph)
        if scene != sheet.enters_scene:
            flag(
                paragraph,
                f"{person.name} first appears in scene {scene}, but the plan has them enter "
                f"in scene {sheet.enters_scene}: {sheet.introduction}",
            )
        if not person.introduced:
            flag(
                paragraph,
                f"{person.name} appears with no introduction. {person.why} "
                f"Planned introduction: {sheet.introduction}",
            )
    for sheet in plan.characters:
        if sheet.name not in seen:
            flag(
                story.first_of_scene(sheet.enters_scene),
                f"{sheet.name} is planned to enter in scene {sheet.enters_scene} but never "
                f"appears: {sheet.introduction}",
            )
    return flags
