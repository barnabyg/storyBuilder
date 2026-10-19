"""The shared prose contract's revision prompt: rewrite a whole story from its brief.

Equal-budget mode gives the cheaper condition revision passes with this prompt
on the prose role, so extra compute goes into the same kind of prose work for
every condition.
"""

from nde.prose.scenes import PROSE_ROLE
from nde.runs.config import PromptRef
from nde.stories.pipeline import Step, StoryContext

REVISION = PromptRef(template="story_revision", version=1)
REVISION_VARIABLES = frozenset({"brief", "story", "target_words"})


def revision_step(number: int) -> str:
    return f"revision-{number:02d}"


def revise(ctx: StoryContext, story: str, *, number: int, seed: int, prompt: PromptRef) -> str:
    """One revision pass over `story`; returns the raw revised text."""
    variables = {
        "brief": ctx.brief.describe(),
        "story": story,
        "target_words": str(ctx.brief.target_words),
    }
    return ctx.complete(Step(revision_step(number), PROSE_ROLE, prompt, variables), seed=seed)
