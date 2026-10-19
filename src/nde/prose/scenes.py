"""The shared prose contract: scene-by-scene prose from a scene list.

Every structured condition (B, B-scrambled, and later C, D and E) writes its
prose through `write_scenes`, with the same template (`SCENE_PROSE`) on the
same role (`PROSE_ROLE`), so conditions differ only in their plans. Reuse
them unchanged; a new condition supplies its own scene list.

Each scene is written from the brief, the whole plan, the current scene's
beats (plus optional free-text notes, for richer planners) and a compact
summary of the scenes written so far: the planned summaries of the earlier
scenes and the last paragraph of the previous scene. The scenes are joined
with blank lines; normalisation then makes them one continuous story.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from nde.runs.config import PromptRef, RoleName
from nde.stories.normalise import normalise
from nde.stories.pipeline import Step, StoryContext

PROSE_ROLE: RoleName = "prose"
SCENE_PROSE = PromptRef(template="scene_prose", version=1)
SCENE_VARIABLES = frozenset(
    {"brief", "plan", "story_so_far", "scene_number", "scene_count", "scene", "scene_words"}
)


class SceneSpec(BaseModel):
    """One planned scene, as the shared prose step sees it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    beats: list[str] = Field(min_length=1)
    # Anything else the planner wants the prose to realise, as plain text.
    notes: str | None = None


def scene_step(index: int) -> str:
    return f"scene-{index + 1:02d}"


def render_plan(scenes: Sequence[SceneSpec]) -> str:
    return "\n".join(f"{number}. {scene.summary}" for number, scene in enumerate(scenes, 1))


def render_scene(scene: SceneSpec) -> str:
    lines = [f"- {beat}" for beat in scene.beats]
    if scene.notes:
        lines += ["", scene.notes]
    return "\n".join(lines)


def story_so_far(scenes: Sequence[SceneSpec], written: Sequence[str]) -> str:
    """A compact summary of the scenes already written."""
    if not written:
        return "This is the first scene."
    summaries = render_plan(scenes[: len(written)])
    paragraphs = normalise(written[-1]).split("\n\n")
    return (
        f"Scenes written so far:\n{summaries}\n\n"
        f"The previous scene ended:\n{paragraphs[-1].strip()}"
    )


def write_scenes(
    ctx: StoryContext, scenes: Sequence[SceneSpec], *, seed: int, prompt: PromptRef = SCENE_PROSE
) -> str:
    """Write every scene in order with the shared prose template; returns the joined prose."""
    brief = ctx.brief
    plan = render_plan(scenes)
    scene_words = round(brief.target_words / len(scenes))
    written: list[str] = []
    for index, scene in enumerate(scenes):
        variables = {
            "brief": brief.describe(),
            "plan": plan,
            "story_so_far": story_so_far(scenes, written),
            "scene_number": str(index + 1),
            "scene_count": str(len(scenes)),
            "scene": render_scene(scene),
            "scene_words": str(scene_words),
        }
        text = ctx.complete(Step(scene_step(index), PROSE_ROLE, prompt, variables), seed=seed)
        ctx.keep(f"{scene_step(index)}.md", text)
        written.append(text)
    return "\n\n".join(text.strip() for text in written)
