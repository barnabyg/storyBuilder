"""The beat outline: Baseline B's plan, a title and an ordered list of scenes with beats."""

from functools import cache

from pydantic import BaseModel, ConfigDict, Field, create_model

from nde.prose.scenes import SceneSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutlineScene(_Strict):
    summary: str = Field(min_length=1, description="what happens in the scene, in one sentence")
    beats: list[str] = Field(
        min_length=1, max_length=6, description="the scene's concrete events, in order"
    )


class BeatOutline(_Strict):
    title: str = Field(min_length=1)
    scenes: list[OutlineScene] = Field(min_length=1)

    def scene_specs(self) -> list[SceneSpec]:
        return [SceneSpec(summary=scene.summary, beats=scene.beats) for scene in self.scenes]


@cache
def outline_model(min_scenes: int, max_scenes: int) -> type[BeatOutline]:
    """A `BeatOutline` whose scene count is bounded by a brief's scene range."""
    return create_model(
        "BeatOutline",
        __base__=BeatOutline,
        scenes=(list[OutlineScene], Field(min_length=min_scenes, max_length=max_scenes)),
    )
