"""Pydantic response models, registered by the name configs use for them."""

from pydantic import BaseModel

from nde.schema.smoke import StoryPremise

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "story_premise": StoryPremise,
}
