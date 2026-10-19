"""Pydantic response models, registered by the name configs use for them."""

from pydantic import BaseModel

from nde.schema.benchmark import StoryOutline
from nde.schema.brief import BriefDrafts
from nde.schema.smoke import StoryPremise

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "story_premise": StoryPremise,
    "story_outline": StoryOutline,
    "brief_drafts": BriefDrafts,
}
