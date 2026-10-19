"""B-scrambled: Baseline B's outline with its scenes shuffled, then fresh prose.

The instrument check's deliberately damaged condition (P1.8). It asks for B's
outline with B's seed, so it gets the very same outline (from the cache when B
ran first), shuffles the scene order with a seeded permutation that is never
the identity, and writes new prose for the shuffled scene list through the
shared prose contract. It never reuses B's scene prose: the shuffled plan
changes every scene prompt. The permutation is seeded by the story's seed and
the brief, so briefs get different orders, and is kept beside the story.
"""

import json
import random
from typing import override

from nde.plan.beats import BeatOutline
from nde.stories.baseline_b import BaselineB
from nde.stories.pipeline import StoryContext, derive_seed

PERMUTATION_FILE = "permutation.json"


def scramble(scenes: int, seed: int) -> list[int]:
    """A seeded permutation of `range(scenes)` that is never the original order.

    `order[i]` is the original index of the scene told in position i.
    """
    if scenes < 2:  # noqa: PLR2004 - one scene has no other order
        raise ValueError("a scramble needs at least two scenes")
    order = list(range(scenes))
    random.Random(derive_seed(seed, "scramble")).shuffle(order)  # noqa: S311 - not cryptography
    if order == sorted(order):
        # Rotate by one: still a deterministic function of the seed, and never the identity.
        order = order[1:] + order[:1]
    return order


class BaselineBScrambled(BaselineB):
    @override
    def plan(self, ctx: StoryContext) -> BeatOutline:
        outline = super().plan(ctx)
        # The brief is part of the seed, so each brief gets its own scene order.
        order = scramble(len(outline.scenes), derive_seed(ctx.seed, ctx.brief.id))
        ctx.keep(PERMUTATION_FILE, json.dumps(order) + "\n")
        return outline.model_copy(update={"scenes": [outline.scenes[i] for i in order]})
