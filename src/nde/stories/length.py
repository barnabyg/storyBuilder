"""Length control: a story must be within a tolerance of its brief's target length.

Length is measured in words on the normalised story. An out-of-range story's
prose is regenerated on a new, deterministic seed, up to a bound. If no attempt
is in range, the attempt closest to the target is kept and flagged.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from nde.stories.normalise import normalise, word_count
from nde.stories.pipeline import derive_seed


@dataclass(frozen=True)
class Draft:
    """One generation of a story's prose."""

    seed: int
    raw: str
    text: str
    words: int
    in_range: bool


def in_range(words: int, target: int, tolerance: float) -> bool:
    return abs(words - target) <= tolerance * target


def attempt_seed(seed: int, attempt: int) -> int:
    """The first attempt uses the story's seed; each regeneration derives a new one."""
    return seed if attempt == 0 else derive_seed(seed, "length", attempt)


def measure(raw: str, *, seed: int, target: int, tolerance: float) -> Draft:
    text = normalise(raw)
    words = word_count(text)
    return Draft(seed, raw, text, words, in_range(words, target, tolerance))


def choose(drafts: Sequence[Draft], target: int) -> int:
    """The first in-range draft, else the one closest to the target (the earliest on ties)."""
    for index, draft in enumerate(drafts):
        if draft.in_range:
            return index
    return min(range(len(drafts)), key=lambda index: abs(drafts[index].words - target))
